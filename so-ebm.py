import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar
import sys
try: import setGPU
except ImportError: pass
import torch
import model_classes, nets
from constants import *    
import cvxpy as cp
import torch.optim as optim
import math   
import time
import logging
from datetime import datetime


def load_data_with_features(filename):
    tz = pytz.timezone("America/New_York")
    df = pd.read_csv(filename, sep=" ", header=None, usecols=[1,2,3], 
        names=["time","load","temp"])
    df["time"] = df["time"].apply(dt.fromtimestamp, tz=tz)
    df["date"] = df["time"].apply(lambda x: x.date())
    df["hour"] = df["time"].apply(lambda x: x.hour)
    df.drop_duplicates("time", inplace=True)

    # Create one-day tables and interpolate missing entries
    df_load = df.pivot(index="date", columns="hour", values="load")
    df_temp = df.pivot(index="date", columns="hour", values="temp")
    df_load = df_load.transpose().fillna(method="backfill").transpose()
    df_load = df_load.transpose().fillna(method="ffill").transpose()
    df_temp = df_temp.transpose().fillna(method="backfill").transpose()
    df_temp = df_temp.transpose().fillna(method="ffill").transpose()

    holidays = USFederalHolidayCalendar().holidays(
        start='2008-01-01', end='2016-12-31').to_pydatetime()
    holiday_dates = set([h.date() for h in holidays])

    s = df_load.reset_index()["date"]
    data={"weekend": s.apply(lambda x: x.isoweekday() >= 6).values,
          "holiday": s.apply(lambda x: x in holiday_dates).values,
          "dst": s.apply(lambda x: tz.localize(
            dt.combine(x, dt.min.time())).dst().seconds > 0).values,
          "cos_doy": s.apply(lambda x: np.cos(
            float(x.timetuple().tm_yday)/365*2*np.pi)).values,
          "sin_doy": s.apply(lambda x: np.sin(
            float(x.timetuple().tm_yday)/365*2*np.pi)).values}
    df_feat = pd.DataFrame(data=data, index=df_load.index)

    # Construct features and normalize (all but intercept)
    X = np.hstack([df_load.iloc[:-1].values,        # past load
                    df_temp.iloc[:-1].values,       # past temp
                    df_temp.iloc[:-1].values**2,    # past temp^2
                    df_temp.iloc[1:].values,        # future temp
                    df_temp.iloc[1:].values**2,     # future temp^2
                    df_temp.iloc[1:].values**3,     # future temp^3
                    df_feat.iloc[1:].values,        
                    np.ones((len(df_feat)-1, 1))]).astype(np.float64)
    # X[:,:-1] = \
    #     (X[:,:-1] - np.mean(X[:,:-1], axis=0)) / np.std(X[:,:-1], axis=0)

    Y = df_load.iloc[1:].values

    return X, Y

def gaussian_pdf(mean, sig, z):
    
    var = sig**2
    denom = (2*math.pi*var)**0.5
    norm = torch.exp(-(z-mean)**2/(2*var))
    pdf = norm/denom
    return pdf

def gaussian_cdf(mean, sig, z):
    x = (z - mean)/(sig*math.sqrt(2))
    cdf = 0.5 * (1+torch.erf(x))    
    return cdf

def task_loss_expectation(Y_sched, mean, sig, params):
    
    pdf = gaussian_pdf(mean, sig, Y_sched)
    cdf = gaussian_cdf(mean, sig, Y_sched)
    
    loss = (params["gamma_under"]+params["gamma_over"])*((sig**2*pdf) + (Y_sched-mean)*cdf) \
            - params["gamma_under"]*(Y_sched-mean) + 0.5*((Y_sched-mean)**2+sig**2)
    
    return loss


def task_loss(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
            params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 
            0.5 * (Y_sched - Y_actual)**2)
    
    

def langevin_dynamics(model, Z, variables, params, steps=32, step_size=0.1, num_samples=1):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    noise = torch.randn(Z.shape, device=Z.device)
    mean, sig = model(variables['X_train_'])
    Z = Z.repeat(num_samples, 1)
    Z.requires_grad = True
    mean, sig = mean.repeat(num_samples, 1), sig.repeat(num_samples, 1)
    for _ in range(steps):
        
        noise.normal_(0, 0.01)
        Z.data.add_(noise.data)
        out_Z = task_loss_expectation(Z, mean, sig, params).sum(1).mean()
        out_Z.backward()
        
        Z.data.add_(-step_size * Z.grad.data)
        Z.grad.detach_()
        Z.grad.zero_()
        
        
    for p in model.parameters():
        p.requires_grad = True   
    model.train()

    return Z

def gauss_density_centered(x, std):
    return torch.exp(-0.5*(x / std)**2) / (math.sqrt(2*math.pi)*std)

def gmm_density_centered(x, std):
    """
    Assumes dim=-1 is the component dimension and dim=-2 is feature dimension. Rest are sample dimension.
    """
    if x.dim() == std.dim() - 1:
        x = x.unsqueeze(-1)
    elif not (x.dim() == std.dim() and x.shape[-1] == 1):
        raise ValueError('Last dimension must be the gmm stds.')
    return gauss_density_centered(x, std).prod(-2).mean(-1)

def sample_gmm_centered(std, num_samples=1):
    num_components = std.shape[-1]
    num_dims = std.numel() // num_components

    std = std.view(1, num_dims, num_components)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0,:,k].t()

    # Sample
    x_centered = std_samp * torch.randn(num_samples, num_dims)
    prob_dens = gmm_density_centered(x_centered, std)

    return x_centered, prob_dens


def main():

    ##load dataset
    parser = argparse.ArgumentParser(
        description='Run electricity scheduling task net experiments.')
    parser.add_argument('--save', type=str, 
        metavar='save-folder', help='prefix to add to save path')
    parser.add_argument('--training_method', type=str, choices=['langevin'], default='langevin',
        help='training method for so-ebm')
    parser.add_argument('--nRuns', type=int, default=10,
        metavar='runs', help='number of runs')
    parser.add_argument('--epochs', type=int, default=50,
        metavar='epochs', help='number of epochs')
    parser.add_argument('--steps', type=int, default=32,
        metavar='steps', help='number of steps of legevin dynamics')
    parser.add_argument('--step_size', type=float, default=0.1,
        metavar='step_size', help='step size in legevin dynamics')
    parser.add_argument('--num_samples_mle', type=int, default=1,
        metavar='num_samples', help='number of samples for negative samples in MLE')
    parser.add_argument('--num_samples_kld', type=int, default=128,
        metavar='num_samples', help='number of samples for negative samples in kld')
    parser.add_argument('--lr', type=float, default=1e-4,
        metavar='lr', help='learning rate of the optimizer')
    args = parser.parse_args()
    
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    logging.basicConfig(filename='{}.log'.format(run_id), filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logging.info(sys.argv[0])
    logging.info('learning rate: {}'.format(args.lr))
    
    X1, Y1 = load_data_with_features('data/pjm_load_data_2008-11.txt')
    X2, Y2 = load_data_with_features('data/pjm_load_data_2012-16.txt')

    X = np.concatenate((X1, X2), axis=0)
    X[:,:-1] = \
        (X[:,:-1] - np.mean(X[:,:-1], axis=0)) / np.std(X[:,:-1], axis=0)
    
    Y = np.concatenate((Y1, Y2), axis=0)
    
    n_tt = int(len(X) * 0.8)
    X_train, Y_train = X[:n_tt], Y[:n_tt]
    X_test, Y_test = X[n_tt:], Y[n_tt:]
    
    params = {"n": 24, "c_ramp": 0.4, "gamma_under": 50, "gamma_over": 0.5}
    #genrate optimization parameters
    D = np.eye(params['n'] - 1, params['n']) - np.eye(params['n'] - 1,  params['n'], 1)
    G = np.vstack([D,-D]) #46*24
    h = params['c_ramp'] * np.ones((params['n'] - 1)*2) 
    ## obtain optimal solution for training data, TODO: use cvxpylayers for batch optimization
    for i in range(Y_train.shape[0]):
        Y_actual = Y_train[i]
        z = cp.Variable(params['n'])
        obj = cp.Minimize(cp.sum(params["gamma_under"]*cp.pos(Y_actual-z)+params["gamma_over"]*cp.pos(z-Y_actual)+0.5 * (z - Y_actual)**2))
        cons = [G@z <= h]
        prob = cp.Problem(obj, cons)
        prob.solve()
        if i == 0:
            Z_train = z.value
        else:
            Z_train = np.vstack((Z_train,z.value))
    
    ZZ = Z_train
    
    # Construct tensors (without intercepts).
    X_train_ = torch.tensor(X_train[:,:-1], dtype=torch.float, device=DEVICE)
    Y_train_ = torch.tensor(Y_train, dtype=torch.float, device=DEVICE)
    X_test_ = torch.tensor(X_test[:,:-1], dtype=torch.float, device=DEVICE)
    Y_test_ = torch.tensor(Y_test, dtype=torch.float, device=DEVICE)
    
    base_save = 'results' if args.save is None else '{}-results'.format(args.save)
    test_loss_list = []
    val_loss_list = []
    for run in range(args.nRuns):
        #run = run+1
        logging.info('{}-th run'.format(run))
        
        np.random.seed(run)  # numpy random generator
        torch.manual_seed(run)
        torch.cuda.manual_seed_all(run)
        
        save_folder = os.path.join(base_save, str(run))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # Randomly construct hold-out set for task net training.
        
        Z_train = ZZ
        th_frac = 0.8
        inds = np.load('train_val_split.npy')
        #inds = np.random.permutation(X_train.shape[0])
        train_inds = inds[ :int(X_train.shape[0] * th_frac)]
        hold_inds = inds[int(X_train.shape[0] * th_frac):]
        X_train2, X_hold2 = X_train[train_inds, :], X_train[hold_inds, :]
        Y_train2, Y_hold2 = Y_train[train_inds, :], Y_train[hold_inds, :]
        Z_train, Z_hold = Z_train[train_inds, :], Z_train[hold_inds, :]
        
        
        X_train2_ = torch.tensor(X_train2[:,:-1], dtype=torch.float32, device=DEVICE)
        Y_train2_ = torch.tensor(Y_train2, dtype=torch.float32, device=DEVICE)
        Z_train_ = torch.tensor(Z_train, dtype=torch.float32, device=DEVICE)
        X_hold2_ = torch.tensor(X_hold2[:,:-1], dtype=torch.float32, device=DEVICE)
        Y_hold2_ = torch.tensor(Y_hold2, dtype=torch.float32, device=DEVICE)
        Z_hold_ = torch.tensor(Z_hold, dtype=torch.float32, device=DEVICE)
        
        
        # load pre-trained two-stage model
        model = model_classes.Net(X_train2[:,:-1], Y_train2, [200, 200])
        model.load_state_dict(torch.load(os.path.join(save_folder, 'two-stage_model')))
        if USE_GPU:
            model = model.cuda()
        model.eval()
        #construct optimizer and solver
        opt = optim.Adam(model.parameters(), lr=args.lr)

        
        solver = model_classes.SolveScheduling(params)
        
        
        mu_pred_test, sig_pred_test = model(X_test_)
        Z_test = solver(mu_pred_test.double(), sig_pred_test.double())
        Z_test = Z_test.detach().cpu().numpy()
        Z_test_ = torch.tensor(Z_test, dtype=torch.float32, device=DEVICE)
        
        
        mu_pred_hold, sig_pred_hold = model(X_hold2_)
        Z_hold = solver(mu_pred_hold.double(), sig_pred_hold.double())
        Z_hold = Z_hold.detach().cpu().numpy()
        Z_hold_ = torch.tensor(Z_hold, dtype=torch.float32, device=DEVICE)
        
        #test_loss = task_loss(
        #Z_test_, Y_test_, params)
        #print(test_loss.sum())
        print('##############')
        ## get the prediected Z  
        with torch.no_grad():
            mu, sig = model(X_train2_)
            Z_init = solver(mu.double(), sig.double())
          
        variables = {'X_train_': X_train2_, 'Y_train_': Y_train2_, 
        'X_hold_': X_hold2_, 'Y_hold_': Y_hold2_,
        'X_test_': X_test_, 'Y_test_': Y_test_,
        'Z_train_': Z_train_, 'Z_init_': Z_init, 'Z_hold_': Z_hold_, 'Z_test_': Z_test_
        }

        stds = torch.zeros((1, 3))
        stds[0, 0] = 0.1
        stds[0, 1] = 0.05
        stds[0, 2] = 0.02
        logging.info('stds: {}, num of samples: {}'.format(stds, args.num_samples_kld))
               


        prev_min = 1000
        for epoch in range(args.epochs):
            prev_time = time.time()
            opt.zero_grad()
            model.train()
            
            #obtain samples 
            mu_pred, sig_pred = model(variables['X_train_'])
            
            mu_pred, sig_pred = model(variables['X_train_'])           
            
            # (shape: batch_size*num_samples, 24)     
            E_gt = task_loss_expectation(variables['Z_train_'], mu_pred, sig_pred, params).sum(1)
            negative_samples = langevin_dynamics(model, variables['Z_train_'], variables, params, args.num_samples_mle)

            E_model = task_loss_expectation(negative_samples, mu_pred, sig_pred, params).sum(1)
            MLE = E_gt.mean(0) - E_model.mean(0)
            
            Z_samples_zero, q_Z_samples = sample_gmm_centered(stds, num_samples=args.num_samples_kld*24)
            Z_samples_zero = Z_samples_zero.to(DEVICE) # (shape: (num_samples*24,1))
            Z_samples_zero = Z_samples_zero.view(1, args.num_samples_kld, 24) #(shape: (1, num_samples, 24))
            
            q_Z_samples = q_Z_samples.view(1, args.num_samples_kld, 24)
            
            Z_s = variables['Z_train_'].view(-1,1, 24) #(shape: (batch_size,1, 24))
            Z_samples = Z_s + Z_samples_zero # (shape: (batch_size, num_samples,24))
            q_Z_samples = q_Z_samples*torch.ones(Z_samples.size())
            q_Z_samples = q_Z_samples.to(DEVICE) # (shape:(batch_size, num_samples, 24))
            
            Y_s = variables['Y_train_']
            #(shape: (batch_size, num_samples, 24))
            Y_train_expand = Y_s[:,None,:].expand(variables['Y_train_'].shape[0], args.num_samples_kld, 24)
            #(shape: (batch_size*num_samples,24))
            p_Z_samples = task_loss(Z_samples.view(-1,24), Y_train_expand.reshape(-1,24), params)
            #p_Z_samples = torch.exp(-p_Z_samples)
            p_Z_samples = p_Z_samples.view(-1, args.num_samples_kld, 24)
            
            
            mu_pred_expand = mu_pred[:,None,:].expand(mu_pred.shape[0], args.num_samples_kld, 24)
            sig_pred_expand = sig_pred[:,None,:].expand(sig_pred.shape[0], args.num_samples_kld, 24)
            
            
            scores_samples = task_loss_expectation(Z_samples.view(-1,24), mu_pred_expand.reshape(-1,24), sig_pred_expand.reshape(-1,24), params)
            scores_samples = scores_samples.view(-1, args.num_samples_kld, 24)
            
            # weight_kld = p_Z_samples/q_Z_samples
            # weight_kld = weight_kld/torch.sum(weight_kld, dim=1, keepdim=True)
            
            q_Z_samples = torch.prod(q_Z_samples, dim=2)
            weight_kld = p_Z_samples.sum(2)/q_Z_samples
            weight_kld = weight_kld/torch.sum(weight_kld, dim=1, keepdim=True)
            
            #E_true_posterior = torch.sum(weight_kld*scores_samples, dim=1).sum(1)
            E_true_posterior = torch.sum(weight_kld*scores_samples.sum(2), dim=1)
            KLD = E_true_posterior.mean(0) - E_model.mean(0)
            
            loss = MLE + 0.01*KLD                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            loss.backward()
            opt.step()
            
            logging.info('Training time for this epoch: {:.3f}'.format(time.time()-prev_time))
            logging.info('epoch: {}, training_loss: {:.3f}, time:{:.3f}'.format(epoch, loss.item(), time.time()-prev_time))

            
            if (epoch+1) % 1 == 0:
            # evaluate the model
                print('####### evaluating ############')
                model.eval()
                with torch.no_grad():
                    mu_pred_test, sig_pred_test = model(variables['X_test_'])
                    mu_pred_hold, sig_pred_hold = model(variables['X_hold_'])
                
                Z_pred_hold = solver(mu_pred_hold.double(), sig_pred_hold.double())
                hold_loss = task_loss(Z_pred_hold, variables['Y_hold_'], params).mean(0)
                
                Z_pred_test = solver(mu_pred_test.double(), sig_pred_test.double())
                test_loss = task_loss(Z_pred_test, variables['Y_test_'], params).mean(0)
                
                if hold_loss.sum().item() < prev_min:
                    prev_min = hold_loss.sum().item()
                    best_test = test_loss.sum().item()

                logging.info('epoch: {}, test loss : {}, val loss: {}'.format(epoch, test_loss.sum().item(), hold_loss.sum().item()))
    
        test_loss_list.append(best_test)
        val_loss_list.append(prev_min)
    test_loss_list = np.array(test_loss_list)
    val_loss_list = np.array(val_loss_list)
    logging.info('Final test mean: {}, std: {}'.format(test_loss_list.mean(), test_loss_list.std()))
    logging.info('Final val mean: {}, std: {}'.format(val_loss_list.mean(), val_loss_list.std()))

if __name__=='__main__':
    main()          