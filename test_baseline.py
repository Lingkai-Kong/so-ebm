import argparse
import setproctitle
from models import RegressionFNP
import os
import pandas as pd
import numpy as np

from datetime import datetime as dt
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar

try: import setGPU
except ImportError: pass

import torch

import model_classes, nets, plot
from constants import *


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
    Y = df_load.iloc[1:].values

    return X, Y

def main():
    parser = argparse.ArgumentParser(
    description='Run electricity scheduling task net experiments.')
    parser.add_argument('--save', type=str, 
        metavar='save-folder', help='prefix to add to save path')
    parser.add_argument('--nRuns', type=int, default=5,
        metavar='runs', help='number of runs')
    parser.add_argument('--ratio', type=float, default=1.,
        metavar='ratio', help='ratio of training data')
    args = parser.parse_args()
    
    X1, Y1 = load_data_with_features('pjm_load_data_2008-11.txt')
    X2, Y2 = load_data_with_features('pjm_load_data_2012-16.txt')

    X = np.concatenate((X1, X2), axis=0)
    X[:,:-1] = \
        (X[:,:-1] - np.mean(X[:,:-1], axis=0)) / np.std(X[:,:-1], axis=0)
    
    Y = np.concatenate((Y1, Y2), axis=0)

    # Train, test split.
    n_tt = int(len(X) * 0.8)
    X_train, Y_train = X[:n_tt], Y[:n_tt]
    X_test, Y_test = X[n_tt:], Y[n_tt:]
    
    # Construct tensors (without intercepts).
    X_train_ = torch.tensor(X_train[:,:-1], dtype=torch.float, device=DEVICE)
    Y_train_ = torch.tensor(Y_train, dtype=torch.float, device=DEVICE)
    X_test_ = torch.tensor(X_test[:,:-1], dtype=torch.float, device=DEVICE)
    Y_test_ = torch.tensor(Y_test, dtype=torch.float, device=DEVICE)
    
    
    # Randomly construct hold-out set for task net training.
    th_frac = 0.8
    #inds = np.random.permutation(X_train.shape[0])
    inds = np.load("train_val_split.npy")
    train_inds = inds[ :int(X_train.shape[0] * th_frac)]
    hold_inds = inds[int(X_train.shape[0] * th_frac):]
    X_train2, X_hold2 = X_train[train_inds, :], X_train[hold_inds, :]
    Y_train2, Y_hold2 = Y_train[train_inds, :], Y_train[hold_inds, :]
    X_train2_ = torch.tensor(X_train2[:,:-1], dtype=torch.float32, device=DEVICE)
    Y_train2_ = torch.tensor(Y_train2, dtype=torch.float32, device=DEVICE)
    X_hold2_ = torch.tensor(X_hold2[:,:-1], dtype=torch.float32, device=DEVICE)
    Y_hold2_ = torch.tensor(Y_hold2, dtype=torch.float32, device=DEVICE)
    variables = {'X_train_': X_train2_, 'Y_train_': Y_train2_, 
            'X_hold_': X_hold2_, 'Y_hold_': Y_hold2_,
            'X_test_': X_test_, 'Y_test_': Y_test_}
    base_save = 'results' if args.save is None else '{}-results'.format(args.save)


    # Generation scheduling problem params.
    params = {"n": 24, "c_ramp": 0.4, "gamma_under": 50, "gamma_over": 0.5}
    
    nll_base_list = []
    task_base_list = []
    nll_task_list = []
    task_task_list = []

    print('#######################################')
    for run in range(1):
        save_folder = os.path.join(base_save, str(run))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        model_nll = model_classes.Net(X_train2[:,:-1], Y_train2, [200, 200])
        if USE_GPU:
            model_nll = model_nll.cuda()
        model_nll.load_state_dict(torch.load(os.path.join(save_folder, 'two-stage')))
        nll_base, task_base = nets.eval_net("nll_net", model_nll, variables, params, save_folder)
        
        nll_base_list.append(nll_base.sum())
        task_base_list.append(task_base.sum())
        
        model_task = model_classes.Net(X_train2[:,:-1], Y_train2, [200, 200])
        if USE_GPU:
            model_task = model_task.cuda()
        model_task.load_state_dict(torch.load(os.path.join(save_folder, 'DFL-model')))
        nll_task, task_task = nets.eval_net("task_net", model_task, variables, params, save_folder)
    
        nll_task_list.append(nll_task.sum())
        task_task_list.append(task_task.sum())
    
    
    nll_base_list = np.array(nll_base_list)
    task_base_list = np.array(task_base_list)
    
    nll_task_list = np.array(nll_task_list)
    task_task_list = np.array(task_task_list)


    print('Vanillia neural network NLL: {}, Task: {}'.format(nll_base_list.mean(), task_base_list.mean()))
    print('Task based neural network NLL: {}, Task: {}'.format(nll_task_list.mean(), task_task_list.mean()))

    print('Vanillia neural network NLL: {}, Task: {}'.format(nll_base_list.std(), task_base_list.std()))
    print('Task based neural network NLL: {}, Task: {}'.format(nll_task_list.std(), task_task_list.std()))

if __name__=='__main__':
    main()
        
        
        