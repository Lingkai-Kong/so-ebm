#/usr/bin/env python3

import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
import math
import model_classes
from constants import *
import torch.distributions.normal as normal
import cvxpy as cp
#from cvxpylayers.torch import CvxpyLayer

def gaussian_pdf(mean, sig, z):
    
    var = float(sig)**2
    denom = (2*math.pi*var)**0.5
    norm = torch.exp(-(float(z)-float(mean))**2/(2*var))
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
            0.5 * (Y_sched - Y_actual)**2).mean(0)

def task_loss_no_mean(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
        params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 
        0.5 * (Y_sched - Y_actual)**2)

def rmse_loss(mu_pred, Y_actual):
    return ((mu_pred - Y_actual)**2).mean(dim=0).sqrt().data.cpu().numpy()

def nll_loss(mu_pred, sig_pred, Y_actual):
    log_likelihood = normal.Normal(mu_pred, sig_pred).log_prob(Y_actual)
    return -log_likelihood.mean(0)


def run_nll_net(model, variables, X_train, Y_train):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    # For early stopping
    prev_min = 0
    hold_costs = []
    model_states = []
    num_stop_rounds = 20

    for i in range(3000):
        t = time.time()
        opt.zero_grad()
        model.train()
        mu_pred, sig_pred = model(variables['X_train_'])
        train_loss = nll_loss(mu_pred, sig_pred, variables['Y_train_'])
        train_loss.sum().backward()
        opt.step()

        model.eval()
        
        mu_pred, sig_pred = model(variables['X_hold_'])
        hold_loss = nll_loss(mu_pred, sig_pred, variables['Y_hold_'])
        
        mu_pred, sig_pred = model(variables['X_test_'])
        test_loss = nll_loss(mu_pred, sig_pred, variables['Y_test_'])

        print(i, train_loss.sum().item(), hold_loss.sum().item(), test_loss.sum().item())
        print('time: {}'.format(time.time()-t))
        # Early stopping
        hold_costs.append(hold_loss.sum().item())
        model_states.append(model.state_dict().copy())
        if i > 0 and i % num_stop_rounds == 0:
            idx = hold_costs.index(min(hold_costs))
            if prev_min == hold_costs[idx]:
                model.eval()
                best_model = model_classes.Net(
                    X_train[:,:-1], Y_train, [200, 200])
                best_model.load_state_dict(model_states[idx])
                if USE_GPU:
                    best_model = best_model.cuda()
                return best_model
            else:
                prev_min = hold_costs[idx]
                hold_costs = [prev_min]
                model_states = [model_states[idx]]

    return model


def run_task_net(model, variables, params, X_train, Y_train):
    opt = optim.Adam(model.parameters(), lr=1e-5)
    solver = model_classes.SolveScheduling(params)

    # For early stopping
    prev_min = 0
    hold_costs = []
    model_states = []
    num_stop_rounds = 20

    for i in range(1000):
        print('Epoch: {}'.format(i))
        t = time.time()
        opt.zero_grad()
        model.train()
        mu_pred_train, sig_pred_train = model(variables['X_train_'])
        Y_sched_train = solver(mu_pred_train.double(), sig_pred_train.double())
        train_loss = task_loss(
            Y_sched_train.float(),variables['Y_train_'], params)
        train_loss.sum().backward()

        print('Training time for this epoch: {:0.3f}'.format(time.time()-t))
        model.eval()
        mu_pred_test, sig_pred_test = model(variables['X_test_'])
        Y_sched_test = solver(mu_pred_test.double(), sig_pred_test.double())
        test_loss = task_loss(
            Y_sched_test.float(), variables['Y_test_'], params)

        mu_pred_hold, sig_pred_hold = model(variables['X_hold_'])
        Y_sched_hold = solver(mu_pred_hold.double(), sig_pred_hold.double())
        hold_loss = task_loss(
            Y_sched_hold.float(), variables['Y_hold_'], params)

        opt.step()

        print('Training loss: {:0.3f}, Test loss: {:0.3f}, validation loss: {:0.3f}'.format(train_loss.sum().item(), test_loss.sum().item(), 
            hold_loss.sum().item()))
        

        # Early stopping
        hold_costs.append(hold_loss.sum().item())
        model_states.append(model.state_dict().copy())
        if i > 0 and i % num_stop_rounds == 0:
            idx = hold_costs.index(min(hold_costs))
            if prev_min == hold_costs[idx]:
                model.eval()
                best_model = model_classes.Net(
                    X_train[:,:-1], Y_train, [200, 200])
                best_model.load_state_dict(model_states[idx])
                if USE_GPU:
                    best_model = best_model.cuda()
                return best_model
            else:
                prev_min = hold_costs[idx]
                hold_costs = [prev_min]
                model_states = [model_states[idx]]

    return model



def eval_net(model, variables, params):
    solver = model_classes.SolveScheduling(params)

    model.eval()
    mu_pred_test, sig_pred_test = model(variables['X_test_'])

    Y_sched_test = solver(mu_pred_test.double(), sig_pred_test.double())
    test_loss_task = task_loss(
        Y_sched_test.float(), variables['Y_test_'], params)
    
    #print('####test_task_loss: {}'.format(test_loss_task.sum()))

    return test_loss_task.detach().cpu().numpy()

