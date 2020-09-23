import os
import torch
import numpy as np

from src import utils
import models.variational_dropout as vd
from models.predict import *

cpu = lambda x: x.cpu().detach().numpy()

def dropout_on(m):
    if type(m) in [torch.nn.Dropout, vd.LSTM]:
        m.train()

def dropout_off(m):
    if type(m) in [torch.nn.Dropout, vd.LSTM]:
        m.eval()
        
def load_trained_model(params, device):
    from models.predict import Predict
    # get paths to saved model artifact
    predict_loc = os.path.join(params['paths']['project_home'], params['paths']['artifacts'], 'predict.pt')
    
    # load, turn on evaluate mode    
    predict = torch.load(predict_loc, map_location=device).eval()
    
    return predict.to(device)


def mc_dropout(params, predict, dataloader, device, use_tqdm=True):
    predict = predict.apply(dropout_on)
    
    pbar = range(params['inference']['B'])
    if use_tqdm:
        from tqdm import tqdm
        pbar = tqdm(pbar)

    y_hats = []
    for b in pbar:
        for x,y in dataloader:
            x,y = x.to(device), y.to(device)
            break
        y_hat_b = predict((x, y[:,0,1:]))
        y_hats.append(cpu(y_hat_b))
    
    ymc_hats = np.mean(y_hats, axis=0)
    eta_1s   = np.mean((ymc_hats[:,0] - np.stack(y_hats)[:,:,0])**2, axis=0)
    return ymc_hats, eta_1s


def inference(params, dataloaders, use_tqdm=True):
    device = utils.get_device()
    
    # mc dropout
    predict = load_trained_model(params, device)
    ymc_hats, eta_1s = mc_dropout(params, predict, dataloaders['test'], device)
    
    # inherent noise
    predict.apply(dropout_off)
    for x,y in dataloaders['valid']:
        x,y = x.to(device), y.to(device)
        break
    eta_2sq = np.mean(cpu(predict((x, y[:,0,1:])))[:,0])
    
    # total noise
    etas = np.sqrt(eta_1s + eta_2sq)
    
    return ymc_hats, etas


def rescale_data(dataloaders, mu, eta):
    # get the dataset scaling values to 'reinflate' the machine learning data back
    # to the scale of the original data 
    dataset = dataloaders['test'].dataset
    train_mu = dataset.train_mu[-1]
    train_sigma = dataset.train_sigma[-1]

    # scale the real output data Y and the predicted outputs Y_hat
    Y = (dataset.y[:,0,4]*train_sigma + train_mu)[:-48]
    Y_hat = ((mu[:,0] + dataset.X[:,0,4])*train_sigma + train_mu)[:-48]

    # compute the upper/lower bounds using the uncertainty estimates
    Y_hat_2upper = Y_hat + 2*eta[:-48] * Y_hat
    Y_hat_2lower = Y_hat - 2*eta[:-48] * Y_hat
    Y_hat_upper = Y_hat + eta[:-48] * Y_hat
    Y_hat_lower = Y_hat - eta[:-48] * Y_hat
    
    return Y, Y_hat, Y_hat_2upper, Y_hat_2lower, Y_hat_upper, Y_hat_lower


def run(params, dataloaders):
    # run the inference algorithm, this returns the mean predcitions for the test data
    # and the associated uncertainty estimates
    device = utils.get_device()
    mu,eta = inference(params, dataloaders)
    
    return rescale_data(dataloaders, mu, eta)
