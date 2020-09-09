import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

def get_device() -> str:
    if torch.cuda.is_available():  
        device = 'cuda:0'
    else:  
        device = 'cpu'

    return torch.device(device)

def train(device: str, model: nn.Module, dataloader: DataLoader, params: dict, use_tqdm: bool=False):
    model.train().to(device)
    optimiser = optim.Adam(lr=params['learning_rate'],
                           params=model.parameters())

    loss_fn = F.mse_loss
    losses = []

    epochs = range(params['num_epochs'])
    if use_tqdm:
        from tqdm import tqdm
        epochs = tqdm(epochs)
    
    for epoch in epochs:
        for i,(x,y) in enumerate(dataloader):
            x,y = x.to(device),y.to(device)
            out = model(x)

            optimiser.zero_grad()
            loss = loss_fn(out, y)
            loss.backward()
            optimiser.step()
            
            step =  i * params['batch_size'] + len(x)
            losses.append([epoch * len(dataloader.dataset) + step, loss.item()])

            if use_tqdm:
                epochs.set_description(
                    'Epoch={0} | [{1:>5}|{2}]\tloss={3:.4f}'
                    .format(epoch, step, len(dataloader.dataset),losses[-1][1]))
    
    return model, losses

def evaluate(device: str, model: nn.Module, valid_loader: DataLoader):
    model = model.eval().to(device)
    for i,(x,y) in enumerate(valid_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = F.mse_loss(out, y)
        
    return {'loss': np.float32(loss.cpu().detach().numpy())}

def save(model: nn.Module, name: str, path: str='../model_artifacts'):
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(model, f'{path}/{name}.pt')