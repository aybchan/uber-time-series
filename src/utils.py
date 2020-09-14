import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from models.encoder_decoder_dropout import * 
from src import data

def get_device() -> str:
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    return torch.device(device)

def train(device: str, model: nn.Module, dataloader: DataLoader, params: dict, use_tqdm: bool=False, validate: DataLoader=None):
    model.to(device)
    optimiser = optim.Adam(lr=params['learning_rate'],
                           params=model.parameters())

    loss_fn = F.mse_loss
    losses = {}
    losses['train'] = []
    if validate is not None:
        losses['valid'] = []
    valid_loss = np.nan

    epochs = range(params['num_epochs'])
    if use_tqdm:
        from tqdm import tqdm
        epochs = tqdm(epochs)

    for epoch in epochs:
        model.train()
        for i,(x,y) in enumerate(dataloader):
            x,y = x.to(device),y.to(device)
            out = model(x)

            optimiser.zero_grad()
            loss = loss_fn(out, y)
            loss.backward()
            optimiser.step()

            step = i * params['batch_size'] + len(x)
            losses['train'].append([epoch * len(dataloader.dataset) + step, loss.item()])

            if use_tqdm:
                epochs.set_description(
                    'Epoch={0} | [{1:>5}|{2}]\ttrain. loss={3:.4f}\tvalid. loss={4:.4f}'
                    .format(epoch, step, len(dataloader.dataset),losses['train'][-1][1],valid_loss))
        if validate is not None:
            valid_loss = evaluate(device, model, validate)['loss']
            losses['valid'].append([epoch * len(dataloader.dataset) + step, valid_loss])


    return model, losses

def evaluate(device: str, model: nn.Module, valid_loader: DataLoader):
    loss_fn = F.mse_loss
    model = model.eval().to(device)
    for i,(x,y) in enumerate(valid_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)

    return {'loss': np.float32(loss.cpu().detach().numpy())}

def train_evaluate(params, datasets):
    in_features = 58
    device = get_device()
    model = VDEncoderDecoder(in_features=in_features, 
                             output_steps=params.get('n_output_steps', 12),
                             p=params.get('variational_dropout_p')).to(device)
    dataloaders = data.get_dataloaders(datasets, params.get('batch_size'))
    in_features = dataloaders['train'].dataset.X.shape[-1]
    trained_model,_ = train(device=device, model=model, dataloader=dataloaders['train'], params=params, use_tqdm=True)
    return evaluate(device=device, model=trained_model, valid_loader=dataloaders['valid'])

def save(model: nn.Module, name: str, path: str='../model_artifacts'):
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(model, f'{path}/{name}.pt')
    print(f'PyTorch model saved at {path}/{name}.pt')

def read_json_params(path):
    with open(path) as json_file:
        params = json.load(json_file)
    return params

def hyperparameter_search(n_jobs: int, params:dict, name: str='hyperparameter_search'):
    _, _, samples = data.pipeline(params['data']['n_input_steps'], params['data']['n_output_steps'], params['paths']['data'])
    datasets = data.get_datasets(samples, params['data']['n_input_steps'])

    # set up ax
    from ax.service.ax_client import AxClient
    ax_client = AxClient(enforce_sequential_optimization=False)

    # define hyperparameter bounds 
    ax_client.create_experiment(
        name=name,
        parameters=[
            {"name": "num_epochs", "type": "range", "bounds": [150, 200]},
            {"name": "learning_rate", "type": "range", "bounds": [5e-4, 1e-3], "log_scale": True},
            {"name": "batch_size", "type": "range", "bounds": [64, 1024]},
            {"name": "variational_dropout_p", "type": "range", "bounds": [0.2,0.5]}
        ],
        objective_name='loss',
        minimize=True
    )

    for job in range(n_jobs):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=train_evaluate(parameters, datasets=datasets)['loss'])
    
    print(f'Best parameters found after {n_jobs}:')
    print(ax_client.get_best_parameters())
    ax_client.save_to_json_file()

