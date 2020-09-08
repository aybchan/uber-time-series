import os
import subprocess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

DATA = 'data'
DATASET = 'Metro_Interstate_Traffic_Volume.csv'

class TrafficDataset(Dataset):
    """
    PyTorch Dataset class for Metro Traffic dataset
    """
    def __init__(self, samples, n_input_steps, key='train'):
        # calculate normalisation parameters for columns
        # `temp`, `rain_1h`, `clouds_all` and `traffic_volume`
        # from training data
        self.X_train = samples['train'][:,:n_input_steps,:].copy()
        
        cols_to_normalise = [0,1,3,4]
        self.train_mu, self.train_sigma  = [], []
        for c in cols_to_normalise:
            self.train_mu.append(np.mean(np.hstack([self.X_train[:,0,c], 
                                               self.X_train[-1,1:,c]])))
            self.train_sigma.append(np.std(np.hstack([self.X_train[:,0,c], 
                                                 self.X_train[-1,1:,c]])))

        # normalise dataset
        self.X = samples[key][:,:n_input_steps,:].copy()
        self.y = samples[key][:,n_input_steps:,:].copy()
        for c,col in enumerate(cols_to_normalise):
            self.X[:,:,col] = (self.X[:,:,col] - self.train_mu[c])/(self.train_sigma[c])
            self.y[:,:,col] = (self.y[:,:,col] - self.train_mu[c])/(self.train_sigma[c])
        
        
    def __len__(self):
        return len(self.y)


    def __getitem__(self, idx):
        x = torch.Tensor(self.X[idx,:,:]).float()
        y = torch.Tensor(self.y[idx,:,4] - self.X[idx,0,4]).float()
        return x, y


def get_datasets(samples, n_input_steps):
    datasets = {}
    for key, sample in samples.items():
        datasets[key] = TrafficDataset(samples, n_input_steps, key)
    
    return datasets
    
    
def get_dataloaders(datasets, train_batch_size):
    dataloaders = {}
    for key, dataset in datasets.items():
        if key == 'train':
            dataloaders[key] = DataLoader(dataset, 
                                          batch_size=train_batch_size, 
                                          shuffle=True)
        else:
            dataloaders[key] = DataLoader(dataset, 
                                          batch_size=len(dataset), 
                                          shuffle=False)

    return dataloaders
    

def pipeline(n_input_steps: int, n_pred_steps: int) -> (pd.DataFrame, dict, dict):
    download() # 1.1.1
    df = pd.read_csv(f'{DATA}/{DATASET}')
    
    df = time_preprocessing(df) # 1.1.3, 1.2
    df = deal_with_anomalies(df) #1.3, 1.3.1, 1.3.2
    df = create_weather_features(df) # 1.4.1
    df = create_holiday_features(df) # 1.4.2
    df = create_time_features(df) # 1.4.3
    df = drop_string_features(df)
    
    split_dfs = split_dataframe(df) # 1.5.1
    samples = create_samples(split_dfs, n_input_steps, n_pred_steps) # 1.5.2

    return df, split_dfs, samples


def download():
    """
    https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
    """
    
    if not os.path.exists(f'{DATA}/{DATASET}'):
        # create data dir
        Path(DATA).mkdir(parents=True, exist_ok=True)

        # download and extract data
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz'
        subprocess.run(['curl', '-o', f'{DATA}/{DATASET}.gz', url])
        subprocess.run(['gzip', '-d', f'{DATA}/{DATASET}.gz'])
        print('Downloaded data')

    else:
        print('Data already downloaded')


def time_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # 1.1.2
    time_col = 'date_time'
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    # 1.1.3
    df = df.iloc[df[time_col].drop_duplicates(keep='last').index]

    # 1.2
    # get the first and last timestamps
    start, end = df.date_time.iloc[[0,-1]].values

    # get a list of hourly timestamps in this range
    full_range = pd.date_range(start, end, freq='H')
    df = pd.DataFrame(full_range, columns=[time_col]).merge(df, on=time_col, how='outer')
    df = df.set_index(time_col)
    
    return df


def deal_with_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    # rain_1h anomaly
    second_largest_rain_1h = df['rain_1h'].sort_values(ascending=False)[1]
    rain_1h_mask = df['rain_1h'] > 5000
    largest_rain_1h_idx = np.where(rain_1h_mask)[0][0]
    df.at[df.iloc[largest_rain_1h_idx].name, 'rain_1h'] = second_largest_rain_1h
    
    # temp anomaly
    temp_mask = df['temp'] < 100
    smallest_temp_idx = np.where(temp_mask)[0][0]
    # get interpolated value
    temp_imputate_value = df['temp'].iloc[[smallest_temp_idx-1,smallest_temp_idx+5]].mean()

    # fill in anomalous values
    for i in range(smallest_temp_idx,smallest_temp_idx+4):
        idx = df.iloc[i].name
        df.at[idx, 'temp'] = temp_imputate_value

    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df['weather_description'], prefix='weather')
    df[dummies.columns] = dummies

    return df


def create_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df['holiday'], prefix='holiday')
    df[dummies.columns] = dummies

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    hour_sin = np.sin(2*np.pi*(df.index.hour.values/24))
    hour_cos = np.cos(2*np.pi*(df.index.hour.values/24))
    df['hour_sin'] = hour_sin
    df['hour_cos'] = hour_cos
    
    weekday_sin = np.sin(2*np.pi*(df.index.weekday.values/7))
    weekday_cos = np.cos(2*np.pi*(df.index.weekday.values/7))
    df['weekday_sin'] = weekday_sin
    df['weekday_cos'] = weekday_cos

    yearweek_sin = np.sin(2*np.pi*(df.index.week.values/52))
    yearweek_cos = np.cos(2*np.pi*(df.index.week.values/52))
    df['yearweek_sin'] = yearweek_sin
    df['yearweek_cos'] = yearweek_cos

    return df


def drop_string_features(df: pd.DataFrame) -> pd.DataFrame:
    str_columns = ['weather_main','weather_description','holiday']
    df = df.drop(str_columns,axis=1)
    
    return df


def split_dataframe(df: pd.DataFrame) -> dict:
    test_start_time  = df.index[-1] - np.timedelta64(30*6,  'D')
    valid_start_time = df.index[-1] - np.timedelta64(30*12, 'D')
    ranges = {
        'train': (df.index[0], valid_start_time),
        'valid': (valid_start_time, test_start_time),
        'test':  (test_start_time, df.index[-1])
    }

    datasets = {}
    time_to_index = lambda time: np.where(df.index == time)[0][0]

    datasets['train'] = df.iloc[:time_to_index(valid_start_time)]
    datasets['valid'] = df.iloc[time_to_index(valid_start_time):
                                time_to_index(test_start_time)]
    datasets['test']  = df.iloc[time_to_index(test_start_time):]

    for key,dataset in datasets.items():
        print(dataset.shape[0], key, 'rows from', ranges[key][0], 'to', ranges[key][1])

    return datasets


def create_samples(datasets: dict, n_input_steps: int, n_pred_steps: int) -> dict:
    data = {}
    for key,dataset in datasets.items():
        dataset = datasets[key]
        n_cols = dataset.shape[1]
        dataset = dataset.values.astype(np.float64)

        idxs = np.arange(dataset.shape[0])
        n_timesteps = n_input_steps + n_pred_steps
        n_samples = dataset.shape[0] - n_timesteps + 1
        stride = idxs.strides[0]
        sample_idxs = np.lib.stride_tricks.as_strided(idxs, shape=(n_samples, n_timesteps), strides=(stride, stride))

        samples = dataset[sample_idxs]
        useable = np.all(~np.isnan(samples.reshape(-1, n_timesteps*n_cols)),axis=-1)
        data[key] = samples[useable]

        print(data[key].shape[0], f'samples of {n_input_steps} input steps and {n_pred_steps} output steps in', key)

    return data