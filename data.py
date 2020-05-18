import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TrafficDataset(Dataset):
    def __init__(self, dataset, set_='train'):
        self.x_train = dataset['train']['X'].copy()
        self.x = dataset[set_]['X'].copy()
        self.y = dataset[set_]['y'].copy()
        
        self.mu = np.mean(self.x.reshape(-1,97),axis=0)[:4]
        self.std = np.std(self.x.reshape(-1,97),axis=0)[:4]
        self.x[:,:,:4] = (self.x[:,:,:4] - self.mu) / self.std
        self.y[:,:,:4] = (self.y[:,:,:4] - self.mu) / self.std
    
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        x = torch.Tensor(self.x[:,:,:][idx]).float()
        y = torch.Tensor(self.y[:,:,3][idx]).float()
        return x, y

def download():
    """
    https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
    https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz
    """

def pipeline():
    df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # one hot encode weather descriptions
    dummies = pd.get_dummies(df['weather_description'],prefix='weather')
    df[dummies.columns] = dummies
    
    # one hot encode holidays
    dummies = pd.get_dummies(df['holiday'],prefix='holiday')
    df[dummies.columns] = dummies
    
    df = df.drop(['weather_main','weather_description','holiday','snow_1h'],axis=1)
    
    df = df.drop_duplicates()
    df = clean(df)
    df = add_date_features(df)
    
    return df

def clean(df):
    # rain_1h anomaly
    df.iloc[24870,1] = 0
    return df

def add_date_features(df):
    """
    Add one hot encoded date/time features
    """
    
    df['hour'] = df['date_time'].dt.hour
    dummies = pd.get_dummies(df['hour'],prefix='hour')
    df[dummies.columns] = dummies
    
    df['month'] = df['date_time'].apply(lambda x: x.month)
    dummies = pd.get_dummies(df['month'],prefix='month')
    df[dummies.columns] = dummies

    df['day_of_week'] = df['date_time'].dt.dayofweek
    dummies = pd.get_dummies(df['day_of_week'],prefix='day_of_week')
    df[dummies.columns] = dummies
    
    df = df.drop(['hour','month','day_of_week'],axis=1)
    return df

def find_idx_months_from_end(df, months):
    """
    Find the dataframe index that is n months from the most recent data
    """
    
    idx = df.date_time[df.date_time > df.date_time.iloc[-1] -np.timedelta64(months,'M')].index[0]
    return idx


def get_idxs(df, time_steps):
    """
    Get starting indices for samples with length `time_steps` with each
    successive time point in the sample being 1 hour apart
    """
    
    array = np.array(np.array(df.date_time.values[1:] - 
                              df.date_time.values[:-1],
                              dtype='timedelta64[s]'),
                     np.float32)
    
    idxs = []
    for i,_ in enumerate(array[:-time_steps]):
        if np.all(array[i:i+25]==3600):
            idxs.append(i)
    return np.array(idxs)


def samples(df, time_steps=24, y_time_steps=12):
    """
    Return the train/valid/test sets
    """
    
    data = df.drop('date_time',axis=1).values

    dataset = dict()
    valid_start_idx = find_idx_months_from_end(df, 12)
    test_start_idx  = find_idx_months_from_end(df, 6)
    
    idxs = get_idxs(df, time_steps+y_time_steps)
    start_idxs = [0, valid_start_idx, test_start_idx, idxs[-1]]

    for i,set_ in enumerate(['train', 'valid', 'test']):    
        dataset[set_] = dict()
        idx = idxs[(idxs>=start_idxs[i]) & (idxs<start_idxs[i+1])]

        x_sample = []
        y_sample = []
        for idx_ in idx:
            datum = (data[idx_:idx_+time_steps+y_time_steps]).copy()
            datum[:,3] -= datum[:,3][0]
            
            x_sample.append(datum[:time_steps])
            y_sample.append(datum[time_steps:time_steps+y_time_steps])
        dataset[set_]['X'] = np.array(x_sample)
        dataset[set_]['y'] = np.array(y_sample)
    
    return dataset