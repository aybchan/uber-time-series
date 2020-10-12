import numpy as np
import pandas as pd

def mae(y, y_hat):
    return np.nanmean(np.abs(y-y_hat))

def rmse(y, y_hat):
    return np.sqrt(np.nanmean((y-y_hat)**2))

def mape(y, y_hat):
    return np.nanmean(np.abs((y-y_hat)/y))

def smape(y, y_hat):
    return np.nanmean(np.abs(y-y_hat)/ ((y+y_hat)*2))

def recover_dates(df):
    dates = []
    idx = 0
    df_ = df.reset_index()
    for i,traffic in enumerate(df_.iloc[start:end]['traffic_volume'].values):
        if np.isnan(traffic):
            continue
        if ( (int(traffic) == int(Y[idx])) or 
            ((int(traffic)-1) == int(Y[idx])) or 
            ((int(traffic)+1) == int(Y[idx]))):
            dates.append(df_.iloc[start+i].date_time)
            idx += 1
    return np.hstack(dates)

def get_df_with_dates(df, Y, Y_hat):
    dates = recover_dates(Y, df)
    df_fulldates = pd.DataFrame(pd.date_range(*dates[[0,-1]], freq='H'),columns=['date'])
    df_original = pd.DataFrame(np.vstack([dates, Y, Y_hat]).T, columns=['date', 'y','y_pred'])
    df_results = df_original.merge(df_fulldates, how='right').set_index('date')
    return df_results
    