import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

spot_parquet_dir = os.path.join('C:/Users/PC/binance_quant/binance', 'spot-1h-0509')

def collect_only(symbol):
    files = [f for f in os.listdir(spot_parquet_dir) if f.endswith('parquet')]
    df_list=[]
    for filename in tqdm(files, desc='Processing parquet files'):
        df = pd.read_parquet(os.path.join(spot_parquet_dir, filename))
        df = df.reset_index()
        if 'base' in df.columns:
            df=df[df['base']==symbol]
            df_list.append(df)
        else:
            print(f"'base' doesn't exist in {filename}")
    df_combined = pd.concat(df_list)
    df_combined.to_csv(f'{symbol}_spot_full.csv')

def plot_ts(data, title=None, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    plt.plot(data)
    plt.grid(True, alpha=0.3)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def calculate_volatility(data, window, annualize=True):
    returns = np.log(data / data.shift(1))
    rolling_std = returns.rolling(window=window).std()
    if annualize:
        rolling_std = rolling_std * np.sqrt(8760)
    return rolling_std