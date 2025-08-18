import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('mse_results.csv')
print(df)
df['kf_vs_garch'] = abs(df['mse_kf'] - df['mse_garch'])/df['mse_garch']
df['pf_vs_garch'] = abs(df['mse_pf'] - df['mse_garch'])/df['mse_garch']
print(df)
