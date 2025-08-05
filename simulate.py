import os
import pandas as pd
import numpy as np
with open('universal_config.json', 'r') as f:
    config = json.load(f)
symbols_dict= config['symbols']
most_frequent=sorted(symbols_dict.items(), key=lambda x: x[1], reverse=True)[:1]
symbols= [x[0] for x in most_frequent]

for symbol in symbols:
    df=pd.read_csv(f'{symbol}_spot_full.csv')

    