import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
import importlib
import json
from utility import collect_only
with open('universal_config.json', 'r') as f:
    config = json.load(f)
symbols_dict= config['symbols']
most_frequent=sorted(symbols_dict.items(), key=lambda x: x[1], reverse=True)[:5]
symbols= [x[0] for x in most_frequent]

for symbol in symbols:
    if not os.path.exists(f'{symbol}_spot_full.csv'):
        collect_only(symbol)

