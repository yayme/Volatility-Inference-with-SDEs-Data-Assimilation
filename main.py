import pandas as pd
from src.pipeline import run_pipeline

if __name__ == "__main__":
    df = pd.read_csv("data/BTC_spot_full.csv")
    print(df.head())
    run_pipeline(df) 