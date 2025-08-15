
# Volatility-Inference-with-SDEs-Data-Assimilation

## Project Structure and File Descriptions

1. **main.py**
   - Main script for running simulations and data assimilation experiments on financial time series.

2. **DA_utility.py**
   - Data assimilation (DA) methods and utilities: rolling volatility, naive DA, generic DA, Kalman filter, and particle filter implementations.

3. **simulator.py**
   - Simulation functions for standard Brownian motion, GARCH(1,1), and Heston-lite models.

4. **utility.py**
   - Utility functions for data collection and plotting.

5. **data_collection.py**
   - Script for collecting and preparing spot price data from raw sources.

6. **requirements.txt**
   - Python dependencies required for the project.

7. **universal_config.json**
   - Configuration file for symbols and other universal settings.

8. **BTC_spot_full.csv, ETH_spot_full.csv, BNB_spot_full.csv, XRP_spot_full.csv, TRX_spot_full.csv**
   - Spot price data for various cryptocurrencies.

9. **Simulation_sandbox.ipynb**
   - Jupyter notebook for prototyping and experimenting with simulation and DA methods.

10. **sandbox.py**
    - Temporary or scratch file for quick experiments .

12. **notebooks/**
    - Directory for additional Jupyter notebooks .
