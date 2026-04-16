import numpy as np
import pandas as pd
from DA_utility import (
    compute_rolling_volatility,
    naive_DA,
    heston_predictor,
    generic_DA,
    naive_combiner,
    kalman_DA,
    particle_filter_DA
)
from arch import arch_model
from sklearn.metrics import mean_squared_error, r2_score
import os

def calculate_mse_for_symbol(symbol):
    """
    Calculate MSE values for all volatility estimation methods for a given symbol.
    
    Args:
        symbol (str): The cryptocurrency symbol (e.g., 'XRP', 'BTC', 'ETH', etc.)
    
    Returns:
        dict: Dictionary containing MSE values for each method
    """
    print(f"Processing {symbol}...")
    
    # Load data
    df = pd.read_csv(f'{symbol}_spot_full.csv')
    df['return'] = df['bam_close'].pct_change().fillna(0)
    df['sigma_obs'] = compute_rolling_volatility(df['return'], window=100)
    
    # --- Data Assimilation (DA) methods ---
    # Naive DA
    sigma_model_naive, sigma_prior_naive, sigma_est_naive = generic_DA(
        df['sigma_obs'], predictor=heston_predictor, combiner=naive_combiner
    )
    df['sigma_model_naive'] = sigma_model_naive
    df['sigma_prior_naive'] = sigma_prior_naive
    df['sigma_est_naive'] = sigma_est_naive
    
    # Kalman Filter DA
    sigma_model_kf, sigma_prior_kf, sigma_est_kf = kalman_DA(
        df['sigma_obs'], Q=0.01, R=0.1  # process noise, measurement noise
    )
    df['sigma_model_kf'] = sigma_model_kf
    df['sigma_prior_kf'] = sigma_prior_kf
    df['sigma_est_kf'] = sigma_est_kf
    
    # Particle Filter DA
    sigma_model_pf, sigma_prior_pf, sigma_est_pf = particle_filter_DA(
        df['sigma_obs'], N_particles=100
    )
    df['sigma_model_pf'] = sigma_model_pf
    df['sigma_prior_pf'] = sigma_prior_pf
    df['sigma_est_pf'] = sigma_est_pf
    
    # GARCH(1,1)
    returns = df['return'] * 100
    garch = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
    res = garch.fit(last_obs=100, disp='off')
    forecasts = res.forecast(horizon=1, start=100)
    sigma_garch_full = np.full(len(returns), np.nan)
    sigma_garch_full[:100] = res.conditional_volatility[:100] / 100
    sigma_garch_full[100:] = np.sqrt(forecasts.variance['h.1'].values) / 100
    df['sigma_garch'] = sigma_garch_full
    
    # GARCH(2,2)
    garch22 = arch_model(returns, vol='Garch', p=2, q=2, dist='normal')
    res22 = garch22.fit(last_obs=100, disp='off')
    forecasts22 = res22.forecast(horizon=1, start=100)
    sigma_garch22_full = np.full(len(returns), np.nan)
    sigma_garch22_full[:100] = res22.conditional_volatility[:100] / 100
    sigma_garch22_full[100:] = np.sqrt(forecasts22.variance['h.1'].values) / 100
    df['sigma_garch22'] = sigma_garch22_full
    
    # --- MSE CALCULATION ---
    m1 = (~np.isnan(df['sigma_obs'])) & (~np.isnan(df['sigma_prior_naive']))
    mse_naive = mean_squared_error(df['sigma_obs'][m1], df['sigma_prior_naive'][m1])
    
    m2 = (~np.isnan(df['sigma_obs'])) & (~np.isnan(df['sigma_prior_kf']))
    mse_kf = mean_squared_error(df['sigma_obs'][m2], df['sigma_prior_kf'][m2])
    
    m3 = (~np.isnan(df['sigma_obs'])) & (~np.isnan(df['sigma_prior_pf']))
    mse_pf = mean_squared_error(df['sigma_obs'][m3], df['sigma_prior_pf'][m3])
    
    m4 = (~np.isnan(df['sigma_obs'])) & (~np.isnan(df['sigma_garch']))
    mse_garch = mean_squared_error(df['sigma_obs'][m4], df['sigma_garch'][m4])
    
    m5 = (~np.isnan(df['sigma_obs'])) & (~np.isnan(df['sigma_garch22']))
    mse_garch22 = mean_squared_error(df['sigma_obs'][m5], df['sigma_garch22'][m5])
    
    # --- R² CALCULATION ---
    r2_naive = r2_score(df['sigma_obs'][m1], df['sigma_prior_naive'][m1])
    r2_kf = r2_score(df['sigma_obs'][m2], df['sigma_prior_kf'][m2])
    r2_pf = r2_score(df['sigma_obs'][m3], df['sigma_prior_pf'][m3])
    r2_garch = r2_score(df['sigma_obs'][m4], df['sigma_garch'][m4])
    r2_garch22 = r2_score(df['sigma_obs'][m5], df['sigma_garch22'][m5])
    
    print(f"{symbol} MSE Results:")
    print(f"Naive DA:       {mse_naive:.20f}")
    print(f"Kalman DA:      {mse_kf:.20f}")
    print(f"Particle Filter:{mse_pf:.20f}")
    print(f"GARCH(1,1):     {mse_garch:.20f}")
    print(f"GARCH(2,2):     {mse_garch22:.20f}")
    print(f"\n{symbol} R² Results:")
    print(f"Naive DA:       {r2_naive:.6f}")
    print(f"Kalman DA:      {r2_kf:.6f}")
    print(f"Particle Filter:{r2_pf:.6f}")
    print(f"GARCH(1,1):     {r2_garch:.6f}")
    print(f"GARCH(2,2):     {r2_garch22:.6f}")
    print("-" * 50)
    
    # Save MSEs and R² to CSV
    mse_row = {
        'symbol': symbol,
        'mse_naive': f'{mse_naive:.20f}',
        'mse_kf': f'{mse_kf:.20f}',
        'mse_pf': f'{mse_pf:.20f}',
        'mse_garch': f'{mse_garch:.20f}',
        'mse_garch22': f'{mse_garch22:.20f}',
        'r2_naive': f'{r2_naive:.6f}',
        'r2_kf': f'{r2_kf:.6f}',
        'r2_pf': f'{r2_pf:.6f}',
        'r2_garch': f'{r2_garch:.6f}',
        'r2_garch22': f'{r2_garch22:.6f}'
    }
    
    mse_path = 'mse_results.csv'
    if os.path.exists(mse_path):
        mse_df = pd.read_csv(mse_path)
        # Check if symbol already exists and overwrite it
        if symbol in mse_df['symbol'].values:
            # Remove the existing row for this symbol
            mse_df = mse_df[mse_df['symbol'] != symbol]
        # Add the new row
        mse_df = pd.concat([mse_df, pd.DataFrame([mse_row])], ignore_index=True)
    else:
        mse_df = pd.DataFrame([mse_row])
    
    mse_df.to_csv(mse_path, index=False)
    print(f"Results saved for {symbol}")
    
    return mse_row

if __name__ == "__main__":
    # List of symbols to process
    symbols = ['BNB', 'BTC', 'TRX', 'XRP', 'ETH']
    
    print("Starting MSE calculation for all symbols...")
    print("=" * 60)
    
    all_results = []
    for symbol in symbols:
        try:
            result = calculate_mse_for_symbol(symbol)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            print("-" * 50)
            continue
    
    print("=" * 60)
    print("All calculations completed!")
    print(f"Results saved to mse_results.csv")
    
    # Display final summary
    if os.path.exists('mse_results.csv'):
        final_df = pd.read_csv('mse_results.csv')
        print("\nFinal Results Summary:")
        print(final_df.to_string(index=False))
