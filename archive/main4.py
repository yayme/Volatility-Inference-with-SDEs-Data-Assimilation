import numpy as np
import pandas as pd
from DA_utility_heston import (
    compute_returns,
    compute_rolling_volatility,
    naive_heston_DA,
    kalman_heston_DA,
    particle_filter_heston_DA,
    heston_predictor,
    generic_heston_DA,
    naive_combiner
)
from sklearn.metrics import mean_squared_error, r2_score
import os

def infer_instantaneous_volatility(symbol):
    """
    Infer instantaneous volatility using Heston-based data assimilation methods.
    
    Args:
        symbol (str): The cryptocurrency symbol (e.g., 'XRP', 'BTC', 'ETH', etc.)
    
    Returns:
        pd.DataFrame: DataFrame containing log returns and instantaneous volatility estimates
    """
    print(f"Processing {symbol} for instantaneous volatility inference...")
    
    # Load data
    df = pd.read_csv(f'{symbol}_spot_full.csv')
    prices = df['bam_close']
    
    # Compute log returns
    log_returns = compute_returns(prices)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'timestamp': df.index,
        'price': prices,
        'log_return': log_returns,
        'abs_log_return': np.abs(log_returns)
    })
    
    # Compute rolling volatility for comparison
    results_df['rolling_vol_100'] = compute_rolling_volatility(log_returns, window=100)
    results_df['rolling_vol_50'] = compute_rolling_volatility(log_returns, window=50)
    results_df['rolling_vol_20'] = compute_rolling_volatility(log_returns, window=20)
    
    # --- Instantaneous Volatility Inference using Heston DA methods ---
    
    # 1. Naive Heston DA
    print(f"  Running Naive Heston DA...")
    sigma_model_naive, sigma_prior_naive, sigma_est_naive = naive_heston_DA(prices, alpha=0.3)
    results_df['heston_naive_model'] = sigma_model_naive
    results_df['heston_naive_prior'] = sigma_prior_naive
    results_df['heston_naive_est'] = sigma_est_naive
    
    # 2. Generic Heston DA 
    print(f"  Running Generic Heston DA...")
    sigma_model_generic, sigma_prior_generic, sigma_est_generic = generic_heston_DA(
        prices, predictor=lambda s, t: heston_predictor(s, t), 
        combiner=lambda pred, obs: naive_combiner(pred, obs, alpha=0.2)
    )
    results_df['heston_generic_model'] = sigma_model_generic
    results_df['heston_generic_prior'] = sigma_prior_generic
    results_df['heston_generic_est'] = sigma_est_generic
    
    # 3. Kalman Heston DA
    print(f"  Running Kalman Heston DA...")
    sigma_model_kalman, sigma_prior_kalman, sigma_est_kalman = kalman_heston_DA(
        prices, R=1e-4, Q=0.01
    )
    results_df['heston_kalman_model'] = sigma_model_kalman
    results_df['heston_kalman_prior'] = sigma_prior_kalman
    results_df['heston_kalman_est'] = sigma_est_kalman
    
    # 4. Particle Filter Heston DA
    print(f"  Running Particle Filter Heston DA...")
    sigma_model_pf, sigma_prior_pf, sigma_est_pf = particle_filter_heston_DA(
        prices, N_particles=150, R=0.001
    )
    results_df['heston_pf_model'] = sigma_model_pf
    results_df['heston_pf_prior'] = sigma_prior_pf
    results_df['heston_pf_est'] = sigma_est_pf
    
    # Add symbol column
    results_df['symbol'] = symbol
    
    print(f"  Completed {symbol} - {len(results_df)} data points processed")
    
    return results_df

def calculate_performance_metrics(all_results):
    """
    Calculate performance metrics from results dataframes without saving large CSV.
    
    Args:
        all_results (list): List of dataframes from infer_instantaneous_volatility
    
    Returns:
        pd.DataFrame: Performance metrics for all symbols
    """
    print(f"\nCalculating Performance Metrics...")
    print("=" * 60)
    
    performance_results = []
    
    for results_df in all_results:
        symbol = results_df['symbol'].iloc[0]
        
        # Use absolute log returns as ground truth for instantaneous volatility
        ground_truth = results_df['abs_log_return']
        
        # Calculate MSE for each method
        methods = {
            'Naive Heston': 'heston_naive_prior',
            'Generic Heston': 'heston_generic_prior', 
            'Kalman Heston': 'heston_kalman_prior',
            'Particle Filter': 'heston_pf_prior',
            'Rolling Vol (20)': 'rolling_vol_20',
            'Rolling Vol (50)': 'rolling_vol_50',
            'Rolling Vol (100)': 'rolling_vol_100'
        }
        
        symbol_perf = {'symbol': symbol}
        
        print(f"\n{symbol} Performance Metrics:")
        print("-" * 40)
        
        for method_name, column_name in methods.items():
            if column_name in results_df.columns:
                predictions = results_df[column_name]
                
                # Remove NaN values for calculation
                mask = (~np.isnan(ground_truth)) & (~np.isnan(predictions))
                if mask.sum() > 0:
                    gt_clean = ground_truth[mask]
                    pred_clean = predictions[mask]
                    
                    mse = mean_squared_error(gt_clean, pred_clean)
                    r2 = r2_score(gt_clean, pred_clean)
                    
                    symbol_perf[f'mse_{method_name.lower().replace(" ", "_")}'] = mse
                    symbol_perf[f'r2_{method_name.lower().replace(" ", "_")}'] = r2
                    
                    print(f"  {method_name:15} - MSE: {mse:.8f}, R²: {r2:.6f}")
                else:
                    print(f"  {method_name:15} - No valid data")
        
        performance_results.append(symbol_perf)
    
    # Save performance results
    if performance_results:
        perf_df = pd.DataFrame(performance_results)
        perf_df.to_csv('instantaneous_volatility_performance.csv', index=False)
        print(f"\nPerformance metrics saved to 'instantaneous_volatility_performance.csv'")
        
        # Show best performing method for each symbol
        print(f"\nBest Performing Methods (by R²):")
        print("-" * 50)
        for _, row in perf_df.iterrows():
            symbol = row['symbol']
            r2_cols = [col for col in row.index if col.startswith('r2_')]
            if r2_cols:
                best_method = max(r2_cols, key=lambda x: row[x] if not pd.isna(row[x]) else -float('inf'))
                best_r2 = row[best_method]
                method_name = best_method.replace('r2_', '').replace('_', ' ').title()
                print(f"  {symbol}: {method_name} (R² = {best_r2:.6f})")
        
        return perf_df
    
    return None

if __name__ == "__main__":
    # List of symbols to process
    symbols = ['BNB', 'BTC', 'TRX', 'XRP', 'ETH']
    
    print("Starting instantaneous volatility inference using Heston-based data assimilation...")
    print("=" * 80)
    
    all_results = []
    
    for symbol in symbols:
        try:
            # Process each symbol
            symbol_results = infer_instantaneous_volatility(symbol)
            all_results.append(symbol_results)
            print(f"Successfully processed {symbol}")
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
        print("-" * 50)
    
    if all_results:
        # Calculate performance metrics directly without saving large CSV
        print("\nCalculating comprehensive performance metrics...")
        performance_df = calculate_performance_metrics(all_results)
        
        print("\n" + "=" * 80)
        print("Instantaneous volatility analysis completed!")
        print("Performance metrics saved to 'instantaneous_volatility_performance.csv'")
        
        if performance_df is not None:
            print(f"\nProcessed {len(all_results)} symbols with {sum(len(df) for df in all_results)} total observations")
    else:
        print("No results to process - all symbols failed.")
