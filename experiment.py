import matplotlib.pyplot as plt
from utils import load_crypto_data, kalman_filter_sde, particle_filter_sde, run_out_of_sample_garch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def report_metrics(target_name, true_vals, kf_priors, pf_priors, garch_priors=None):
    # Slice first 101 to avoid burn-in NaNs and 1-step logic misalignment
    true = true_vals.iloc[101:].values
    kf = kf_priors[101:]
    pf = pf_priors[101:]
    
    kf_r2 = r2_score(true, kf)
    pf_r2 = r2_score(true, pf)
    print(f"[{target_name}] Kalman Filter R2: {kf_r2:.4f}")
    print(f"[{target_name}] Particle Filter R2: {pf_r2:.4f}")
    if garch_priors is not None:
        garch_r2 = r2_score(true, garch_priors[101:])
        print(f"[{target_name}] GARCH(1,1) R2: {garch_r2:.4f}")

def main():
    print("Loading BTC data...")
    df = load_crypto_data('BTC_spot_full.csv', limit=5000, window=100)
    
    # Extract the two targets
    target_inst = df['target_inst']
    target_roll = df['target_roll']
    
    # The scale of raw returns is around ~0.001
    # Thus, xi (volatility process noise) should be small, and R (measurement noise) should match!
    # A generic tuning for non-annualized scale:
    kf_inst_priors, _ = kalman_filter_sde(target_inst, sde='OU', xi=0.001, R=1e-6)
    pf_inst_priors, _ = particle_filter_sde(target_inst, sde='CIR', xi=0.01, R=1e-6)
    
    kf_roll_priors, _ = kalman_filter_sde(target_roll, sde='OU', xi=0.001, R=1e-8)
    pf_roll_priors, _ = particle_filter_sde(target_roll, sde='CIR', xi=0.01, R=1e-8)
    
    print("Running out-of-sample GARCH benchmark...")
    garch_priors = run_out_of_sample_garch(df['log_return'].fillna(0))
    
    print("--- 1-Step Ahead Forecasting Performance ---")
    report_metrics("Instantaneous Volatility", target_inst, kf_inst_priors, pf_inst_priors)
    report_metrics("Rolling Smoothed Volatility", target_roll, kf_roll_priors, pf_roll_priors, garch_priors)
    print("Notice that 1-Step R2 on Instantaneous Volatility is physically expected to be negative (white noise barrier),")
    print("whereas Rolling Volatility allows R2 > 0.99 via trivial lag tracking.")
    
    print("Plotting target variables...")
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    
    axs[0].plot(target_inst, label='True Instantaneous Vol (abs log returns)', color='red', alpha=0.3)
    axs[0].plot(kf_inst_priors, label='Kalman Filter (Prior)', color='blue', alpha=0.7)
    axs[0].set_title("Target 1: Instantaneous Volatility Tracking")
    axs[0].legend()
    
    axs[1].plot(target_roll, label='True Rolling Vol (100-pd std)', color='blue', linewidth=2)
    axs[1].plot(kf_roll_priors, label='Kalman Filter (Prior)', color='green', alpha=0.9, linestyle='--')
    axs[1].plot(garch_priors, label='GARCH(1,1) (Prior)', color='purple', alpha=0.7, linestyle=':')
    axs[1].set_title("Target 2: Rolling Volatility Tracking")
    axs[1].legend()

    plt.tight_layout()
    output_img = "refactored_targets.png"
    plt.savefig(output_img, dpi=150)
    print(f"Plot saved successfully to {output_img}!")

if __name__ == "__main__":
    main()
