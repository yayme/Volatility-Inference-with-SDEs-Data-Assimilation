import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from arch import arch_model

from DA_utility import compute_rolling_volatility


def fit_garch_variance(returns, p, q, warmup=200):
    model = arch_model(returns * 100, vol="Garch", p=p, q=q, dist="normal")
    res = model.fit(last_obs=warmup, disp="off")
    forecasts = res.forecast(horizon=1, start=warmup)
    sigma = np.full(len(returns), np.nan)
    sigma[warmup:] = np.sqrt(forecasts.variance["h.1"].values) / 100
    return sigma


def evaluate(y_true, y_pred, burn_in=200):
    total = len(y_true)
    burn_mask = np.ones(total, dtype=bool)
    if burn_in > 0:
        burn_mask = np.arange(total) >= burn_in
    valid_mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    score_mask = burn_mask & valid_mask

    audit = {
        "total": int(total),
        "excluded_burn_in": int((~burn_mask).sum()),
        "excluded_nan": int((burn_mask & ~valid_mask).sum()),
        "scored": int(score_mask.sum()),
    }

    if audit["scored"] == 0:
        return np.nan, np.nan, audit
    return (
        mean_squared_error(y_true[score_mask], y_pred[score_mask]),
        r2_score(y_true[score_mask], y_pred[score_mask]),
        audit,
    )


def print_eval_audit(symbol, target, audit_by_method):
    print(f"audit {symbol} ({target})")
    for method, stats in audit_by_method.items():
        print(
            f"  {method}: total={stats['total']} burn_in_excluded={stats['excluded_burn_in']} "
            f"nan_excluded={stats['excluded_nan']} scored={stats['scored']}"
        )


def save_comparison_plot(symbol, target, actual, methods, plot_dir, plot_points=1000):
    n = len(actual)
    start = max(0, n - plot_points)
    idx = np.arange(start, n)

    plt.figure(figsize=(12, 5))
    plt.plot(idx, actual[start:], label="actual", linewidth=1.0, alpha=0.5)
    for name, pred in methods.items():
        plt.plot(idx, pred[start:], label=name, linewidth=1.0, alpha=0.5)
    plt.title(f"{symbol} {target}: actual vs methods (last {n-start} points)")
    plt.xlabel("time")
    plt.ylabel("volatility")
    plt.legend()
    plt.tight_layout()
    file_path = f"{plot_dir}/{symbol}_{target}_hmm_comparison.png"
    plt.savefig(file_path, dpi=140)
    plt.close()
    print(f"saved plot {file_path}")


def gaussian_pdf(x, means, variances):
    variances = np.maximum(variances, 1e-10)
    coef = 1.0 / np.sqrt(2.0 * np.pi * variances)
    exponent = -0.5 * ((x - means) ** 2) / variances
    return coef * np.exp(exponent)


def fit_hmm_train(obs_train, n_states=3, n_iter=200, random_state=42):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError("hmmlearn is required. Install with: pip install hmmlearn") from e

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(obs_train.reshape(-1, 1))

    means = model.means_.reshape(-1)
    covars = model.covars_.reshape(-1)
    return model.startprob_.copy(), model.transmat_.copy(), means, covars


def hmm_online_prior(obs, startprob, transmat, means, variances):
    n = len(obs)
    n_states = len(means)
    alpha = np.full(n_states, 1.0 / n_states)
    prior = np.zeros(n)

    for t in range(n):
        if t == 0:
            alpha_pred = startprob
        else:
            alpha_pred = alpha @ transmat

        prior[t] = float(alpha_pred @ means)

        lik = gaussian_pdf(obs[t], means, variances)
        numer = alpha_pred * lik
        s = np.sum(numer)
        if (not np.isfinite(s)) or s <= 0:
            alpha = alpha_pred / np.sum(alpha_pred)
        else:
            alpha = numer / s

    return prior


def run_symbol(
    symbol,
    burn_in,
    rolling_window=20,
    rolling_lag=20,
    n_states=3,
    train_ratio=0.7,
    n_iter=200,
    random_state=42,
    plot=False,
    plot_dir="plots",
    plot_points=1000,
):
    df = pd.read_csv(f"{symbol}_spot_full.csv")
    returns = df["bam_close"].pct_change().fillna(0)

    sigma_target = compute_rolling_volatility(returns, window=rolling_window).fillna(0.0)
    sigma_obs_da = sigma_target.shift(rolling_lag)
    if rolling_lag > 0:
        sigma_obs_da.iloc[:rolling_lag] = sigma_target.iloc[0]
    sigma_obs_da = sigma_obs_da.fillna(0.0)

    y_obs = sigma_obs_da.values
    split_idx = max(2, int(len(y_obs) * train_ratio))

    startprob, transmat, means, variances = fit_hmm_train(
        y_obs[:split_idx],
        n_states=n_states,
        n_iter=n_iter,
        random_state=random_state,
    )
    sigma_hmm_prior = hmm_online_prior(y_obs, startprob, transmat, means, variances)

    sigma_g11 = fit_garch_variance(returns, 1, 1, warmup=burn_in)
    sigma_g22 = fit_garch_variance(returns, 2, 2, warmup=burn_in)

    methods = {
        "hmm_prior": sigma_hmm_prior,
        "garch_11": sigma_g11,
        "garch_22": sigma_g22,
        "lagged_obs": y_obs,
    }

    out = {
        "symbol": symbol,
        "target": f"rolling_vol_{rolling_window}_lag{rolling_lag}",
        "n_states": n_states,
    }
    audit_by_method = {}

    y_true = sigma_target.values
    for name, pred in methods.items():
        mse, r2, audit = evaluate(y_true, pred, burn_in=burn_in)
        out[f"mse_{name}"] = mse
        out[f"r2_{name}"] = r2
        audit_by_method[name] = audit

    if plot:
        save_comparison_plot(
            symbol=symbol,
            target=f"rolling_vol_{rolling_window}_lag{rolling_lag}",
            actual=y_true,
            methods=methods,
            plot_dir=plot_dir,
            plot_points=plot_points,
        )

    return out, audit_by_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BNB", "BTC", "TRX", "XRP", "ETH"])
    parser.add_argument("--output", default="hmm_results.csv")
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--rolling-lag", type=int, default=20)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", default="plots")
    parser.add_argument("--plot-points", type=int, default=1000)
    args = parser.parse_args()

    if args.plot:
        import os
        os.makedirs(args.plot_dir, exist_ok=True)

    rows = []
    for symbol in args.symbols:
        try:
            row, audit = run_symbol(
                symbol=symbol,
                burn_in=args.burn_in,
                rolling_window=args.rolling_window,
                rolling_lag=args.rolling_lag,
                n_states=args.n_states,
                train_ratio=args.train_ratio,
                n_iter=args.n_iter,
                random_state=args.seed,
                plot=args.plot,
                plot_dir=args.plot_dir,
                plot_points=args.plot_points,
            )
            rows.append(row)
            print_eval_audit(symbol, row["target"], audit)
            print(f"processed {symbol}")
        except Exception as e:
            print(f"failed {symbol}: {e}")

    if not rows:
        print("no results")
        return

    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.output, index=False)
    print(result_df.to_string(index=False))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
