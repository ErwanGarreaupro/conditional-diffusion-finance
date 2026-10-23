from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_path = Path(__file__).resolve().parent

scenarios = pd.read_csv(
    base_path / "generated_apple_scenarios.csv"
)

historical = pd.read_csv(
    base_path / "aapl_us.csv"
)

historical["Date"] = pd.to_datetime(historical["Date"])
historical = historical.sort_values("Date").reset_index(drop=True)

prediction_date = pd.Timestamp("2025-12-31")

paths = scenarios.drop(columns=["Scenario"]).to_numpy(dtype=float)

horizon = paths.shape[1] - 1
days = np.arange(horizon + 1)

prediction_index = historical.index[
    historical["Date"] == prediction_date
]

if len(prediction_index) == 0:
    raise ValueError("Prediction date not found in historical data")

start = prediction_index[0]

real_prices = historical["Close"].iloc[
    start:start + horizon + 1
].to_numpy(dtype=float)

if len(real_prices) < horizon + 1:
    raise ValueError("Not enough real prices to evaluate the full horizon")

real_days = np.arange(len(real_prices))

median = np.median(paths, axis=0)
q05 = np.quantile(paths, 0.05, axis=0)
q95 = np.quantile(paths, 0.95, axis=0)

simulated_returns = np.log(
    paths[:, 1:] / paths[:, :-1]
)

real_returns = np.log(
    real_prices[1:] / real_prices[:-1]
)

simulated_mean_returns = np.mean(simulated_returns, axis=0)
simulated_q05_returns = np.quantile(simulated_returns, 0.05, axis=0)
simulated_q95_returns = np.quantile(simulated_returns, 0.95, axis=0)

plt.figure(figsize=(14, 8))

for path in paths[:100]:
    plt.plot(
        days,
        path,
        color="steelblue",
        alpha=0.12,
        linewidth=0.8
    )

plt.fill_between(
    days,
    q05,
    q95,
    color="steelblue",
    alpha=0.2,
    label="90% simulation interval"
)

plt.plot(
    days,
    median,
    color="red",
    linewidth=2.5,
    label="Median simulation"
)

plt.plot(
    real_days,
    real_prices,
    color="black",
    linewidth=3,
    label="Real Apple price"
)

plt.title("Apple - Conditional Diffusion vs Real Prices")
plt.xlabel("Trading days after 2025-12-31")
plt.ylabel("Apple stock price ($)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.figure(figsize=(14, 7))

return_days = np.arange(1, horizon + 1)

plt.fill_between(
    return_days,
    100 * simulated_q05_returns,
    100 * simulated_q95_returns,
    color="steelblue",
    alpha=0.2,
    label="90% simulated return interval"
)

plt.plot(
    return_days,
    100 * simulated_mean_returns,
    color="red",
    linewidth=2,
    label="Mean simulated return"
)

plt.plot(
    return_days,
    100 * real_returns,
    color="black",
    linewidth=1.5,
    label="Real Apple return"
)

plt.axhline(0, color="gray", linewidth=1)

plt.title("Apple - Daily Log Returns")
plt.xlabel("Trading days after 2025-12-31")
plt.ylabel("Daily log return (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.figure(figsize=(12, 7))

plt.hist(
    100 * simulated_returns.flatten(),
    bins=100,
    density=True,
    alpha=0.5,
    color="steelblue",
    label="Simulated returns"
)

plt.hist(
    100 * real_returns,
    bins=20,
    density=True,
    alpha=0.5,
    color="black",
    label="Real returns"
)

plt.title("Apple - Distribution of Daily Log Returns")
plt.xlabel("Daily log return (%)")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

simulated_volatility = np.std(
    simulated_returns,
    axis=1,
    ddof=1
)

real_volatility = np.std(
    real_returns,
    ddof=1
)

plt.figure(figsize=(12, 7))

plt.hist(
    100 * simulated_volatility,
    bins=40,
    density=True,
    alpha=0.7,
    color="steelblue",
    label="Simulated volatility"
)

plt.axvline(
    100 * real_volatility,
    color="red",
    linewidth=2.5,
    label="Realized volatility"
)

plt.title("Apple - Volatility Over 60 Trading Days")
plt.xlabel("Daily volatility (%)")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

coverage = np.mean(
    (real_prices >= q05) &
    (real_prices <= q95)
)

print("\n========== MODEL EVALUATION ==========")

print(f"Prediction date: {prediction_date.date()}")
print(f"Forecast horizon: {horizon} trading days")
print(f"Generated scenarios: {paths.shape[0]}")

print("\n========== FINAL PRICES ==========")

print(f"Initial price: ${real_prices[0]:.2f}")
print(f"Real final price: ${real_prices[-1]:.2f}")
print(f"Median simulated final price: ${median[-1]:.2f}")

print("\n========== DAILY RETURNS ==========")

print(
    f"Mean simulated daily return: "
    f"{100 * np.mean(simulated_returns):.4f}%"
)

print(
    f"Mean real daily return: "
    f"{100 * np.mean(real_returns):.4f}%"
)

print("\n========== VOLATILITY ==========")

print(
    f"Mean simulated daily volatility: "
    f"{100 * np.mean(simulated_volatility):.4f}%"
)

print(
    f"Realized daily volatility: "
    f"{100 * real_volatility:.4f}%"
)

print("\n========== COVERAGE ==========")

print(
    f"Real price inside 90% interval: "
    f"{100 * coverage:.2f}% of trading days"
)
def squared_returns_autocorrelation(returns, max_lag=20):
    squared = np.asarray(returns, dtype=float) ** 2

    correlations = []

    for lag in range(1, max_lag + 1):
        x = squared[:-lag]
        y = squared[lag:]

        if np.std(x) == 0 or np.std(y) == 0:
            correlations.append(np.nan)
        else:
            correlations.append(np.corrcoef(x, y)[0, 1])

    return np.array(correlations)


max_lag = 20
lags = np.arange(1, max_lag + 1)

historical_prices = historical["Close"].iloc[:start + 1].to_numpy(
    dtype=float
)

historical_returns = np.diff(np.log(historical_prices))

historical_acf = squared_returns_autocorrelation(
    historical_returns,
    max_lag
)

real_acf = squared_returns_autocorrelation(
    real_returns,
    max_lag
)

simulated_acfs = np.array([
    squared_returns_autocorrelation(path_returns, max_lag)
    for path_returns in simulated_returns
])

simulated_acf_mean = np.nanmean(simulated_acfs, axis=0)

simulated_acf_q05 = np.nanquantile(
    simulated_acfs, 0.05, axis=0
)

simulated_acf_q95 = np.nanquantile(
    simulated_acfs, 0.95, axis=0
)

plt.figure(figsize=(13, 7))

plt.fill_between(
    lags,
    simulated_acf_q05,
    simulated_acf_q95,
    color="steelblue",
    alpha=0.2,
    label="90% simulated interval"
)

plt.plot(
    lags,
    simulated_acf_mean,
    color="steelblue",
    linewidth=2.5,
    marker="o",
    label="Mean simulated ACF"
)

plt.plot(
    lags,
    historical_acf,
    color="black",
    linewidth=2.5,
    marker="o",
    label="Historical Apple ACF"
)

plt.plot(
    lags,
    real_acf,
    color="red",
    linewidth=2,
    marker="o",
    label="Real future Apple ACF"
)

plt.axhline(
    0,
    color="gray",
    linestyle="--",
    linewidth=1
)

plt.title("Apple - Autocorrelation of Squared Returns")
plt.xlabel("Lag (trading days)")
plt.ylabel("Autocorrelation")
plt.xticks(lags)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()