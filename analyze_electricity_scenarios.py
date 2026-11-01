import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCENARIOS_FILE = BASE_DIR / "electricity_backtest_2025.csv"
REAL_PRICES_FILE = BASE_DIR / "france_day_ahead_daily.csv"
OUTPUT_DIR = BASE_DIR / "backtest_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2025
ROLLING_WINDOW = 30

scenarios_df = pd.read_csv(SCENARIOS_FILE)
real_df = pd.read_csv(REAL_PRICES_FILE)

scenarios_df["PredictionDate"] = pd.to_datetime(scenarios_df["PredictionDate"])
real_df["Date"] = pd.to_datetime(real_df["Date"])
real_df["Price"] = pd.to_numeric(real_df["Price"], errors="raise")

day_columns = sorted(
    [col for col in scenarios_df.columns if col.startswith("Day_")],
    key=lambda col: int(col.split("_")[1])
)

horizon = len(day_columns)

if horizon < 2:
    raise ValueError("Forecast horizon must be at least 2 days")

if day_columns != [f"Day_{i}" for i in range(1, horizon + 1)]:
    raise ValueError("Invalid forecast horizon columns")

if real_df["Date"].duplicated().any():
    raise ValueError("Duplicate dates in real price data")

real_prices = real_df.set_index("Date")["Price"]

if not np.isfinite(real_prices.to_numpy()).all():
    raise ValueError("Invalid real prices")

records = []
weekly_volatility_records = []

for prediction_date, group in scenarios_df.groupby("PredictionDate", sort=True):
    if prediction_date.year != YEAR:
        continue

    scenarios = group[day_columns].to_numpy(dtype=float)

    if scenarios.shape[0] < 2 or not np.isfinite(scenarios).all():
        raise ValueError(f"Invalid scenarios for {prediction_date.date()}")

    forecast_dates = pd.date_range(
        start=prediction_date,
        periods=horizon,
        freq="D"
    )

    observed = real_prices.reindex(forecast_dates)

    if observed.isna().any():
        raise ValueError(f"Missing real prices for {prediction_date.date()}")

    actual = observed.to_numpy(dtype=float)
    simulated_mean = scenarios.mean(axis=0)
    simulated_std = scenarios.std(axis=0, ddof=1)
    q05 = np.quantile(scenarios, 0.05, axis=0)
    q95 = np.quantile(scenarios, 0.95, axis=0)

    real_weekly_volatility = np.std(np.diff(actual), ddof=1)
    mean_weekly_volatility = np.std(np.diff(simulated_mean), ddof=1)
    scenario_weekly_volatilities = np.std(
        np.diff(scenarios, axis=1),
        axis=1,
        ddof=1
    )

    weekly_volatility_records.append({
        "PredictionDate": prediction_date,
        "RealWeeklyVolatility": real_weekly_volatility,
        "MeanWeeklyVolatility": mean_weekly_volatility,
        "ScenarioWeeklyVolatility": scenario_weekly_volatilities.mean()
    })

    for h, date in enumerate(forecast_dates):
        records.append({
            "Date": date,
            "PredictionDate": prediction_date,
            "Horizon": h + 1,
            "RealPrice": actual[h],
            "SimulatedMean": simulated_mean[h],
            "SimulatedStd": simulated_std[h],
            "Q05": q05[h],
            "Q95": q95[h]
        })

results = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
weekly_volatility = pd.DataFrame(weekly_volatility_records)

if results.empty:
    raise ValueError("No forecasts found")

if results["Date"].duplicated().any():
    raise ValueError("Overlapping forecast dates detected")

if len(results) < ROLLING_WINDOW + 1:
    raise ValueError("Not enough observations for rolling analysis")

real = results["RealPrice"].to_numpy(dtype=float)
simulated = results["SimulatedMean"].to_numpy(dtype=float)

results["RealVariation"] = results["RealPrice"].diff()
results["SimulatedVariation"] = results["SimulatedMean"].diff()

results["RollingPriceCorrelation"] = (
    results["RealPrice"]
    .rolling(ROLLING_WINDOW)
    .corr(results["SimulatedMean"])
)

results["RollingVariationCorrelation"] = (
    results["RealVariation"]
    .rolling(ROLLING_WINDOW)
    .corr(results["SimulatedVariation"])
)

results["RollingRealVolatility"] = (
    results["RealVariation"]
    .rolling(ROLLING_WINDOW)
    .std()
)

results["RollingMeanVolatility"] = (
    results["SimulatedVariation"]
    .rolling(ROLLING_WINDOW)
    .std()
)

weekly_volatility["RollingRealScenarioVolatility"] = (
    weekly_volatility["RealWeeklyVolatility"]
    .rolling(4, min_periods=4)
    .mean()
)

weekly_volatility["RollingScenarioVolatility"] = (
    weekly_volatility["ScenarioWeeklyVolatility"]
    .rolling(4, min_periods=4)
    .mean()
)

weekly_volatility["RollingScenarioVolatilityRatio"] = (
    weekly_volatility["RollingScenarioVolatility"]
    / weekly_volatility["RollingRealScenarioVolatility"].replace(0, np.nan)
)

results["RollingScenarioVolatility"] = np.nan
results["RollingScenarioVolatilityRatio"] = np.nan

for _, row in weekly_volatility.iterrows():
    mask = results["PredictionDate"] == row["PredictionDate"]

    results.loc[mask, "RollingScenarioVolatility"] = (
        row["RollingScenarioVolatility"]
    )

    results.loc[mask, "RollingScenarioVolatilityRatio"] = (
        row["RollingScenarioVolatilityRatio"]
    )

global_price_correlation = np.corrcoef(real, simulated)[0, 1]
global_variation_correlation = np.corrcoef(
    np.diff(real),
    np.diff(simulated)
)[0, 1]

real_volatility = np.std(np.diff(real), ddof=1)
mean_volatility = np.std(np.diff(simulated), ddof=1)

average_real_weekly_volatility = weekly_volatility[
    "RealWeeklyVolatility"
].mean()

average_scenario_volatility = weekly_volatility[
    "ScenarioWeeklyVolatility"
].mean()

scenario_volatility_ratio = (
    average_scenario_volatility / average_real_weekly_volatility
    if average_real_weekly_volatility > 0 else np.nan
)

mean_volatility_ratio = (
    mean_volatility / real_volatility
    if real_volatility > 0 else np.nan
)

mae = np.mean(np.abs(simulated - real))
rmse = np.sqrt(np.mean((simulated - real) ** 2))
bias = np.mean(simulated - real)

coverage90 = np.mean(
    (real >= results["Q05"].to_numpy()) &
    (real <= results["Q95"].to_numpy())
)

mean_predictive_std = results["SimulatedStd"].mean()
mean_interval_width = (results["Q95"] - results["Q05"]).mean()

print("\n========== GLOBAL BACKTEST 2025 ==========")
print(f"Evaluated days: {len(results)}")
print(f"Global price correlation: {global_price_correlation:.4f}")
print(f"Global variation correlation: {global_variation_correlation:.4f}")
print(f"Real daily volatility: {real_volatility:.4f} EUR/MWh")
print(f"Mean forecast volatility: {mean_volatility:.4f} EUR/MWh")
print(f"Mean forecast volatility ratio: {mean_volatility_ratio:.4f}")
print(f"Average real weekly volatility: {average_real_weekly_volatility:.4f} EUR/MWh")
print(f"Average scenario volatility: {average_scenario_volatility:.4f} EUR/MWh")
print(f"Scenario volatility ratio: {scenario_volatility_ratio:.4f}")
print(f"Mean predictive std: {mean_predictive_std:.4f} EUR/MWh")
print(f"MAE: {mae:.4f} EUR/MWh")
print(f"RMSE: {rmse:.4f} EUR/MWh")
print(f"Bias: {bias:.4f} EUR/MWh")
print(f"90% interval coverage: {coverage90:.2%}")
print(f"Mean 90% interval width: {mean_interval_width:.4f} EUR/MWh")

fig, axes = plt.subplots(
    4, 1,
    figsize=(17, 14),
    sharex=True,
    gridspec_kw={"height_ratios": [2.8, 1.2, 1.2, 1]}
)

axes[0].fill_between(
    results["Date"],
    results["Q05"],
    results["Q95"],
    color="lightblue",
    alpha=0.45,
    label="90% prediction interval"
)

axes[0].plot(
    results["Date"],
    real,
    color="red",
    linewidth=1.5,
    label="Real electricity prices"
)

axes[0].plot(
    results["Date"],
    simulated,
    color="navy",
    linewidth=1.5,
    label="Diffusion mean forecast"
)

axes[0].set_title(
    "France Day-Ahead Electricity Prices - Conditional Diffusion Backtest (2025)",
    fontsize=14
)

axes[0].set_ylabel("Price (EUR/MWh)")
axes[0].legend(loc="upper right")
axes[0].grid(alpha=0.25)

axes[1].plot(
    results["Date"],
    results["RollingPriceCorrelation"],
    color="darkgreen",
    linewidth=1.6,
    label="30-day price correlation"
)

axes[1].plot(
    results["Date"],
    results["RollingVariationCorrelation"],
    color="purple",
    linewidth=1.6,
    label="30-day variation correlation"
)

axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1].axhline(1, color="gray", linestyle=":", linewidth=1)
axes[1].set_ylim(-1.05, 1.05)
axes[1].set_ylabel("Correlation")
axes[1].legend(loc="upper right")
axes[1].grid(alpha=0.25)

axes[2].plot(
    results["Date"],
    results["RollingRealVolatility"],
    color="red",
    linewidth=1.6,
    label="Real 30-day volatility"
)

axes[2].plot(
    results["Date"],
    results["RollingMeanVolatility"],
    color="navy",
    linewidth=1.6,
    label="Mean forecast 30-day volatility"
)

axes[2].plot(
    results["Date"],
    results["RollingScenarioVolatility"],
    color="darkorange",
    linewidth=1.8,
    label="Scenario volatility (4-week average)"
)

axes[2].set_ylabel("Volatility (EUR/MWh)")
axes[2].legend(loc="upper right")
axes[2].grid(alpha=0.25)

axes[3].plot(
    results["Date"],
    results["RollingScenarioVolatilityRatio"],
    color="darkorange",
    linewidth=1.8,
    label="Scenario / real volatility ratio"
)

axes[3].axhline(
    1,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Equal volatility"
)

axes[3].set_ylabel("Volatility ratio")
axes[3].set_xlabel("Date")
axes[3].legend(loc="upper right")
axes[3].grid(alpha=0.25)

metrics_text = (
    f"Price correlation: {global_price_correlation:.3f}\n"
    f"Variation correlation: {global_variation_correlation:.3f}\n"
    f"Real volatility: {real_volatility:.2f}\n"
    f"Mean forecast volatility: {mean_volatility:.2f}\n"
    f"Scenario volatility ratio: {scenario_volatility_ratio:.3f}\n"
    f"Mean predictive std: {mean_predictive_std:.2f}\n"
    f"MAE: {mae:.2f}\n"
    f"RMSE: {rmse:.2f}\n"
    f"Coverage 90%: {coverage90:.1%}"
)

axes[0].text(
    0.02,
    0.98,
    metrics_text,
    transform=axes[0].transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.9,
        edgecolor="gray"
    )
)

plt.tight_layout()

output_path = OUTPUT_DIR / "global_backtest_2025.png"

plt.savefig(output_path, dpi=200)
plt.show()

print(f"\nGraph saved: {output_path}")