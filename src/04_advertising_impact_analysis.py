"""
04_advertising_impact_analysis.py
------------------------------------
Quantifies how changes in advertising spend on each channel translate into
changes in sales: marginal effect (regression coefficients), elasticity
(% sales change per % spend change), and a "what-if" budget reallocation
scenario, used to produce actionable marketing recommendations.

Outputs:
    outputs/figures/fig8_elasticity.png
    outputs/figures/fig9_budget_reallocation_scenario.png
    outputs/tables/elasticity_by_channel.csv
    outputs/tables/budget_reallocation_scenario.csv

Run from the project root:
    python src/04_advertising_impact_analysis.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "outputs" / "models"

sns.set_theme(style="whitegrid", font_scale=1.0)
FEATURES = ["TV", "Radio", "Newspaper"]
CHANNEL_COLORS = {"TV": "#1f5fa6", "Radio": "#2e8b57", "Newspaper": "#d4a017"}


def compute_elasticity(df: pd.DataFrame, lr: LinearRegression) -> pd.DataFrame:
    """Elasticity = (coefficient * mean(spend)) / mean(sales).
    Interpreted as: a 1% increase in this channel's spend is associated
    with an `elasticity`% change in sales, evaluated at average spend
    levels (a standard linear-model elasticity approximation)."""
    mean_sales = df["Sales"].mean()
    rows = []
    for feature, coef in zip(FEATURES, lr.coef_):
        mean_spend = df[feature].mean()
        elasticity = coef * mean_spend / mean_sales
        rows.append({
            "channel": feature,
            "coefficient": coef,
            "mean_spend_$k": mean_spend,
            "elasticity_pct": elasticity * 100,
        })
    elasticity_df = pd.DataFrame(rows).sort_values("elasticity_pct", ascending=False)
    return elasticity_df


def plot_elasticity(elasticity_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = elasticity_df.sort_values("elasticity_pct")
    colors = [CHANNEL_COLORS[c] for c in order["channel"]]
    bars = ax.barh(order["channel"], order["elasticity_pct"], color=colors)
    for bar, val in zip(bars, order["elasticity_pct"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", fontsize=10)
    ax.set_xlabel("Sales Elasticity (% change in Sales per 1% change in channel spend)")
    ax.set_title("Advertising Impact: Sales Elasticity by Channel", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig8_elasticity.png", dpi=160)
    plt.close()
    print("Saved fig8_elasticity.png")


def budget_reallocation_scenario(df: pd.DataFrame, lr: LinearRegression) -> pd.DataFrame:
    """Simulates shifting 20% of the lowest-elasticity channel's (Newspaper)
    average budget into the highest-elasticity channel (Radio), holding
    total budget fixed, and predicts the resulting change in sales using
    the fitted linear model. This is illustrative, not a guarantee — see
    README limitations on extrapolation beyond the observed spend range."""
    avg_spend = df[FEATURES].mean()
    baseline_sales = lr.predict(pd.DataFrame([avg_spend.values], columns=FEATURES))[0]

    shift_amount = 0.20 * avg_spend["Newspaper"]
    scenario_spend = avg_spend.copy()
    scenario_spend["Newspaper"] -= shift_amount
    scenario_spend["Radio"] += shift_amount
    scenario_sales = lr.predict(pd.DataFrame([scenario_spend.values], columns=FEATURES))[0]

    result = pd.DataFrame({
        "scenario": ["Current average allocation", "Shift 20% of Newspaper budget to Radio"],
        "TV_$k": [avg_spend["TV"], scenario_spend["TV"]],
        "Radio_$k": [avg_spend["Radio"], scenario_spend["Radio"]],
        "Newspaper_$k": [avg_spend["Newspaper"], scenario_spend["Newspaper"]],
        "predicted_sales": [baseline_sales, scenario_sales],
    })
    result["sales_change"] = result["predicted_sales"] - baseline_sales
    result["sales_change_pct"] = 100 * result["sales_change"] / baseline_sales

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(result["scenario"], result["predicted_sales"], color=["#7f8c8d", "#1f5fa6"], width=0.55)
    for bar, val in zip(bars, result["predicted_sales"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f"{val:.2f}", ha="center", fontweight="bold")
    ax.set_ylabel("Predicted Sales (thousands of units)")
    ax.set_title("Budget Reallocation Scenario:\nShifting Newspaper Spend \u2192 Radio (same total budget)",
                 fontsize=11, fontweight="bold")
    plt.xticks(rotation=8, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig9_budget_reallocation_scenario.png", dpi=160)
    plt.close()
    print("Saved fig9_budget_reallocation_scenario.png")

    return result


def main():
    df = pd.read_csv(PROCESSED_DIR / "advertising_clean.csv")
    lr = joblib.load(MODEL_DIR / "linear_regression_model.pkl")

    elasticity_df = compute_elasticity(df, lr)
    elasticity_df.to_csv(TABLE_DIR / "elasticity_by_channel.csv", index=False)
    print("Sales elasticity by channel:")
    print(elasticity_df.round(4))
    plot_elasticity(elasticity_df)

    scenario_df = budget_reallocation_scenario(df, lr)
    scenario_df.to_csv(TABLE_DIR / "budget_reallocation_scenario.csv", index=False)
    print("\nBudget reallocation scenario:")
    print(scenario_df.round(3))


if __name__ == "__main__":
    main()
