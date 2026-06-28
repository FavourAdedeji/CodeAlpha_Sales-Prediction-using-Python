"""
02_eda_feature_selection.py
------------------------------
Exploratory data analysis and feature selection for the sales prediction
task: distributions, correlations, pairwise relationships, and a
multicollinearity (VIF) check to justify which features feed the model.

Outputs (outputs/figures, outputs/tables):
    fig1_distributions.png
    fig2_correlation_heatmap.png
    fig3_spend_vs_sales_scatter.png
    fig4_pairplot.png
    correlation_matrix.csv
    vif_scores.csv

Run from the project root:
    python src/02_eda_feature_selection.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)
CHANNEL_COLORS = {"TV": "#1f5fa6", "Radio": "#2e8b57", "Newspaper": "#d4a017"}


def plot_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    cols = ["TV", "Radio", "Newspaper", "Sales"]
    colors = ["#1f5fa6", "#2e8b57", "#d4a017", "#c0392b"]
    for ax, col, color in zip(axes, cols, colors):
        sns.histplot(df[col], kde=True, ax=ax, color=color)
        ax.set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{col} ({'$ thousands' if col != 'Sales' else 'units (thousands)'})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_distributions.png", dpi=160)
    plt.close()
    print("Saved fig1_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    core_cols = ["TV", "Radio", "Newspaper", "Sales"]
    corr = df[core_cols].corr()
    corr.to_csv(TABLE_DIR / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation Matrix: Ad Spend vs. Sales", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_correlation_heatmap.png", dpi=160)
    plt.close()
    print("Saved fig2_correlation_heatmap.png")
    print("\nCorrelation with Sales:")
    print(corr["Sales"].sort_values(ascending=False).round(3))


def plot_spend_vs_sales(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, channel in zip(axes, ["TV", "Radio", "Newspaper"]):
        sns.regplot(data=df, x=channel, y="Sales", ax=ax,
                    scatter_kws={"alpha": 0.5, "color": CHANNEL_COLORS[channel]},
                    line_kws={"color": "#c0392b"})
        r = df[channel].corr(df["Sales"])
        ax.set_title(f"{channel} Spend vs. Sales (r = {r:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{channel} Spend ($ thousands)")
        ax.set_ylabel("Sales (thousands of units)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_spend_vs_sales_scatter.png", dpi=160)
    plt.close()
    print("Saved fig3_spend_vs_sales_scatter.png")


def plot_pairplot(df: pd.DataFrame):
    g = sns.pairplot(df[["TV", "Radio", "Newspaper", "Sales"]], corner=True,
                      plot_kws={"alpha": 0.5, "color": "#1f5fa6"},
                      diag_kws={"color": "#1f5fa6"})
    g.fig.suptitle("Pairwise Relationships: Ad Spend and Sales", y=1.02, fontsize=13, fontweight="bold")
    g.fig.savefig(FIG_DIR / "fig4_pairplot.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved fig4_pairplot.png")


def compute_vif(df: pd.DataFrame):
    """Variance Inflation Factor on the three raw spend channels — checks
    whether multicollinearity between TV/Radio/Newspaper spend could distort
    the regression coefficients used later for impact analysis."""
    X = df[["TV", "Radio", "Newspaper"]].copy()
    X.insert(0, "const", 1.0)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data[vif_data["feature"] != "const"].reset_index(drop=True)
    vif_data.to_csv(TABLE_DIR / "vif_scores.csv", index=False)
    print("\nVariance Inflation Factor (multicollinearity check):")
    print(vif_data.round(2))
    print("(VIF < 5 indicates low multicollinearity — all three channels are safe to include together.)")


def main():
    df = pd.read_csv(PROCESSED_DIR / "advertising_clean.csv")

    plot_distributions(df)
    plot_correlation_heatmap(df)
    plot_spend_vs_sales(df)
    plot_pairplot(df)
    compute_vif(df)


if __name__ == "__main__":
    main()
