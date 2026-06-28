"""
03_model_training.py
-----------------------
Trains and compares several regression models to predict Sales from
advertising spend (TV, Radio, Newspaper), selects the best model by
cross-validated performance, and saves the fitted model + predictions.

Models compared:
    - Linear Regression (baseline, interpretable coefficients)
    - Ridge Regression (regularized linear)
    - Polynomial Regression (degree 2, captures diminishing returns / synergy)
    - Random Forest Regressor (non-linear, captures interactions automatically)

Outputs:
    outputs/figures/fig5_actual_vs_predicted.png
    outputs/figures/fig6_residuals.png
    outputs/figures/fig7_model_comparison.png
    outputs/tables/model_performance.csv
    outputs/tables/linear_regression_coefficients.csv
    outputs/models/best_model.pkl

Run from the project root:
    python src/03_model_training.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "outputs" / "models"
for d in (FIG_DIR, TABLE_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)
RANDOM_STATE = 42
FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"


def build_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Polynomial Regression (deg=2)": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            LinearRegression(),
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=5, random_state=RANDOM_STATE
        ),
    }


def evaluate_models(X_train, X_test, y_train, y_test):
    models = build_models()
    rows = []
    fitted = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

        rows.append({
            "model": name,
            "test_r2": r2,
            "test_mae": mae,
            "test_rmse": rmse,
            "cv_r2_mean": cv_r2.mean(),
            "cv_r2_std": cv_r2.std(),
        })
        fitted[name] = model
        print(f"{name:32s} | Test R2={r2:.4f}  RMSE={rmse:.3f}  MAE={mae:.3f}  | CV R2={cv_r2.mean():.4f} (+/-{cv_r2.std():.4f})")

    results = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)
    return results, fitted


def plot_model_comparison(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    order = results.sort_values("cv_r2_mean")
    bars = ax.barh(order["model"], order["cv_r2_mean"], xerr=order["cv_r2_std"],
                    color="#1f5fa6", capsize=4)
    ax.set_xlabel("Cross-Validated R\u00b2 (5-fold, mean \u00b1 std)")
    ax.set_title("Model Comparison: Predicting Sales from Ad Spend", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, order["cv_r2_mean"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_model_comparison.png", dpi=160)
    plt.close()
    print("Saved fig7_model_comparison.png")


def plot_actual_vs_predicted(y_test, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_test, y_pred, alpha=0.6, color="#1f5fa6", edgecolor="white", s=60)
    lims = [min(y_test.min(), y_pred.min()) - 1, max(y_test.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, color="#c0392b", linestyle="--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Sales (thousands of units)")
    ax.set_ylabel("Predicted Sales (thousands of units)")
    ax.set_title(f"Actual vs. Predicted Sales\n({model_name}, held-out test set)", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_actual_vs_predicted.png", dpi=160)
    plt.close()
    print("Saved fig5_actual_vs_predicted.png")


def plot_residuals(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.6, color="#2e8b57", edgecolor="white", s=60)
    axes[0].axhline(0, color="#c0392b", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Sales")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title("Residuals vs. Predicted Values", fontsize=11, fontweight="bold")

    sns.histplot(residuals, kde=True, ax=axes[1], color="#2e8b57")
    axes[1].set_title("Distribution of Residuals", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Residual")

    fig.suptitle(f"Residual Diagnostics \u2014 {model_name}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_residuals.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved fig6_residuals.png")


def main():
    df = pd.read_csv(PROCESSED_DIR / "advertising_clean.csv")
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")
    results, fitted = evaluate_models(X_train, X_test, y_train, y_test)
    results.to_csv(TABLE_DIR / "model_performance.csv", index=False)
    print(f"\nSaved model_performance.csv to {TABLE_DIR}")

    plot_model_comparison(results)

    best_name = results.iloc[0]["model"]
    best_model = fitted[best_name]
    print(f"\nBest model by cross-validated R2: {best_name}")

    y_pred_best = best_model.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred_best, best_name)
    plot_residuals(y_test, y_pred_best, best_name)

    # Save the linear regression coefficients separately — these are the
    # most directly interpretable for the "advertising impact" analysis,
    # regardless of which model wins on raw predictive accuracy.
    lr = fitted["Linear Regression"]
    coef_df = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": lr.coef_,
    }).sort_values("coefficient", ascending=False)
    coef_df.loc[len(coef_df)] = ["intercept", lr.intercept_]
    coef_df.to_csv(TABLE_DIR / "linear_regression_coefficients.csv", index=False)
    print("\nLinear Regression coefficients (Sales = intercept + sum(coef_i * spend_i)):")
    print(coef_df.round(4))

    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(lr, MODEL_DIR / "linear_regression_model.pkl")
    print(f"\nSaved best model ({best_name}) and linear regression model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
