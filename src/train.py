import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.data import load_data
from src.features import make_features


# -------------------- Metrics --------------------

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


# -------------------- Walk-forward validation --------------------

def walk_forward_validation(df, features, target, model, horizon=30, step=30):
    metrics = []

    df = df.sort_values("date").reset_index(drop=True)

    start_date = df["date"].min() + np.timedelta64(365, "D")
    end_date = df["date"].max() - np.timedelta64(horizon, "D")

    current = start_date
    while current <= end_date:
        train_df = df[df["date"] < current]
        test_df = df[
            (df["date"] >= current)
            & (df["date"] < current + np.timedelta64(horizon, "D"))
        ]

        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics.append({
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": rmse(y_test, preds),
            "MAPE": mape(y_test, preds),
        })

        current += np.timedelta64(step, "D")

    return metrics


# -------------------- Main --------------------

def main():
    df = load_data("data/raw/train.csv")
    print("Raw shape:", df.shape)

    df = make_features(df)
    df = df.dropna(subset=["lag_1", "lag_7", "roll_mean_7"])

    # ---- Sanity checks (VERY important) ----
    print("\nSanity check (lags & rolling):")
    print(df[["date", "store", "item", "sales", "lag_1", "roll_mean_7"]].head(10))

    example = df[(df["store"] == 1) & (df["item"] == 1)]
    print("\nSingle series check (store 1, item 1):")
    print(example.tail(10)[["date", "sales", "lag_1", "roll_mean_7"]])

    target = "sales"
    features = [
        "dayofweek", "weekofyear", "month", "year",
        "lag_1", "lag_7", "lag_14", "lag_28",
        "roll_mean_7", "roll_mean_14", "roll_mean_28",
        "roll_std_7", "roll_std_14", "roll_std_28",
    ]

    # ---- Time-based split ----
    cutoff = df["date"].max() - np.timedelta64(30, "D")
    train_df = df[df["date"] <= cutoff]
    test_df = df[df["date"] > cutoff]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # ---- Baseline (lag-1) ----
    y_pred_naive = X_test["lag_1"].values
    print("\nBaseline (lag_1)")
    print("  MAE :", mean_absolute_error(y_test, y_pred_naive))
    print("  RMSE:", rmse(y_test, y_pred_naive))
    print("  MAPE:", mape(y_test, y_pred_naive))

    # ---- Ridge Regression ----
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("ridge", Ridge(alpha=1.0)),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nRidge Regression")
    print("  MAE :", mean_absolute_error(y_test, y_pred))
    print("  RMSE:", rmse(y_test, y_pred))
    print("  MAPE:", mape(y_test, y_pred))

    # ---- Walk-forward validation ----
    wf_metrics = walk_forward_validation(
        df, features, target, model, horizon=30, step=30
    )

    print("\nWalk-forward validation (Ridge)")
    print("Avg MAE :", np.mean([m["MAE"] for m in wf_metrics]))
    print("Avg RMSE:", np.mean([m["RMSE"] for m in wf_metrics]))
    print("Avg MAPE:", np.mean([m["MAPE"] for m in wf_metrics]))

    # ---- Visual check ----
    plt.figure(figsize=(12, 4))
    plt.plot(test_df["date"][:300], y_test.values[:300], label="Actual")
    plt.plot(test_df["date"][:300], y_pred[:300], label="Prediction")
    plt.title("Actual vs Predicted Demand (sample)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
