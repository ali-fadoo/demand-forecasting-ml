import pandas as pd


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # calendar features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # grouped target
    g = df.groupby(["store", "item"], sort=False)["sales"]

    # lags
    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = g.shift(lag)

    # rolling stats (use shift(1) to prevent leakage)
    for w in (7, 14, 28):
        df[f"roll_mean_{w}"] = g.transform(lambda s: s.shift(1).rolling(w).mean())
        df[f"roll_std_{w}"] = g.transform(lambda s: s.shift(1).rolling(w).std())

    return df
