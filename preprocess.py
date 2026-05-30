import pandas as pd
import numpy as np

TARGET_COL = "is_fraud"

FEATURES = [
    "amount",
    "session_duration",
    "authentication_attempts",
    "receiver_account_age",
    "geographic_disparity",
    "transaction_time_of_day",
    "unusual_device_flag",
    "unusual_ip_flag",
    "unusual_location_flag",
    "transaction_velocity"
]

def build_dataset(csv_path="fraud_dataset.csv"):
    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    X = df[FEATURES].copy()
    y = df[TARGET_COL].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    meta = {
        "feature_columns": FEATURES
    }

    return X, y, meta

def build_preprocessor(meta):
    
    from sklearn.preprocessing import StandardScaler
    return StandardScaler()

def prepare_input(payload_df, meta):
    df = payload_df.copy()

    for col in meta["feature_columns"]:
        if col not in df.columns:
            df[col] = 0

    df = df[meta["feature_columns"]]
    df = df.replace("", 0)
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in meta["feature_columns"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df