from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    expected = {"Time", "Amount", "Class"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Colonnes attendues manquantes. Colonnes présentes: {df.columns.tolist()}")
    return df

def add_features(df):
    df = df.copy()
    df["hour"] = (df["Time"] // 3600) % 24
    df["log_amount"] = np.log1p(df["Amount"])
    return df

def split_xy(df):
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return X, y

def train_test_split_stratified(X, y, test_size, random_state):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

def scale_amount_features(X_train, X_test):

    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    to_scale = [c for c in ["Amount", "log_amount"] if c in X_train.columns]

    if to_scale:
        X_train[to_scale] = scaler.fit_transform(X_train[to_scale])
        X_test[to_scale] = scaler.transform(X_test[to_scale])

    return X_train, X_test, scaler
