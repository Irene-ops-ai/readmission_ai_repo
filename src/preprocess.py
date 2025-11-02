"""Preprocessing utilities for readmission prediction.
Contains functions to load data, basic cleaning, imputation, feature engineering, and splitting.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = ['age', 'num_prior_adm', 'length_of_stay', 'charlson', 'med_count']
CATEGORICAL_FEATURES = ['social_risk']  # binary in our synthetic data

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_clean(df):
    # Example cleaning: drop duplicates
    df = df.drop_duplicates().copy()
    # Ensure types
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype('int')
    return df

def impute(df):
    # Simple imputation for numeric features
    imp = SimpleImputer(strategy='median')
    df[NUMERIC_FEATURES] = imp.fit_transform(df[NUMERIC_FEATURES])
    # For categorical, fill missing with mode (0)
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna(0)
    return df

def feature_engineer(df):
    # Example derived features: medication burden per day of stay
    df['med_per_day'] = df['med_count'] / (df['length_of_stay'] + 1e-6)
    return df

def get_features_and_labels(df):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['med_per_day']]
    y = df['readmit_30d']
    return X, y

def train_val_test_split(X, y, test_size=0.15, val_size=0.15, random_state=42):
    # Split out test set first
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    # Split train/val
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative, random_state=random_state, stratify=y_trainval)
    return X_train, X_val, X_test, y_train, y_val, y_test
