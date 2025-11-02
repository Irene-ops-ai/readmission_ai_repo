"""Evaluate a saved model on a dataset and print common metrics and confusion matrix.
Usage:
    python src/evaluate.py --data_path data/sample_readmission.csv --model_path models/readmit_model.joblib
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from src.preprocess import load_data, basic_clean, impute, feature_engineer, get_features_and_labels

def main(data_path, model_path):
    artifact = joblib.load(model_path)
    model = artifact['model']
    features = artifact['features']

    df = load_data(data_path)
    df = basic_clean(df)
    df = impute(df)
    df = feature_engineer(df)
    X, y = get_features_and_labels(df)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]

    cm = confusion_matrix(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, probs)

    print("Confusion Matrix:\n", cm)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/sample_readmission.csv')
    parser.add_argument('--model_path', default='models/readmit_model.joblib')
    args = parser.parse_args()
    main(args.data_path, args.model_path)
