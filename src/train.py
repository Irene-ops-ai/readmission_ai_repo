"""Train a Gradient Boosting model on preprocessed data and save the model artifact.
Usage:
    python src/train.py --data_path data/sample_readmission.csv --model_out models/readmit_model.joblib
"""
import os
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from src.preprocess import load_data, basic_clean, impute, feature_engineer, get_features_and_labels, train_val_test_split

def main(data_path, model_out):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    df = load_data(data_path)
    df = basic_clean(df)
    df = impute(df)
    df = feature_engineer(df)
    X, y = get_features_and_labels(df)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # Validation AUC
    val_preds = model.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC: {auc:.4f}")

    # Save model and a simple metadata dict
    joblib.dump({'model': model, 'features': list(X.columns)}, model_out)
    print(f"Saved model to {model_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/sample_readmission.csv')
    parser.add_argument('--model_out', default='models/readmit_model.joblib')
    args = parser.parse_args()
    main(args.data_path, args.model_out)
