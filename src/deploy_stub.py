"""A minimal Flask deployment stub demonstrating how the model could be served.
Not production-ready — for educational/demo purposes only.
Usage:
    python src/deploy_stub.py --model_path models/readmit_model.joblib
Then POST JSON to http://127.0.0.1:5000/predict
Example payload:
{
    "age": 65,
    "num_prior_adm": 2,
    "length_of_stay": 4,
    "charlson": 2,
    "med_count": 6,
    "social_risk": 0
}
"""
import argparse
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model_artifact = None
FEATURES = None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Basic input validation
    try:
        x = [data[f] for f in FEATURES]
    except Exception as e:
        return jsonify({'error': 'Invalid input format', 'details': str(e)}), 400
    proba = model_artifact['model'].predict_proba([x])[0,1]
    return jsonify({'readmit_30d_prob': float(proba)})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='models/readmit_model.joblib')
    args = parser.parse_args()
    model_artifact = joblib.load(args.model_path)
    FEATURES = model_artifact['features']
    app.run(debug=True)
