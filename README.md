# Hospital Readmission Prediction (Example)

**Author:** Irene Ouko

This repository contains example code for the AI assignment: predicting 30-day hospital readmission risk.
The code is illustrative and intended for educational purposes. It includes:

- data/: synthetic data generator and small sample dataset
- src/: preprocessing, training, evaluation, and a deployment stub
- requirements.txt: Python package requirements
- LICENSE: MIT
- .gitignore

## Quick start
1. Create a Python environment (recommended): `python -m venv venv && source venv/bin/activate`
2. Install requirements: `pip install -r requirements.txt`
3. Generate synthetic data: `python data/generate_synthetic.py --out data/sample_readmission.csv`
4. Preprocess & train: `python src/train.py --data_path data/sample_readmission.csv --model_out models/readmit_model.joblib`
5. Evaluate: `python src/evaluate.py --data_path data/sample_readmission.csv --model_path models/readmit_model.joblib`
6. (Optional) Run deployment stub: `python src/deploy_stub.py` and POST JSON to `/predict`

## Notes
- This is a template. Replace synthetic data with your real dataset and update feature lists accordingly.
- The code is well-commented to help you adapt it for your project submission.
