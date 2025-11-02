# 🏥 Patient Readmission Risk Prediction using AI

## 📘 Overview
This project applies the AI Development Workflow to a real-world healthcare problem — predicting whether a patient is likely to be readmitted within 30 days after discharge.
The goal is to help hospitals optimize care, reduce readmission rates, and improve patient outcomes using a machine learning model trained on synthetic patient data.
## 🎯 Objectives
- Predict the probability of 30-day readmission using patient demographics and medical history.
- Support hospital decision-making for follow-up care.
- Demonstrate an end-to-end AI workflow — from problem definition to deployment
## 👩‍⚕️ Stakeholders
- Hospital Administration: To allocate resources efficiently and monitor performance.
- Medical Staff: To identify high-risk patients for targeted care.
## 📊 Key Performance Indicator (KPI)
Primary KPI: Area Under the ROC Curve (AUC) — to measure how well the model distinguishes between readmitted and non-readmitted patients.

## 🧠 AI Workflow
1. Problem Definition – Identify the goal and stakeholders.
2. Data Collection – Synthetic data simulating hospital records.
3. Preprocessing – Cleaning, encoding, scaling, and splitting.
4. Model Development – Logistic Regression model for interpretability.
5. Evaluation – Metrics: Precision, Recall, F1, AUC.
6. Deployment – Exported model ready for API or web integration.

## 🧩 Repository Structure
readmission_ai_repo/
│
├── data/
│   └── sample_readmission.csv        # Synthetic dataset
│
├── models/
│   └── readmit_model.joblib          # Saved trained model
│
├── src/
│   ├── train.py                      # Model training script
│   ├── evaluate.py                   # Evaluation script
│   └── utils.py                      # Helper functions
│
├── requirements.txt                  # Dependencies
└── README.md                         # Project documentation

## ⚙️ Setup Instructions
### Create a virtual environment
python -m venv venv
source venv/Scripts/activate   # On Windows
### Install dependencies
pip install -r requirements.txt
### Run training
python -m src.train
### Evaluate model
python -m src.evaluate
## 📈 Example Output
Validation AUC: 0.4235
[[353 136]
 [ 86 424]]
Precision: 0.7571
Recall: 0.8314
F1-score: 0.7925
AUC: 0.8375

## 🧰 Technologies Used
- Python 3.10+
- scikit-learn
- pandas / numpy
- joblib
- matplotlib



