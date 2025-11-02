"""Generate a small synthetic dataset for readmission prediction.
Usage:
    python data/generate_synthetic.py --out data/sample_readmission.csv --n 1000
"""
import argparse
import numpy as np
import pandas as pd

def generate(n=1000, seed=42):
    rng = np.random.RandomState(seed)
    age = rng.randint(18, 90, size=n)
    num_prior_adm = rng.poisson(1.5, size=n)
    length_of_stay = rng.exponential(scale=3, size=n).round().astype(int) + 1
    charlson = rng.randint(0, 6, size=n)
    med_count = rng.poisson(5, size=n)
    # Social risk proxy (0 = low risk, 1 = high)
    social_risk = rng.binomial(1, p=0.15, size=n)
    # Outcome: readmitted within 30 days (synthetic rule + noise)
    score = (0.02*age + 0.5*num_prior_adm + 0.3*length_of_stay + 0.6*charlson
            + 0.4*med_count + 1.5*social_risk + rng.normal(scale=3, size=n))
    prob = 1 / (1 + np.exp(-0.01*(score-10)))
    readmit = rng.binomial(1, prob)
    df = pd.DataFrame({
        'age': age,
        'num_prior_adm': num_prior_adm,
        'length_of_stay': length_of_stay,
        'charlson': charlson,
        'med_count': med_count,
        'social_risk': social_risk,
        'readmit_30d': readmit
    })
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/sample_readmission.csv')
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()
    df = generate(n=args.n)
    df.to_csv(args.out, index=False)
    print(f"Wrote synthetic dataset to {args.out} with {len(df)} rows.")
