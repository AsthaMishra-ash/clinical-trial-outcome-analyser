import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

ages = np.random.randint(25, 75, n)
genders = np.random.choice(['Male', 'Female'], n)
treatment_group = np.random.choice(['Drug', 'Placebo'], n, p=[0.6, 0.4])
dosage = np.where(treatment_group == 'Drug', np.random.choice([50, 100, 150, 200], n), 0)
duration_weeks = np.random.randint(4, 24, n)
adverse_events = np.random.choice([0, 1], n, p=[0.75, 0.25])
comorbidities = np.random.choice([0, 1, 2], n, p=[0.5, 0.35, 0.15])

# Outcome logic: drug + higher dose + younger + fewer comorbidities = more likely to respond
prob = (
    0.3
    + 0.25 * (treatment_group == 'Drug')
    + 0.001 * dosage
    - 0.003 * ages
    - 0.08 * comorbidities
    - 0.05 * adverse_events
    + 0.005 * duration_weeks
)
prob = np.clip(prob, 0.05, 0.95)
outcome = np.random.binomial(1, prob)

df = pd.DataFrame({
    'patient_id': [f'P{str(i).zfill(4)}' for i in range(1, n+1)],
    'age': ages,
    'gender': genders,
    'treatment_group': treatment_group,
    'dosage_mg': dosage,
    'duration_weeks': duration_weeks,
    'adverse_events': adverse_events,
    'comorbidities': comorbidities,
    'outcome': outcome  # 1 = responded, 0 = did not respond
})

df.to_csv('trial_data.csv', index=False)
print(f"Dataset generated: {n} patients")
print(df.head())
print(f"\nOutcome distribution:\n{df['outcome'].value_counts()}")
