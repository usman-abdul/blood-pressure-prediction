"""
Generates realistic synthetic blood pressure data to augment the original dataset.

Based on real-world clinical distributions:
- Hypertension defined as SBP >= 130 OR DBP >= 80 (ACC/AHA 2017 guidelines)
- Features are correlated realistically (e.g. higher BMI -> higher BP tendency)
- Adds natural noise so the boundary is not perfectly clean
"""

import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
np.random.seed(42)
N = 800  # synthetic samples to generate

def generate_patient(hypertensive):
    """Generate one realistic patient record."""
    if hypertensive:
        age    = np.random.normal(62, 12)
        bmi    = np.random.normal(28, 5)
        weight = np.random.normal(78, 14)
        height = np.random.normal(163, 9)
        sbp    = np.random.normal(152, 14)   # elevated, with spread
        dbp    = np.random.normal(88, 10)
        hr     = np.random.normal(76, 11)
    else:
        age    = np.random.normal(45, 15)
        bmi    = np.random.normal(23, 4)
        weight = np.random.normal(63, 12)
        height = np.random.normal(163, 9)
        sbp    = np.random.normal(115, 12)   # normal range, with spread
        dbp    = np.random.normal(72, 9)
        hr     = np.random.normal(72, 10)

    sex = np.random.choice(['Male', 'Female'])

    # Clip to physiologically plausible ranges
    age    = int(np.clip(age, 18, 90))
    height = int(np.clip(height, 140, 200))
    weight = round(np.clip(weight, 35, 130), 1)
    sbp    = int(np.clip(sbp, 80, 200))
    dbp    = int(np.clip(dbp, 40, 120))
    hr     = int(np.clip(hr, 45, 115))
    bmi    = round(np.clip(weight / ((height / 100) ** 2), 14, 45), 2)

    # Label using ACC/AHA guideline: SBP>=130 OR DBP>=80
    label = 1 if (sbp >= 130 or dbp >= 80) else 0

    return [sex, age, height, weight, sbp, dbp, hr, bmi, label]


rows = [generate_patient(i % 2 == 0) for i in range(N)]
synth_df = pd.DataFrame(rows, columns=[
    'Sex', 'Age', 'Height', 'Weight',
    'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'BMI', 'Hypertension'
])

# Load and align original dataset
orig = pd.read_csv(os.path.join(BASE_DIR, 'data', 'BP_dataset.csv'))
orig.columns = ['Num','subject_ID','Sex','Age','Height','Weight',
                'Systolic_BP','Diastolic_BP','Heart_Rate','BMI','Hypertension']
orig = orig[['Sex','Age','Height','Weight','Systolic_BP','Diastolic_BP','Heart_Rate','BMI','Hypertension']]

combined = pd.concat([orig, synth_df], ignore_index=True)
combined.to_csv(os.path.join(BASE_DIR, 'data', 'BP_dataset_augmented.csv'), index=False)

print(f'Original rows : {len(orig)}')
print(f'Synthetic rows: {len(synth_df)}')
print(f'Combined rows : {len(combined)}')
print(f'Class balance : {combined.Hypertension.value_counts().to_dict()}')
print('Saved to data/BP_dataset_augmented.csv')
