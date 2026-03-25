"""
Standalone prediction script for Blood Pressure classification.
Loads the trained model and makes a prediction for a single patient.
"""

import numpy as np
import joblib

# Load saved model artifacts
model = joblib.load('../dumps/bp_model-2.plk')
scaler = joblib.load('../dumps/scaler.pk1')

def predict(sex, age, height, weight, systolic_bp, diastolic_bp, heart_rate, bmi):
    """
    Predict hypertension status for a patient.

    Parameters:
        sex (str): 'Male' or 'Female'
        age (float): Age in years
        height (float): Height in cm
        weight (float): Weight in kg
        systolic_bp (float): Systolic blood pressure in mmHg
        diastolic_bp (float): Diastolic blood pressure in mmHg
        heart_rate (float): Heart rate in beats/min
        bmi (float): Body Mass Index

    Returns:
        str: 'High' or 'Normal'
    """
    # One-hot encode sex: Female -> [1, 0], Male -> [0, 1]
    sex_encoded = [0, 1] if sex.strip().title() == 'Male' else [1, 0]

    features = sex_encoded + [age, height, weight, systolic_bp, diastolic_bp, heart_rate, bmi]
    features = np.array(features).reshape(1, -1)

    # Scale numeric features (skip the first 2 one-hot columns)
    features[:, 2:] = scaler.transform(features[:, 2:])

    prediction = model.predict(features)
    return 'High' if prediction[0] == 1 else 'Normal'


if __name__ == '__main__':
    # Example patient
    result = predict(
        sex='Female',
        age=45,
        height=152,
        weight=63,
        systolic_bp=161,
        diastolic_bp=89,
        heart_rate=97,
        bmi=27.27
    )
    print(f'Predicted Blood Pressure Status: {result}')
