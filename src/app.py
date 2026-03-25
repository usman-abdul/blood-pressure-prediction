import os
import numpy as np
import joblib
from flask import Flask, render_template, request

# Resolve paths relative to this file so the app works regardless of where it's run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMPS_DIR = os.path.join(BASE_DIR, 'dumps')

app = Flask(__name__)

# Load trained model artifacts
loaded_model = joblib.load(os.path.join(DUMPS_DIR, 'bp_model-2.plk'))
scaler = joblib.load(os.path.join(DUMPS_DIR, 'scaler.pk1'))
label_encoder = joblib.load(os.path.join(DUMPS_DIR, 'label_encoder.pk1'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sex = request.form['sex'].strip().title()
        if sex == 'Male':
            sex_encoded = [0, 1]
        elif sex == 'Female':
            sex_encoded = [1, 0]
        else:
            return render_template('result.html', prediction="Invalid sex value. Enter 'Male' or 'Female'.")

        form_features = [
            float(request.form['age']),
            float(request.form['height']),
            float(request.form['weight']),
            float(request.form['s_b_p']),
            float(request.form['d_b_p']),
            float(request.form['heart_rate']),
            float(request.form['bmi']),
        ]

        features = np.array(sex_encoded + form_features).reshape(1, -1)
        features[:, 2:] = scaler.transform(features[:, 2:])

        prediction = loaded_model.predict(features)
        result = 'High' if prediction[0] == 1 else 'Normal'
        return render_template('result.html', prediction=result)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
