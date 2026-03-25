# Blood Pressure Prediction

A machine learning system that predicts **hypertension risk** (high or normal blood pressure) from basic patient health data. Built to demonstrate a complete ML workflow — from raw data to a deployable Flask web app — including real-world problem solving around data quality and model behavior.

---

## Problem Statement

Hypertension affects millions of people and often goes undetected until it causes serious complications. This project uses clinical measurements to automatically classify a patient's blood pressure status, supporting earlier and easier screening.

---

## Features Used

| Feature | Description |
|---|---|
| Sex | Patient gender (Male / Female) |
| Age | Age in years |
| Height | Height in cm |
| Weight | Weight in kg |
| Systolic Blood Pressure | Systolic BP in mmHg |
| Diastolic Blood Pressure | Diastolic BP in mmHg |
| Heart Rate | Heart rate in beats/min |
| BMI | Body Mass Index (kg/m²) |

**Dataset:** 219 original records + 800 synthetic records = 1,019 total — `data/BP_dataset_augmented.csv`
**Target:** `Hypertension` — `1` (High) or `0` (Normal)

---

## Model Choice

**Logistic Regression** was selected because:

- The task is binary classification (High vs. Normal), which is exactly what logistic regression is designed for
- It is highly interpretable — you can inspect which features drive the prediction
- It works well on clean, structured datasets without overfitting
- It provides a strong, honest baseline before reaching for more complex models

---

## How It Works

```
Patient Data → Preprocessing → Logistic Regression Model → Prediction
```

1. **Input** — Patient fills in health metrics (age, sex, BP readings, BMI, etc.)
2. **Encoding** — Sex is one-hot encoded (Male/Female → numeric)
3. **Scaling** — Numeric features are standardized using `StandardScaler`
4. **Prediction** — The trained model outputs `0` (Normal) or `1` (High)
5. **Output** — Result is displayed as **"Normal"** or **"High"**

---

## Data Augmentation and Model Improvement

### The Problem with the Original Dataset

The original dataset (219 records) had a near-perfect feature boundary:

- Every patient with SBP ≤ 139 was labeled **Normal**
- Every patient with SBP ≥ 140 was labeled **High**
- Zero overlap between the two classes

This caused the model to learn one rule: **if SBP ≥ 140 → High, else → Normal**. It completely ignored DBP, BMI, Age, and all other features. In practice, a patient with SBP = 138, DBP = 92, and BMI = 38 would be predicted Normal — which is clinically wrong.

This is a classic example of a model that scores well on paper but fails in the real world.

### What We Did

800 synthetic patient records were generated to introduce realistic variability:

- Hypertension labeled using the ACC/AHA guideline: **SBP ≥ 130 OR DBP ≥ 80**
- Feature distributions based on real clinical ranges (age, BMI, weight, BP)
- Natural noise added so the boundary between classes is not perfectly clean
- Combined with the original 219 records for a total of 1,019 samples

The model was then retrained on this augmented dataset.

### Why Accuracy Went Down — and Why That's a Good Thing

| | Original Model | Improved Model |
|---|---|---|
| Train Accuracy | 98.86% | 89.82% |
| Test Accuracy | 97.73% | 89.71% |
| Overfit gap | 1.1% | 0.1% |
| CV F1 (5-fold) | — | 0.89 ± 0.03 |
| Uses multiple features | No (SBP only) | Yes |

The original model's 97% accuracy was misleading — it was essentially memorizing a single threshold. The improved model at 90% is genuinely learning from DBP, BMI, Age, and other features, and generalizes far better to unseen inputs.

---

## Results

Evaluated on a held-out test set (20% of augmented data, 204 samples):

| Metric | Value |
|---|---|
| Train Accuracy | 89.82% |
| Test Accuracy | 89.71% |
| CV F1 (5-fold) | 0.89 ± 0.03 |

**Classification Report (Test Set)**

```
              precision    recall  f1-score

      Normal       0.89      0.89      0.89
        High       0.90      0.91      0.90

    accuracy                           0.90
```

**Top feature weights learned by the model:**

| Feature | Coefficient | Role |
|---|---|---|
| DBP | +2.36 | Strongest predictor |
| SBP | +2.21 | Strong predictor |
| BMI | +0.98 | Meaningful contributor |
| Weight | -0.50 | Moderate |
| Height | +0.33 | Moderate |

---

## Engineering Insight

> High accuracy does not always mean a good model.

The original model scored 97% — but it was only learning one thing: whether SBP was above or below 140. That's not machine learning, that's a threshold check.

A well-built model should:
- Use all available features meaningfully
- Generalize to inputs it hasn't seen before
- Reflect real-world complexity, not just dataset patterns

This project demonstrates that understanding *why* a model performs the way it does is just as important as the performance number itself. Diagnosing data bias, identifying shortcut learning, and improving generalization are core ML engineering skills.

---

## Project Structure

```
Blood_Pressure_Prediction/
├── data/
│   ├── BP_dataset.csv                  # Original dataset (219 records)
│   └── BP_dataset_augmented.csv        # Augmented dataset (1,019 records)
├── notebooks/
│   └── blood_pressure_model.ipynb      # EDA, training, and evaluation
├── src/
│   ├── app.py                          # Flask web app
│   ├── predict.py                      # Standalone prediction script
│   ├── retrain.py                      # Retrain and save model artifacts
│   └── generate_data.py               # Synthetic data generation
├── dumps/                              # Saved model artifacts (generated)
├── requirements.txt
└── README.md
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/blood-pressure-prediction.git
cd blood-pressure-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Generate augmented data and retrain**
```bash
python src/generate_data.py
python src/retrain.py
```

**4. Run the web app**
```bash
python src/app.py
```

Visit `http://127.0.0.1:5000` in your browser.

**5. Explore the notebook**

Open `notebooks/blood_pressure_model.ipynb` in Jupyter for the full EDA, preprocessing, and evaluation pipeline.

---

## Tech Stack

Python · scikit-learn · pandas · numpy · matplotlib · seaborn · Flask · Jupyter

---

## Notes & Limitations

- Synthetic data was generated to improve generalization, not to simulate a real clinical dataset
- The model has not been validated on external clinical data
- For any medical application, results should be reviewed by a qualified professional
- This project is intended for learning and demonstration purposes
