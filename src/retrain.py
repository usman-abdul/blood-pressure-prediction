"""
Retrains the model on the augmented dataset and saves all artifacts to dumps/.
Run this to regenerate model files.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use augmented dataset if available, otherwise fall back to original
aug_path = os.path.join(BASE_DIR, 'data', 'BP_dataset_augmented.csv')
orig_path = os.path.join(BASE_DIR, 'data', 'BP_dataset.csv')
data_path = aug_path if os.path.exists(aug_path) else orig_path
print(f'Using dataset: {os.path.basename(data_path)}')

df = pd.read_csv(data_path)

# Handle both original and augmented column formats
if 'Systolic_BP' in df.columns:
    # augmented format
    data = df[['Sex','Age','Height','Weight','Systolic_BP','Diastolic_BP','Heart_Rate','BMI','Hypertension']]
else:
    df.columns = ['Num','subject_ID','Sex','Age','Height','Weight',
                  'Systolic_BP','Diastolic_BP','Heart_Rate','BMI','Hypertension']
    data = df.drop(columns=['Num','subject_ID'])

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode Sex (column 0) with OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features (skip first 2 one-hot columns)
sc = StandardScaler()
X_train[:, 2:] = sc.fit_transform(X_train[:, 2:])
X_test[:, 2:] = sc.transform(X_test[:, 2:])

# Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, lr.predict(X_train))
test_acc  = accuracy_score(y_test,  lr.predict(X_test))
cv_scores = cross_val_score(lr, X, y, cv=5, scoring='f1_macro')

print(f'\nTrain Accuracy : {train_acc:.4f}')
print(f'Test Accuracy  : {test_acc:.4f}')
print(f'Overfit gap    : {train_acc - test_acc:.4f}')
print(f'CV F1 (5-fold) : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')
print('\nClassification Report:')
print(classification_report(y_test, lr.predict(X_test), target_names=['Normal', 'High']))

# Save artifacts
dumps_dir = os.path.join(BASE_DIR, 'dumps')
os.makedirs(dumps_dir, exist_ok=True)
joblib.dump(lr, os.path.join(dumps_dir, 'bp_model-2.plk'))
joblib.dump(sc, os.path.join(dumps_dir, 'scaler.pk1'))
joblib.dump(le, os.path.join(dumps_dir, 'label_encoder.pk1'))
print('Model artifacts saved to dumps/')
