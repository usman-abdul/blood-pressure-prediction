"""
Blood Pressure Prediction — Data Visualizations
Run: python notebooks/visualize.py
Saves all plots to notebooks/plots/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'BP_dataset_augmented.csv')
PLOT_DIR  = os.path.join(BASE_DIR, 'notebooks', 'plots')
DUMPS_DIR = os.path.join(BASE_DIR, 'dumps')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['Status'] = df['Hypertension'].map({0: 'Normal', 1: 'High'})

PALETTE = {'Normal': '#38a169', 'High': '#e53e3e'}
COLORS  = [PALETTE['Normal'], PALETTE['High']]

# ── Theme ──────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font='DejaVu Sans')
plt.rcParams.update({
    'figure.dpi': 130,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
})

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{name}.png'), bbox_inches='tight')
    plt.close()
    print(f'  Saved: plots/{name}.png')


# ── 1. Class Distribution ──────────────────────────────────────────────────
print('1. Class distribution...')
fig, ax = plt.subplots(figsize=(5, 4))
counts = df['Status'].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[PALETTE[s] for s in counts.index],
              width=0.5, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            str(val), ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_title('Class Distribution')
ax.set_ylabel('Number of Patients')
ax.set_xlabel('Blood Pressure Status')
ax.set_ylim(0, counts.max() * 1.15)
save('01_class_distribution')


# ── 2. Age Distribution by Status ─────────────────────────────────────────
print('2. Age distribution...')
fig, ax = plt.subplots(figsize=(7, 4))
for status, color in PALETTE.items():
    ax.hist(df[df['Status'] == status]['Age'], bins=20,
            alpha=0.65, color=color, label=status, edgecolor='white')
ax.set_title('Age Distribution by Blood Pressure Status')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Count')
ax.legend(title='Status')
save('02_age_distribution')


# ── 3. SBP vs DBP Scatter ─────────────────────────────────────────────────
print('3. SBP vs DBP scatter...')
fig, ax = plt.subplots(figsize=(7, 5))
for status, color in PALETTE.items():
    sub = df[df['Status'] == status]
    ax.scatter(sub['Systolic_BP'], sub['Diastolic_BP'],
               c=color, label=status, alpha=0.45, s=22, edgecolors='none')
ax.axvline(130, color='#718096', linestyle='--', linewidth=1, label='SBP = 130')
ax.axhline(80,  color='#a0aec0', linestyle='--', linewidth=1, label='DBP = 80')
ax.set_title('Systolic vs Diastolic Blood Pressure')
ax.set_xlabel('Systolic BP (mmHg)')
ax.set_ylabel('Diastolic BP (mmHg)')
ax.legend(title='Status', markerscale=1.5)
save('03_sbp_vs_dbp')


# ── 4. BMI Distribution by Status ─────────────────────────────────────────
print('4. BMI distribution...')
fig, ax = plt.subplots(figsize=(7, 4))
for status, color in PALETTE.items():
    sns.kdeplot(df[df['Status'] == status]['BMI'],
                ax=ax, color=color, label=status, fill=True, alpha=0.3, linewidth=2)
ax.set_title('BMI Distribution by Blood Pressure Status')
ax.set_xlabel('BMI (kg/m²)')
ax.set_ylabel('Density')
ax.legend(title='Status')
save('04_bmi_distribution')


# ── 5. Feature Correlation Heatmap ────────────────────────────────────────
print('5. Correlation heatmap...')
fig, ax = plt.subplots(figsize=(8, 6))
num_cols = ['Age', 'Height', 'Weight', 'Systolic_BP',
            'Diastolic_BP', 'Heart_Rate', 'BMI', 'Hypertension']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax,
            annot_kws={'size': 9}, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap')
save('05_correlation_heatmap')


# ── 6. Boxplots — Key Features by Status ──────────────────────────────────
print('6. Boxplots...')
features = ['Systolic_BP', 'Diastolic_BP', 'BMI', 'Age']
labels   = ['Systolic BP (mmHg)', 'Diastolic BP (mmHg)', 'BMI (kg/m²)', 'Age (years)']
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
for ax, feat, label in zip(axes, features, labels):
    sns.boxplot(data=df, x='Status', y=feat, hue='Status', palette=PALETTE,
                order=['Normal', 'High'], ax=ax, legend=False,
                width=0.5, linewidth=1.2, fliersize=3)
    ax.set_title(label)
    ax.set_xlabel('')
    ax.set_ylabel('')
fig.suptitle('Key Features by Blood Pressure Status', fontsize=14, fontweight='bold', y=1.02)
save('06_boxplots')


# ── 8. Model Feature Importance ───────────────────────────────────────────
print('7. Feature importance...')
try:
    model = joblib.load(os.path.join(DUMPS_DIR, 'bp_model-2.plk'))
    feat_names = ['Female', 'Male', 'Age', 'Height', 'Weight', 'SBP', 'DBP', 'HR', 'BMI']
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({'Feature': feat_names, 'Coefficient': coefs})
    coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values().index)

    colors = ['#e53e3e' if c > 0 else '#3182ce' for c in coef_df['Coefficient']]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(coef_df['Feature'], coef_df['Coefficient'],
            color=colors, edgecolor='white', height=0.6)
    ax.axvline(0, color='#718096', linewidth=1)
    ax.set_title('Logistic Regression — Feature Coefficients')
    ax.set_xlabel('Coefficient (log-odds)')
    red_patch  = mpatches.Patch(color='#e53e3e', label='Increases risk')
    blue_patch = mpatches.Patch(color='#3182ce', label='Decreases risk')
    ax.legend(handles=[red_patch, blue_patch])
    save('07_feature_importance')
except Exception as e:
    print(f'  Skipped: {e}')


print(f'\nAll plots saved to notebooks/plots/')
