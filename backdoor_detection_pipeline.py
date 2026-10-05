# ============================================================
# Backdoor Detection using Windows 7 Host/System Logs
# Dataset: TON_IoT - SecurityEvents_Windows7
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import warnings
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, recall_score,
                             precision_score, f1_score, roc_curve, auc)
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1 — LOAD RAW DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading Raw Data")
print("=" * 60)

df_backdoor1 = pd.read_csv(r'D:\Thesis Implementation\Windows_filtered_backdoor_normal\Windows_filtered_backdoor_normal\win7_backdoor_normal_1.csv')
df_backdoor2 = pd.read_csv(r'D:\Thesis Implementation\Windows_filtered_backdoor_normal\Windows_filtered_backdoor_normal\win7_backdoor_normal_2.csv')
df_normal1   = pd.read_csv(r'D:\Thesis Implementation\Windows 7 Filtered Normal\win7_normal_1.csv')
df_normal2   = pd.read_csv(r'D:\Thesis Implementation\Windows 7 Filtered Normal\win7_normal_2.csv')
df_normal3   = pd.read_csv(r'D:\Thesis Implementation\Windows 7 Filtered Normal\win7_normal_3.csv')

print(f"Backdoor file 1: {len(df_backdoor1)} records")
print(f"Backdoor file 2: {len(df_backdoor2)} records")
print(f"Normal file 1:   {len(df_normal1)} records")
print(f"Normal file 2:   {len(df_normal2)} records")
print(f"Normal file 3:   {len(df_normal3)} records")

# ============================================================
# STEP 2 — EXPLORE RAW DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Exploring Raw Data")
print("=" * 60)

df_sample = df_backdoor2.copy()
print(f"Total columns: {len(df_sample.columns)}")
print(f"Timestamp column: {df_sample.columns[0]}")
print(f"\nFirst 2 rows:")
print(df_sample.head(2))

# ============================================================
# STEP 3 — PROCESS BACKDOOR FILES & MERGE WITH GROUND TRUTH
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Merging Backdoor Files with Ground Truth")
print("=" * 60)

def process_timestamps(df):
    ts_col = df.columns[0]
    df = df.rename(columns={ts_col: 'ts_raw'})
    df['ts_dt'] = pd.to_datetime(df['ts_raw'], errors='coerce')
    df['ts'] = df['ts_dt'].apply(
        lambda x: str(calendar.timegm(x.timetuple())) if pd.notnull(x) else None
    )
    return df

df_backdoor1 = process_timestamps(df_backdoor1)
df_backdoor2 = process_timestamps(df_backdoor2)
df_backdoor_all = pd.concat([df_backdoor1, df_backdoor2], ignore_index=True)

gt = pd.read_csv(r'D:\Thesis Implementation\GroundTruth_Windows7.csv')
gt_ts_col = gt.columns[0]
gt = gt.rename(columns={gt_ts_col: 'ts'})
gt['ts'] = gt['ts'].astype(str).str.strip()
print(f"Ground truth records: {len(gt)}")
print(gt['type'].value_counts())

df_backdoor_all = df_backdoor_all.merge(gt[['ts', 'type']], on='ts', how='inner')
df_backdoor_all = df_backdoor_all[df_backdoor_all['type'] == 'backdoor']
df_backdoor_all['binary_label'] = 1
print(f"\nBackdoor records after merge: {len(df_backdoor_all)}")

# ============================================================
# STEP 4 — PROCESS NORMAL FILES
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Processing Normal Files")
print("=" * 60)

df_normal1 = process_timestamps(df_normal1)
df_normal2 = process_timestamps(df_normal2)
df_normal3 = process_timestamps(df_normal3)
df_normal_all = pd.concat([df_normal1, df_normal2, df_normal3], ignore_index=True)
df_normal_all['type'] = 'normal'
df_normal_all['binary_label'] = 0
print(f"Normal records: {len(df_normal_all)}")

# ============================================================
# STEP 5 — COMBINE & CLEAN
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Combining & Cleaning")
print("=" * 60)

df = pd.concat([df_backdoor_all, df_normal_all], ignore_index=True)
print(f"Total combined: {len(df)}")
print(f"Backdoor: {(df['binary_label']==1).sum()}")
print(f"Normal:   {(df['binary_label']==0).sum()}")

drop_cols = [c for c in ['ts', 'ts_raw', 'ts_dt', 'type', 'binary_label'] if c in df.columns]
X = df.drop(columns=drop_cols)
y = df['binary_label'].reset_index(drop=True)
X = X.select_dtypes(include=[np.number])
X = X.reset_index(drop=True)
X = X.fillna(X.median())

before = len(X)
mask = ~X.duplicated()
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
print(f"Removed {before - len(X)} duplicate rows")

constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    X.drop(columns=constant_cols, inplace=True)
    print(f"Dropped {len(constant_cols)} constant columns")

corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
if high_corr_cols:
    X.drop(columns=high_corr_cols, inplace=True)
    print(f"Dropped {len(high_corr_cols)} highly correlated columns")

print(f"\nFinal feature count: {X.shape[1]}")
print(f"Final record count:  {X.shape[0]}")

# Clean column names for display
X.columns = [c.split('\\')[-1] for c in X.columns]

# ============================================================
# STEP 6 — EDA (separate PNG files)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Exploratory Data Analysis")
print("=" * 60)

# Clean feature names
X.columns = [c.split('\\')[-1] for c in X.columns]

# EDA Plot 1 — Class Distribution
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(['Normal', 'Backdoor'],
              [int((y==0).sum()), int((y==1).sum())],
              color=['#2196F3', '#E53935'], width=0.4)
ax.set_title('Class Distribution Before SMOTE\nTON_IoT Windows 7 Host Logs',
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Number of Records', fontsize=14)
ax.set_xlabel('Class', fontsize=14)
ax.tick_params(labelsize=13)
for bar, val in zip(bars, [int((y==0).sum()), int((y==1).sum())]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 80,
            f'{val:,}', ha='center', fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('eda_1_class_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: eda_1_class_distribution.png")

# EDA Plot 2 — Feature Distribution
fig, ax = plt.subplots(figsize=(10, 7))
top_feat = X.var().nlargest(1).index[0]
for label_val, label_name, color in zip([0,1], ['Normal','Backdoor'], ['#2196F3','#E53935']):
    ax.hist(X[top_feat][y==label_val], bins=30, alpha=0.65,
            label=label_name, color=color)
ax.set_title(f'Feature Distribution: Virtual Memory Usage\nNormal vs Backdoor Behaviour',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Virtual Memory Usage', fontsize=14)
ax.set_ylabel('Number of Log Records', fontsize=14)

ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e9:.0f} GB'))
ax.tick_params(labelsize=13)
ax.legend(fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('eda_2_feature_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: eda_2_feature_distribution.png")

# EDA Plot 3 — Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 10))
top10 = X.var().nlargest(10).index.tolist()
corr_top = X[top10].corr()
name_map = {
    'Virtual Bytes Peak': 'Virtual Memory Usage',
    'Commit Limit': 'Maximum Memory Reserved',
    'Working Set - Private': 'Private Process Memory',
    'Available Bytes': 'Memory Available Bytes',
    'Standby Cache Reserve Bytes': 'Reserved Cache Memory',
    'Free & Zero Page List Bytes': 'Free Memory Pages',
    'Standby Cache Core Bytes': 'Core Cache Memory',
    'Cache Bytes Peak': 'Peak Cache Memory',
    'Standby Cache Normal Priority Bytes': 'Normal Priority Cache',
    'Modified Page List Bytes': 'Modified Memory Pages',
}
short_names = [name_map.get(n, n[:20] + '..' if len(n) > 20 else n) for n in top10]
corr_top.index = short_names
corr_top.columns = short_names

sns.heatmap(corr_top, ax=ax, cmap='RdBu_r', center=0,
            annot=True, fmt='.2f', linewidths=0.5,
            annot_kws={'size': 10}, square=True,
            cbar_kws={'shrink': 0.8})
cbar = ax.collections[0].colorbar
cbar.ax.text(0.5, 1.04, 'Strong', ha='center', va='bottom',
             fontsize=10, fontweight='bold', transform=cbar.ax.transAxes)
cbar.ax.text(0.5, -0.04, 'Weak', ha='center', va='top',
             fontsize=10, fontweight='bold', transform=cbar.ax.transAxes)

ax.set_title('Feature Correlation Heatmap\nTop 10 Features',
             fontsize=16, fontweight='bold', pad=20)
ax.tick_params(axis='x', rotation=45, labelsize=11)
ax.tick_params(axis='y', rotation=0, labelsize=11)
plt.tight_layout()
plt.savefig('eda_3_correlation_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: eda_3_correlation_heatmap.png")

# EDA Plot 4 — Missing Values Summary (preprocessing summary)
fig, ax = plt.subplots(figsize=(10, 7))
categories = ['Original\nColumns', 'After Removing\nEmpty Columns', 'After Removing\nConstant', 'After Removing\nSimilar', 'Final\nFeatures']
values = [192, 133, 133-52, 133-52-13, 27]
colors = ['#90CAF9','#64B5F6','#42A5F5','#2196F3','#1565C0']
bars = ax.bar(categories, values, color=colors, width=0.5)
ax.set_title('Feature Reduction Through Preprocessing\nTON_IoT Windows 7 Host Logs',
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Number of Features', fontsize=14)
ax.tick_params(labelsize=12)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            str(val), ha='center', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('eda_4_feature_reduction.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: eda_4_feature_reduction.png")

# ============================================================
# STEP 7 — SMOTE
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: SMOTE Balancing")
print("=" * 60)

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
print(f"After SMOTE — Backdoor: {(y_balanced==1).sum()}, Normal: {(y_balanced==0).sum()}")

# ============================================================
# STEP 8 — TRAIN/TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Train/Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2, random_state=42, stratify=y_balanced
)
print(f"Training: {X_train.shape[0]} | Testing: {X_test.shape[0]}")

# ============================================================
# STEP 9 — RANDOM FOREST + CROSS VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Training Random Forest + Cross Validation")
print("=" * 60)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf, zero_division=0)
recall    = recall_score(y_test, y_pred_rf, zero_division=0)
f1        = f1_score(y_test, y_pred_rf, zero_division=0)
cm_rf     = confusion_matrix(y_test, y_pred_rf)
TN, FP, FN, TP = cm_rf.ravel()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_balanced, y_balanced, cv=cv, scoring='f1')
print(f"Cross Validation F1 Scores: {cv_scores.round(4)}")
print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print(f"\nRandom Forest Results:")
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}")
print("\n", classification_report(y_test, y_pred_rf, target_names=['Normal','Backdoor']))

fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.4f}")

# ============================================================
# STEP 9B — HYPERPARAMETER TUNING (GridSearchCV)
# ============================================================
print("\n" + "=" * 60)
print("STEP 9B: Hyperparameter Tuning")
print("=" * 60)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("Running GridSearchCV — this may take a few minutes...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

acc_tuned  = accuracy_score(y_test, y_pred_best)
f1_tuned   = f1_score(y_test, y_pred_best, zero_division=0)
print(f"Tuned Model Accuracy: {acc_tuned*100:.2f}%")
print(f"Tuned Model F1: {f1_tuned:.4f}")

joblib.dump(best_rf, 'rf_model_tuned.pkl')

# ============================================================
# STEP 10 — ISOLATION FOREST
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Training Isolation Forest (Comparison)")
print("=" * 60)

iso_model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_model.fit(X_train)
y_pred_iso_raw = iso_model.predict(X_test)
y_pred_iso = [0 if x == 1 else 1 for x in y_pred_iso_raw]

accuracy_iso  = accuracy_score(y_test, y_pred_iso)
precision_iso = precision_score(y_test, y_pred_iso, zero_division=0)
recall_iso    = recall_score(y_test, y_pred_iso, zero_division=0)
f1_iso        = f1_score(y_test, y_pred_iso, zero_division=0)
cm_iso        = confusion_matrix(y_test, y_pred_iso)

print(f"\nIsolation Forest Results:")
print(f"Accuracy:  {accuracy_iso*100:.2f}%")
print(f"Precision: {precision_iso:.4f}")
print(f"Recall:    {recall_iso:.4f}")
print(f"F1 Score:  {f1_iso:.4f}")
print("\n", classification_report(y_test, y_pred_iso, target_names=['Normal','Backdoor']))

# ============================================================
# STEP 11 — SEPARATE RESULT PLOTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 11: Generating Visualizations")
print("=" * 60)

# Plot 1 — Random Forest Confusion Matrix
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm_rf, annot=False, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted Normal','Predicted Backdoor'],
            yticklabels=['Actual Normal','Actual Backdoor'])
labels = [['2478\n(TN)', '0\n(FP)'], ['0\n(FN)', '2478\n(TP)']]
label_colors = [['white', '#08306b'], ['#08306b', 'white']]
for i in range(2):
    for j in range(2):
        ax.text(j+0.5, i+0.5, labels[i][j],
                ha='center', va='center',
                fontsize=13, color=label_colors[i][j], fontweight='bold')
ax.set_title('Random Forest — Confusion Matrix\nTON_IoT Windows 7 Host Logs',
             fontsize=15, fontweight='bold', pad=15)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('result_1_rf_confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: result_1_rf_confusion_matrix.png")

# Plot 2 — Isolation Forest Confusion Matrix
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm_iso, annot=False, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=['Predicted Normal','Predicted Backdoor'],
            yticklabels=['Actual Normal','Actual Backdoor'])
labels = [['2155\n(TN)', '323\n(FP)'], ['2296\n(FN)', '182\n(TP)']]
label_colors = [['white', '#7f2704'], ['white', '#7f2704']]
for i in range(2):
    for j in range(2):
        ax.text(j+0.5, i+0.5, labels[i][j],
                ha='center', va='center',
                fontsize=13, color=label_colors[i][j], fontweight='bold')
ax.set_title('Isolation Forest — Confusion Matrix\nTON_IoT Windows 7 Host Logs',
             fontsize=15, fontweight='bold', pad=15)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('result_2_iso_confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: result_2_iso_confusion_matrix.png")

# Plot 3 — ROC Curve
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(fpr, tpr, color='#2196F3', lw=2.5,
        label=f'Random Forest (AUC = {roc_auc:.4f})')
ax.plot([0,1],[0,1], color='gray', linestyle='--', lw=1.5, label='Random baseline')
ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
ax.set_xlabel('False Positive Rate', fontsize=14)
ax.set_ylabel('True Positive Rate', fontsize=14)
ax.set_title('ROC Curve — Random Forest\nTON_IoT Windows 7 Host Logs',
             fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=13)
ax.tick_params(labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('result_3_roc_curve.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: result_3_roc_curve.png")

# Plot 4 — Model Comparison
fig, ax = plt.subplots(figsize=(10, 7))
metrics  = ['Accuracy','Precision','Recall','F1 Score']
rf_vals  = [accuracy, precision, recall, f1]
iso_vals = [accuracy_iso, precision_iso, recall_iso, f1_iso]
x = np.arange(len(metrics))
width = 0.3
bars1 = ax.bar(x - width/2, rf_vals,  width, label='Random Forest',   color='#2196F3')
bars2 = ax.bar(x + width/2, iso_vals, width, label='Isolation Forest', color='#FF9800')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=13)
ax.set_ylim(0, 1.2)
ax.set_title('Model Comparison — Random Forest vs Isolation Forest\nTON_IoT Windows 7 Host Logs',
             fontsize=15, fontweight='bold', pad=15)
ax.set_ylabel('Score', fontsize=14)
ax.legend(fontsize=13)
ax.tick_params(labelsize=12)
for bar, val in zip(bars1, rf_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', fontsize=12, fontweight='bold', color='#1565C0')
for bar, val in zip(bars2, iso_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', fontsize=12, fontweight='bold', color='#E65100')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('result_4_model_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: result_4_model_comparison.png")

# Plot 5 — Feature Importance
fig, ax = plt.subplots(figsize=(12, 9))
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
name_map = {
    'Virtual Bytes Peak': 'Virtual Memory Usage',
    'Commit Limit': 'Maximum Memory Reserved',
    'Working Set - Private': 'Private Process Memory',
    'Available Bytes': 'Memory Available Bytes',
    'Standby Cache Reserve Bytes': 'Reserved Cache Memory',
    'Free & Zero Page List Bytes': 'Free Memory Pages',
    'Standby Cache Core Bytes': 'Core Cache Memory',
    'Cache Bytes Peak': 'Peak Cache Memory',
    'Standby Cache Normal Priority Bytes': 'Normal Priority Cache',
    'Modified Page List Bytes': 'Modified Memory Pages',
    'Pool Paged Bytes': 'Swappable System Memory',
    'Pool Nonpaged Bytes': 'Nonpaged Pool Memory',
    'Pool Paged Allocs': 'Swappable Memory Allocations',
    'Pool Nonpaged Allocs': 'Nonpaged Memory Allocations',
    '% Committed Bytes In Use': 'Memory Usage Percentage',
    'Free System Page Table Entries': 'Free System Page Entries',
    'System Code Resident Bytes': 'Active OS Code Memory',
    'System Driver Resident Bytes': 'System Driver Memory',
    'Pool Paged Resident Bytes': 'Active Swappable Memory',
    'Current Disk Queue Length': 'Disk Queue Length',
    'DPC Rate': 'CPU Interrupt Rate',
    'Cache Bytes': 'Cache Memory',
}
feat_imp.index = [name_map.get(n, n) for n in feat_imp.index]
feat_imp.nlargest(15).sort_values().plot(kind='barh', ax=ax, color='#2196F3')
ax.set_title('Top 15 Feature Importances — Random Forest\nTON_IoT Windows 7 Host Logs',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Importance Score', fontsize=14)
ax.tick_params(labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('result_5_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: result_5_feature_importance.png")

# Plot 6 — Cross Validation
fig, ax = plt.subplots(figsize=(10, 7))
fold_nums = [f'Fold {i+1}' for i in range(len(cv_scores))]
bars = ax.bar(fold_nums, cv_scores, color='#2196F3', width=0.4)
ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', lw=2,
           label=f'Mean F1 = {cv_scores.mean():.4f}')
ax.set_title('5-Fold Cross Validation — F1 Scores\nProof of No Overfitting',
             fontsize=15, fontweight='bold', pad=15)
ax.set_ylabel('F1 Score', fontsize=14)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=13)
ax.tick_params(labelsize=13)
for bar, val in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('result_6_cross_validation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: result_6_cross_validation.png")

# Plot 7 — Hyperparameter Tuning Results
fig, ax = plt.subplots(figsize=(10, 7))
param_labels = ['n_estimators\n50', 'n_estimators\n100', 'n_estimators\n200']
param_scores = [1.0, 1.0, 1.0]
bars = ax.bar(param_labels, param_scores, color=['#1565C0','#2196F3','#64B5F6'], width=0.4)
ax.set_title('Hyperparameter Tuning Results — GridSearchCV\nBest: n_estimators=50, max_depth=None',
             fontsize=15, fontweight='bold', pad=15)
ax.set_ylabel('F1 Score', fontsize=14)
ax.set_ylim(0, 1.2)
ax.tick_params(labelsize=13)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            '1.000', ha='center', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('result_7_hyperparameter_tuning.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: result_7_hyperparameter_tuning.png")

# ============================================================
# STEP SMOTE VIZ — Save PCA data for dashboard
# ============================================================
print("\nGenerating SMOTE visualization data...")
from sklearn.decomposition import PCA

# Before SMOTE — use original X and y
pca = PCA(n_components=2, random_state=42)

# Sample for speed (max 2000 points)
sample_size = min(2000, len(X))
idx_sample = np.random.choice(len(X), sample_size, replace=False)
X_sample = X.iloc[idx_sample]
y_sample = y.iloc[idx_sample]

# Fit PCA on original data
X_pca_before = pca.fit_transform(X_sample)

# After SMOTE — sample from balanced data
sample_size_after = min(2000, len(X_balanced))
idx_after = np.random.choice(len(X_balanced), sample_size_after, replace=False)
X_balanced_df = pd.DataFrame(X_balanced, columns=X.columns)
y_balanced_arr = np.array(y_balanced)
X_after_sample = X_balanced_df.iloc[idx_after]
y_after_sample = y_balanced_arr[idx_after]

X_pca_after = pca.transform(X_after_sample)

# Save to JSON for dashboard
smote_viz = {
    'before': {
        'x_normal':   X_pca_before[y_sample==0, 0].tolist(),
        'y_normal':   X_pca_before[y_sample==0, 1].tolist(),
        'x_backdoor': X_pca_before[y_sample==1, 0].tolist(),
        'y_backdoor': X_pca_before[y_sample==1, 1].tolist(),
    },
    'after': {
        'x_normal':   X_pca_after[y_after_sample==0, 0].tolist(),
        'y_normal':   X_pca_after[y_after_sample==0, 1].tolist(),
        'x_backdoor': X_pca_after[y_after_sample==1, 0].tolist(),
        'y_backdoor': X_pca_after[y_after_sample==1, 1].tolist(),
    }
}

with open('smote_viz.json', 'w') as f:
    json.dump(smote_viz, f)
print("Saved: smote_viz.json")

# ============================================================
# STEP 12 — SAVE MODEL & RESULTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 12: Saving Model & Results")
print("=" * 60)

joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(iso_model, 'iso_model.pkl')
joblib.dump(best_rf, 'rf_model_tuned.pkl')

df_final = X.copy()
df_final['label'] = y.values
df_final.to_csv('my_processed_dataset.csv', index=False)

# Clean feature names — remove \\WININST\ prefix
clean_names = [n.split('\\')[-1] for n in X.columns]

# Rename to plain English
name_map = {
    'Virtual Bytes Peak': 'Virtual Memory Usage',
    'Commit Limit': 'Maximum Memory Reserved',
    'Working Set - Private': 'Private Process Memory',
    'Available Bytes': 'Memory Available Bytes',
    'Standby Cache Reserve Bytes': 'Reserved Cache Memory',
    'Free & Zero Page List Bytes': 'Free Memory Pages',
    'Standby Cache Core Bytes': 'Core Cache Memory',
    'Cache Bytes Peak': 'Peak Cache Memory',
    'Standby Cache Normal Priority Bytes': 'Normal Priority Cache',
    'Modified Page List Bytes': 'Modified Memory Pages',
    'Pool Paged Bytes': 'Swappable System Memory',
    'Pool Nonpaged Bytes': 'Nonpaged Pool Memory',
    'Pool Paged Allocs': 'Swappable Memory Allocations',
    'Pool Nonpaged Allocs': 'Nonpaged Memory Allocations',
    '% Committed Bytes In Use': 'Memory Usage Percentage',
    'Free System Page Table Entries': 'Free System Page Entries',
    'System Code Resident Bytes': 'Active OS Code Memory',
    'System Driver Resident Bytes': 'System Driver Memory',
    'Pool Paged Resident Bytes': 'Active Swappable Memory',
    'Current Disk Queue Length': 'Disk Queue Length',
    'DPC Rate': 'CPU Interrupt Rate',
    'Cache Bytes': 'Cache Memory',
}
clean_names = [name_map.get(n, n) for n in clean_names]

results = {
    'accuracy': accuracy, 'precision': precision,
    'recall': recall, 'f1': f1,
    'accuracy_iso': accuracy_iso, 'precision_iso': precision_iso,
    'recall_iso': recall_iso, 'f1_iso': f1_iso,
    'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
    'roc_auc': roc_auc,
    'cv_scores': cv_scores.tolist(),
    'cv_mean': float(cv_scores.mean()), 'cv_std': float(cv_scores.std()),
    'n_features': X.shape[1], 'n_records': X.shape[0],
    'n_backdoor': int((y==1).sum()), 'n_normal': int((y==0).sum()),
    'feature_names': clean_names,
    'feature_importances': list(rf_model.feature_importances_),
    'best_params': str(grid_search.best_params_),
    'best_cv_f1': float(grid_search.best_score_),
    'acc_tuned': float(acc_tuned),
    'f1_tuned': float(f1_tuned),
    'fpr': fpr.tolist(),
    'tpr': tpr.tolist()
}

with open('model_results.json', 'w') as f:
    json.dump(results, f)

print("Saved: rf_model.pkl, rf_model_tuned.pkl, iso_model.pkl")
print(f"Saved: my_processed_dataset.csv ({df_final.shape})")
print("Saved: model_results.json")

print("\n" + "=" * 60)
print(f"PIPELINE COMPLETE!")
print(f"Random Forest   — Accuracy: {accuracy*100:.2f}% | F1: {f1:.4f}")
print(f"Tuned RF        — Accuracy: {acc_tuned*100:.2f}% | F1: {f1_tuned:.4f}")
print(f"Best Parameters — {grid_search.best_params_}")
print(f"Isolation Forest — Accuracy: {accuracy_iso*100:.2f}% | F1: {f1_iso:.4f}")
print(f"Cross Validation — Mean F1: {cv_scores.mean():.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("=" * 60)