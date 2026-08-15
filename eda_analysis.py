import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.titlesize': 16,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': True
})

OUTPUT_DIR = "EDA_Reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory created/verified: '{OUTPUT_DIR}/'")


DATASET_PATH = "project_risk_dataset.csv"
print(f"\n{'='*60}\nSTEP 1: LOADING DATASET\n{'='*60}")
df = pd.read_csv(DATASET_PATH)
print(f"Successfully loaded dataset from '{DATASET_PATH}'.")


print(f"\n{'='*60}\nSTEP 2: DATASET SHAPE\n{'='*60}")
rows, cols = df.shape
print(f"Number of Rows    : {rows:,}")
print(f"Number of Columns : {cols}")


print(f"\n{'='*60}\nSTEP 3: COLUMN NAMES & DATA TYPES\n{'='*60}")
schema_info = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': df.dtypes.values,
    'Non-Null Count': df.notnull().sum().values,
    'Null Count': df.isnull().sum().values
})
print(schema_info.to_string(index=False))


print(f"\n{'='*60}\nSTEP 4: MISSING VALUES CHECK\n{'='*60}")
missing_series = df.isnull().sum()
total_missing = missing_series.sum()
print(f"Total Missing Values in Dataset: {total_missing}")
if total_missing > 0:
    missing_df = pd.DataFrame({
        'Missing Count': missing_series[missing_series > 0],
        'Percentage (%)': (missing_series[missing_series > 0] / len(df)) * 100
    })
    print(missing_df.to_string())
else:
    print("Zero missing values detected across all columns. Data is 100% complete.")


print(f"\n{'='*60}\nSTEP 5: DUPLICATE RECORDS CHECK\n{'='*60}")
duplicate_count = df.duplicated().sum()
print(f"Number of Duplicate Records: {duplicate_count} ({(duplicate_count/rows)*100:.2f}%)")


print(f"\n{'='*60}\nSTEP 8: FEATURE IDENTIFICATION\n{'='*60}")
id_col = ['project_id'] if 'project_id' in df.columns else []
target_col = 'risk_category'

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in id_col]
numerical_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col not in id_col]

print(f"Identifier Feature ({len(id_col)})     : {id_col}")
print(f"Target Feature                      : '{target_col}'")
print(f"Categorical Features ({len(categorical_cols)}) : {categorical_cols}")
print(f"Numerical Features ({len(numerical_cols)})   : {numerical_cols}")


print(f"\n{'='*60}\nSTEP 6: STATISTICAL SUMMARY (NUMERICAL FEATURES)\n{'='*60}")
stats_df = df[numerical_cols].describe().T
stats_df['skewness'] = df[numerical_cols].skew()
stats_df['kurtosis'] = df[numerical_cols].kurtosis()
print(stats_df[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness']].round(2).to_string())


print(f"\n{'='*60}\nSTEP 7: CATEGORICAL FEATURES ANALYSIS\n{'='*60}")
for col in categorical_cols:
    uniques = df[col].unique()
    print(f"\nFeature: '{col}' (Unique Count: {len(uniques)})")
    val_counts = df[col].value_counts(normalize=True) * 100
    for val, pct in val_counts.items():
        count = df[col].value_counts()[val]
        print(f"  - {val}: {count:,} ({pct:.2f}%)")


print(f"\n{'='*60}\nSTEP 9: TARGET CLASS DISTRIBUTION ('{target_col}')\n{'='*60}")
target_counts = df[target_col].value_counts()
target_pcts = df[target_col].value_counts(normalize=True) * 100
target_summary = pd.DataFrame({'Count': target_counts, 'Percentage (%)': target_pcts})
print(target_summary.to_string())


print(f"\n{'='*60}\nSTEP 11: OUTLIER IDENTIFICATION (IQR METHOD)\n{'='*60}")
outlier_summary = []
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    count = len(outliers)
    pct = (count / len(df)) * 100
    outlier_summary.append({
        'Feature': col,
        'Q1 (25%)': round(Q1, 2),
        'Q3 (75%)': round(Q3, 2),
        'IQR': round(IQR, 2),
        'Lower Bound': round(lower_bound, 2),
        'Upper Bound': round(upper_bound, 2),
        'Outlier Count': count,
        'Outlier Pct (%)': round(pct, 2)
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values(by='Outlier Pct (%)', ascending=False)
print(outlier_df.to_string(index=False))


print(f"\n{'='*60}\nSTEP 10: GENERATING VISUALIZATIONS IN '{OUTPUT_DIR}'\n{'='*60}")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Categorical Features & Target Class Distribution", fontsize=18, fontweight='bold', y=1.02)
axes = axes.flatten()

risk_order = ['Low', 'Medium', 'High', 'Critical']
palette_risk = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e67e22', 'Critical': '#e74c3c'}

sns.countplot(data=df, x='risk_category', order=risk_order, palette=palette_risk, ax=axes[0])
axes[0].set_title("Target Distribution: Risk Category", fontweight='bold')
axes[0].set_xlabel("Risk Category")
axes[0].set_ylabel("Count")
for p in axes[0].patches:
    height = p.get_height()
    axes[0].annotate(f'{height:,}\n({(height/rows)*100:.1f}%)', (p.get_x() + p.get_width()/2., height/2),
                     ha='center', va='center', color='white', fontweight='bold', fontsize=10)

cat_to_plot = [c for c in categorical_cols if c != 'risk_category'][:7]
for idx, col in enumerate(cat_to_plot, start=1):
    sns.countplot(data=df, y=col, palette="Blues_r", order=df[col].value_counts().index, ax=axes[idx])
    axes[idx].set_title(f"Distribution of {col.replace('_', ' ').title()}", fontweight='bold')
    axes[idx].set_xlabel("Count")
    axes[idx].set_ylabel(col.replace('_', ' ').title())

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "01_target_distribution_countplots.png")
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig1_path}")


key_num_features = [
    'planned_duration_days', 'actual_duration_days', 'budget_usd', 'actual_cost_usd',
    'cost_overrun_pct', 'schedule_overrun_pct', 'team_size', 'risk_score'
]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Histograms & Density Distributions of Key Numerical Features", fontsize=18, fontweight='bold', y=1.02)
axes = axes.flatten()

for idx, col in enumerate(key_num_features):
    sns.histplot(data=df, x=col, kde=True, color='#3498db', ax=axes[idx], bins=30)
    axes[idx].set_title(f"Distribution: {col.replace('_', ' ').title()}", fontweight='bold')
    axes[idx].set_xlabel(col.replace('_', ' ').title())
    axes[idx].set_ylabel("Frequency")

plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "02_numerical_histograms_distributions.png")
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig2_path}")


fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Outlier Inspection via Boxplots", fontsize=18, fontweight='bold', y=1.02)
axes = axes.flatten()

for idx, col in enumerate(key_num_features):
    sns.boxplot(data=df, x=col, color='#9b59b6', ax=axes[idx], flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3})
    axes[idx].set_title(f"Boxplot: {col.replace('_', ' ').title()}", fontweight='bold')
    axes[idx].set_xlabel(col.replace('_', ' ').title())

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, "03_outlier_boxplots.png")
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig3_path}")


plt.figure(figsize=(16, 14))
corr_matrix = df[numerical_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
plt.title("Numerical Features Correlation Heatmap", fontsize=18, fontweight='bold', pad=20)
fig4_path = os.path.join(OUTPUT_DIR, "04_correlation_heatmap.png")
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig4_path}")


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Relationship: Continuous Risk Score vs Risk Category (Leakage Inspection)", fontsize=16, fontweight='bold')

sns.boxplot(data=df, x='risk_category', y='risk_score', order=risk_order, palette=palette_risk, ax=axes[0])
axes[0].set_title("Risk Score Distribution per Risk Category (Boxplot)", fontweight='bold')

sns.violinplot(data=df, x='risk_category', y='risk_score', order=risk_order, palette=palette_risk, ax=axes[1])
axes[1].set_title("Risk Score Density per Risk Category (Violin Plot)", fontweight='bold')

plt.tight_layout()
fig5_path = os.path.join(OUTPUT_DIR, "05_risk_score_vs_category_relationship.png")
plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig5_path}")


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Bivariate Analysis of Risk Drivers across Risk Categories", fontsize=16, fontweight='bold')

sns.scatterplot(data=df.sample(5000, random_state=42), x='cost_overrun_pct', y='schedule_overrun_pct',
                hue='risk_category', hue_order=risk_order, palette=palette_risk, alpha=0.6, ax=axes[0])
axes[0].set_title("Cost Overrun % vs Schedule Overrun % by Risk Category", fontweight='bold')
axes[0].set_xlabel("Cost Overrun (%)")
axes[0].set_ylabel("Schedule Overrun (%)")

sns.boxplot(data=df, x='risk_category', y='defect_count', order=risk_order, palette=palette_risk, ax=axes[1])
axes[1].set_title("Defect Count across Risk Categories", fontweight='bold')
axes[1].set_xlabel("Risk Category")
axes[1].set_ylabel("Defect Count")

plt.tight_layout()
fig6_path = os.path.join(OUTPUT_DIR, "06_feature_risk_bivariate_analysis.png")
plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig6_path}")

print(f"\n{'='*60}\nEDA EXECUTION & VISUALIZATION COMPLETE!\n{'='*60}")
