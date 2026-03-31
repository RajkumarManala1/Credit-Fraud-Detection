# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "f7f578d8-5e58-4281-82c7-c91aa8d5814b",
# META       "default_lakehouse_name": "fraud_lakehouse",
# META       "default_lakehouse_workspace_id": "633902ff-9ec7-4005-a95a-6550752cf8cf",
# META       "known_lakehouses": [
# META         {
# META           "id": "f7f578d8-5e58-4281-82c7-c91aa8d5814b"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Read Silver data (no prefix needed)
df_silver = spark.table("silver_transactions")

# Convert to pandas — 568K rows is fine for pandas
pdf = df_silver.toPandas()
print(f"Dataset shape: {pdf.shape}")
print(f"\nClass distribution:\n{pdf['Class'].value_counts()}")
print(f"\nFraud percentage: {pdf['Class'].mean() * 100:.4f}%")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

class_counts = pdf['Class'].value_counts()
axes[0].bar(['Legitimate (0)', 'Fraud (1)'], class_counts.values, 
            color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Transaction Count by Class')
axes[0].set_ylabel('Count')
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 1000, str(v), ha='center', fontweight='bold')

axes[1].pie(class_counts.values, labels=['Legitimate', 'Fraud'], 
            autopct='%1.3f%%', colors=['#2ecc71', '#e74c3c'],
            startangle=90)
axes[1].set_title('Class Distribution')

plt.tight_layout()
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

### Amount distribution

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Amount distribution by class
pdf[pdf['Class'] == 0]['Amount'].hist(bins=50, ax=axes[0], alpha=0.7, 
                                       color='#2ecc71', label='Legitimate')
pdf[pdf['Class'] == 1]['Amount'].hist(bins=50, ax=axes[0], alpha=0.7, 
                                       color='#e74c3c', label='Fraud')
axes[0].set_title('Amount Distribution by Class')
axes[0].set_xlabel('Amount')
axes[0].legend()

# Log Amount distribution
pdf[pdf['Class'] == 0]['Amount_log'].hist(bins=50, ax=axes[1], alpha=0.7, 
                                           color='#2ecc71', label='Legitimate')
pdf[pdf['Class'] == 1]['Amount_log'].hist(bins=50, ax=axes[1], alpha=0.7, 
                                           color='#e74c3c', label='Fraud')
axes[1].set_title('Log(Amount) Distribution by Class')
axes[1].set_xlabel('Log(Amount + 1)')
axes[1].legend()

plt.tight_layout()
plt.show()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

###Correlation heatmap of top features

# Top features correlation with fraud
correlations = pdf[[f'V{i}' for i in range(1, 29)] + ['Amount']].corrwith(pdf['Class'])
top_features = correlations.abs().sort_values(ascending=False).head(10)

print("Top 10 features most correlated with fraud:")
print(top_features)

# Heatmap of top features
top_feature_names = top_features.index.tolist() + ['Class']
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(pdf[top_feature_names].corr(), annot=True, fmt='.2f', 
            cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Correlation Heatmap — Top Features vs Fraud')
plt.tight_layout()
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

###Box plots of top features by class

top_6 = correlations.abs().sort_values(ascending=False).head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, feat in enumerate(top_6):
    row, col_idx = idx // 3, idx % 3
    pdf.boxplot(column=feat, by='Class', ax=axes[row][col_idx])
    axes[row][col_idx].set_title(f'{feat} by Class')
    axes[row][col_idx].set_xlabel('Class (0=Legit, 1=Fraud)')

plt.suptitle('Top 6 Features — Fraud vs Legitimate', fontsize=14)
plt.tight_layout()
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
