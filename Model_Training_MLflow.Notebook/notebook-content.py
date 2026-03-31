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

%pip install imbalanced-learn xgboost

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read from Silver layer (no prefix)
df_silver = spark.table("silver_transactions")

feature_cols = [f"V{i}" for i in range(1, 29)] + [
    "Amount_log", "Amount_percentile",
    "V1_abs", "V2_abs", "V3_abs", "V4_abs", "V5_abs",
    "V1_V2_interaction", "V1_Amount_interaction"
]

pdf = df_silver.select(feature_cols + ["Class"]).toPandas()

X = pdf[feature_cols]
y = pdf["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} records")
print(f"Test set: {X_test.shape[0]} records")
print(f"Training fraud rate: {y_train.mean() * 100:.4f}%")
print(f"Test fraud rate: {y_test.mean() * 100:.4f}%")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(f"Class distribution in training set:")
print(f"{y_train.value_counts().to_dict()}")
print(f"\nFraud rate: {y_train.mean() * 100:.2f}%")
print(f"Dataset is already balanced — skipping SMOTE")

# Use the scaled training data directly (no resampling needed)
X_train_resampled = X_train_scaled
y_train_resampled = y_train

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve, 
    average_precision_score
)

# Set the MLflow experiment
EXPERIMENT_NAME = "fraud-detection-experiment"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from sklearn.linear_model import LogisticRegression

with mlflow.start_run(run_name="logistic_regression"):
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("smote_applied", True)
    mlflow.log_param("smote_ratio", 0.5)
    mlflow.log_param("n_features", len(feature_cols))
    
    # Train model
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_model.fit(X_train_resampled, y_train_resampled)
    
    # Predict on test set (using original scaled test data, NOT resampled)
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_metric("avg_precision", avg_precision)
    
    # Log confusion matrix as artifact
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                          index=['Actual Legit', 'Actual Fraud'],
                          columns=['Predicted Legit', 'Predicted Fraud'])
    cm_df.to_csv("/tmp/confusion_matrix_lr.csv")
    mlflow.log_artifact("/tmp/confusion_matrix_lr.csv")
    
    # Log the model
    mlflow.sklearn.log_model(lr_model, artifact_path="model")
    
    print("=== Logistic Regression Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"\nConfusion Matrix:\n{cm_df}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from xgboost import XGBClassifier

with mlflow.start_run(run_name="xgboost"):
    scale_pos_weight = 1.0  # Already balanced dataset
    
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_param("smote_applied", False)
    mlflow.log_param("n_features", len(feature_cols))
    
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='aucpr'
    )
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    y_pred = xgb_model.predict(X_test_scaled)
    y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_metric("avg_precision", avg_precision)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv("/tmp/feature_importance_xgb.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance_xgb.csv")
    
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                          index=['Actual Legit', 'Actual Fraud'],
                          columns=['Predicted Legit', 'Predicted Fraud'])
    cm_df.to_csv("/tmp/confusion_matrix_xgb.csv")
    mlflow.log_artifact("/tmp/confusion_matrix_xgb.csv")
    
    mlflow.sklearn.log_model(xgb_model, artifact_path="model")
    
    print("=== XGBoost Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"\nTop 10 Features:\n{feature_importance.head(10)}")
    print(f"\nConfusion Matrix:\n{cm_df}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))

# Retrain quickly for ROC comparison (or store predictions from above)
models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Model Comparison')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Find the best run based on AUC-ROC
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.auc_roc DESC"]
)

# Display comparison table
print("Model Comparison:")
print(runs[["run_id", "params.model_type", "metrics.auc_roc", 
            "metrics.precision", "metrics.recall", "metrics.f1_score"]].head())

# Get the best run
best_run = runs.iloc[0]
best_run_id = best_run["run_id"]
best_model_type = best_run["params.model_type"]
best_auc = best_run["metrics.auc_roc"]

print(f"\nBest model: {best_model_type}")
print(f"Best AUC-ROC: {best_auc:.4f}")
print(f"Best Run ID: {best_run_id}")

# Register the best model in MLflow Model Registry
model_uri = f"runs:/{best_run_id}/model"
model_name = "fraud-detection-model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print(f"\nModel registered: {model_name}")
print(f"Version: {registered_model.version}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
