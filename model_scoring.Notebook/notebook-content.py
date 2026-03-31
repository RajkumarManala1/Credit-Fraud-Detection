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

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler

model_name = "fraud-detection-model"
model_version = 1

model_uri = f"models:/{model_name}/{model_version}"
loaded_model = mlflow.pyfunc.load_model(model_uri)

print(f"Loaded model: {model_name} v{model_version}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_silver = spark.table("silver_transactions")

feature_cols = [f"V{i}" for i in range(1, 29)] + [
    "Amount_log", "Amount_percentile",
    "V1_abs", "V2_abs", "V3_abs", "V4_abs", "V5_abs",
    "V1_V2_interaction", "V1_Amount_interaction"
]

pdf = df_silver.select(["id"] + feature_cols + ["Amount", "Class"]).toPandas()
X_score = pdf[feature_cols]

scaler = StandardScaler()
X_score_scaled = scaler.fit_transform(X_score)

print(f"Records to score: {len(pdf)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

###Generate Predictions

import numpy as np

best_run_id = mlflow.search_runs(
    experiment_ids=[mlflow.get_experiment_by_name("fraud-detection-experiment").experiment_id],
    order_by=["metrics.auc_roc DESC"]
).iloc[0]["run_id"]

model_uri = f"runs:/{best_run_id}/model"
sklearn_model = mlflow.sklearn.load_model(model_uri)

predictions = sklearn_model.predict(X_score_scaled)
probabilities = sklearn_model.predict_proba(X_score_scaled)[:, 1]

pdf["fraud_prediction"] = predictions
pdf["fraud_probability"] = np.round(probabilities, 6)

pdf["risk_category"] = pd.cut(
    pdf["fraud_probability"],
    bins=[-0.001, 0.3, 0.7, 1.001],
    labels=["Low", "Medium", "High"]
)

print(f"Prediction distribution:\n{pdf['fraud_prediction'].value_counts()}")
print(f"\nRisk category distribution:\n{pdf['risk_category'].value_counts()}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

###Write scored predictions to Gold layer

from pyspark.sql.functions import current_timestamp, lit

df_scored = spark.createDataFrame(
    pdf[["id", "Amount", "Class", "fraud_prediction", 
         "fraud_probability", "risk_category"]]
)

df_scored = df_scored \
    .withColumn("scored_at", current_timestamp()) \
    .withColumn("model_name", lit(model_name)) \
    .withColumn("model_version", lit(str(model_version)))

df_scored.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold_fraud_scores")

print(f"Scored {df_scored.count()} transactions written to gold_fraud_scores")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

###Create Model performance summary

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

perf = pd.DataFrame([{
    "model_name": model_name,
    "model_version": model_version,
    "accuracy": round(accuracy_score(pdf["Class"], pdf["fraud_prediction"]), 4),
    "precision": round(precision_score(pdf["Class"], pdf["fraud_prediction"]), 4),
    "recall": round(recall_score(pdf["Class"], pdf["fraud_prediction"]), 4),
    "f1_score": round(f1_score(pdf["Class"], pdf["fraud_prediction"]), 4),
    "auc_roc": round(roc_auc_score(pdf["Class"], pdf["fraud_probability"]), 4),
    "total_scored": len(pdf),
    "total_flagged": int(pdf["fraud_prediction"].sum())
}])

df_perf = spark.createDataFrame(perf)
df_perf = df_perf.withColumn("scored_at", current_timestamp())

df_perf.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold_model_performance")

df_perf.show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
