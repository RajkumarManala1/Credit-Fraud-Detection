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

df = spark.sql("SELECT * FROM fraud_lakehouse.dbo.silver_transactions LIMIT 1000")
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_silver = spark.table("silver_transactions")
print(f"Silver records: {df_silver.count()}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, count, sum as spark_sum, avg, round as spark_round, when

df_fraud_by_amount = df_silver.groupBy("Amount_category").agg(
    count("*").alias("total_transactions"),
    spark_sum(when(col("Class") == 1, 1).otherwise(0)).alias("fraud_count"),
    spark_sum(when(col("Class") == 0, 1).otherwise(0)).alias("legit_count"),
    spark_round(avg("Amount"), 2).alias("avg_amount"),
    spark_round(
        spark_sum(when(col("Class") == 1, 1).otherwise(0)) * 100.0 / count("*"), 4
    ).alias("fraud_rate_pct")
)

df_fraud_by_amount.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold_fraud_by_amount")

df_fraud_by_amount.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_fraud_by_quartile = df_silver.groupBy("Amount_quartile").agg(
    count("*").alias("total_transactions"),
    spark_sum(when(col("Class") == 1, 1).otherwise(0)).alias("fraud_count"),
    spark_round(avg("Amount"), 2).alias("avg_amount"),
    spark_round(
        spark_sum(when(col("Class") == 1, 1).otherwise(0)) * 100.0 / count("*"), 4
    ).alias("fraud_rate_pct")
).orderBy("Amount_quartile")

df_fraud_by_quartile.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold_fraud_by_quartile")

df_fraud_by_quartile.show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import current_timestamp

df_kpis = df_silver.agg(
    count("*").alias("total_transactions"),
    spark_sum(when(col("Class") == 1, 1).otherwise(0)).alias("total_frauds"),
    spark_sum(when(col("Class") == 0, 1).otherwise(0)).alias("total_legit"),
    spark_round(
        spark_sum(when(col("Class") == 1, 1).otherwise(0)) * 100.0 / count("*"), 4
    ).alias("overall_fraud_rate"),
    spark_round(
        spark_sum(when(col("Class") == 1, col("Amount")).otherwise(0)), 2
    ).alias("total_fraud_amount"),
    spark_round(avg("Amount"), 2).alias("avg_transaction_amount")
).withColumn("report_generated_at", current_timestamp())

df_kpis.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold_kpis")

df_kpis.show(truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

feature_cols_for_summary = [f"V{i}" for i in range(1, 29)] + ["Amount", "Amount_log"]

df_feature_summary = df_silver.groupBy("Class").agg(
    *[spark_round(avg(c), 4).alias(f"avg_{c}") for c in feature_cols_for_summary]
)

df_feature_summary.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold_feature_summary")

print("Gold feature summary table created")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
