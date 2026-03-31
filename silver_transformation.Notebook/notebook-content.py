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

df_bronze = spark.table("abfss://633902ff-9ec7-4005-a95a-6550752cf8cf@onelake.dfs.fabric.microsoft.com/f7f578d8-5e58-4281-82c7-c91aa8d5814b/Tables/dbo/bronze_transactions")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

print(f"Bronze records: {df_bronze.count()}")
df_bronze.printSchema()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, count, when, isnan, isnull
from pyspark.sql.types import DoubleType, FloatType

# Check for nulls
null_counts = df_bronze.select([
    count(
        when(
            isnull(c) | (isnan(c) if isinstance(df_bronze.schema[c].dataType, (DoubleType, FloatType)) else isnull(c)),
            c
        )
    ).alias(c)
    for c in df_bronze.columns
])
print("Null counts per column:")
null_counts.show(truncate=False)

# Check for duplicates
total = df_bronze.count()
distinct = df_bronze.select("id").distinct().count()
print(f"Total records: {total}")
print(f"Distinct IDs: {distinct}")
print(f"Duplicates: {total - distinct}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col

# Remove duplicates
df_deduped = df_bronze.dropDuplicates(["id"])

# Cast columns to correct types
feature_cols = [f"V{i}" for i in range(1, 29)]

df_typed = df_deduped
for col_name in feature_cols:
    df_typed = df_typed.withColumn(col_name, col(col_name).cast("double"))

df_typed = df_typed \
    .withColumn("Amount", col("Amount").cast("double")) \
    .withColumn("Class", col("Class").cast("integer")) \
    .withColumn("id", col("id").cast("integer"))

print(f"Records after dedup: {df_typed.count()}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import (
    col, when, log1p, abs as spark_abs, 
    percent_rank, ntile, round as spark_round
)
from pyspark.sql.window import Window

# 1. Log-transform Amount (reduces skewness)
df_features = df_typed.withColumn("Amount_log", log1p(col("Amount")))

# 2. Amount bins
df_features = df_features.withColumn(
    "Amount_category",
    when(col("Amount") < 50, "Low")
    .when(col("Amount") < 200, "Medium")
    .when(col("Amount") < 1000, "High")
    .otherwise("Very_High")
)

# 3. Absolute values of key PCA components
for i in range(1, 6):
    df_features = df_features.withColumn(
        f"V{i}_abs", spark_abs(col(f"V{i}"))
    )

# 4. Interaction features
df_features = df_features \
    .withColumn("V1_V2_interaction", col("V1") * col("V2")) \
    .withColumn("V1_Amount_interaction", col("V1") * col("Amount_log"))

# 5. Amount percentile rank
window_spec = Window.orderBy("Amount")
df_features = df_features.withColumn(
    "Amount_percentile", spark_round(percent_rank().over(window_spec), 4)
)

# 6. Amount quartile
df_features = df_features.withColumn(
    "Amount_quartile", ntile(4).over(window_spec)
)

print(f"Total features after engineering: {len(df_features.columns)}")
print(f"New columns: {[c for c in df_features.columns if c not in df_typed.columns]}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import current_timestamp

df_silver = df_features \
    .withColumn("silver_processed_at", current_timestamp()) \
    .drop("ingestion_timestamp", "source_file", "data_source", "source_endpoint")

df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("silver_transactions")

print(f"Silver table created with {df_silver.count()} records")
print(f"Total columns: {len(df_silver.columns)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

quality_checks = spark.sql("""
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT id) as unique_ids,
        SUM(CASE WHEN Class IS NULL THEN 1 ELSE 0 END) as null_labels,
        SUM(CASE WHEN Amount < 0 THEN 1 ELSE 0 END) as negative_amounts,
        MIN(Amount) as min_amount,
        MAX(Amount) as max_amount,
        ROUND(AVG(Amount), 2) as avg_amount,
        SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) as fraud_count,
        SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) as legit_count
    FROM silver_transactions
""")
quality_checks.show(truncate=False)

# Assert quality rules
row = quality_checks.collect()[0]
assert row["null_labels"] == 0, "ERROR: Null labels found!"
assert row["negative_amounts"] == 0, "ERROR: Negative amounts found!"
assert row["total_records"] == row["unique_ids"], "ERROR: Duplicate IDs found!"
print("All data quality checks PASSED!")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
