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

%pip install kaggle

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import os
import json

# Set Kaggle API credentials (from your kaggle.json)
os.environ['KAGGLE_USERNAME'] = 'rajkumarmanala'
os.environ['KAGGLE_KEY'] = 'KGAT_869672217f1ca1d05a77a87807e0f3b2'

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Download dataset via API
dataset = "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
download_path = "/lakehouse/default/Files/raw_data"

# Create directory if it doesn't exist
os.makedirs(download_path, exist_ok=True)

# Download and unzip
api.dataset_download_files(dataset, path=download_path, unzip=True)

# Verify download
for f in os.listdir(download_path):
    size_mb = os.path.getsize(os.path.join(download_path, f)) / (1024 * 1024)
    print(f"Downloaded: {f} ({size_mb:.1f} MB)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import current_timestamp, lit

# Read the API-downloaded CSV
df_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("Files/raw_data/creditcard_2023.csv")

print(f"Total records: {df_raw.count()}")
print(f"Total columns: {len(df_raw.columns)}")
df_raw.printSchema()

# Add ingestion metadata showing it came from API
df_bronze = df_raw \
    .withColumn("ingestion_timestamp", current_timestamp()) \
    .withColumn("data_source", lit("kaggle_api")) \
    .withColumn("source_endpoint", lit("https://www.kaggle.com/api/v1/datasets/download"))

df_bronze.show(5, truncate=False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df_bronze.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("bronze_transactions")

print("Bronze table created via API ingestion!")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

df = spark.sql("SELECT * FROM fraud_lakehouse.dbo.bronze_transactions LIMIT 1000")
display(df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.sql("""
    SELECT 
        COUNT(*) as total_records,
        SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) as fraud_count,
        SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) as legit_count,
        ROUND(SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) as fraud_pct
    FROM bronze_transactions
""").show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
