# Credit Card Fraud Detection Platform

An end-to-end data engineering and data science platform built on **Microsoft Fabric**, designed to detect fraudulent credit card transactions using a medallion lakehouse architecture and machine learning.

## Architecture

```
                    ┌──────────────┐
                    │  Kaggle API  │
                    └──────┬───────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Bronze Layer         │
              │   (Raw Transactions)   │
              │   568,630 records      │
              └────────────┬───────────┘
                           │  PySpark
                           ▼
              ┌────────────────────────┐
              │   Silver Layer         │
              │   (Cleaned + Features) │
              │   37 engineered cols   │
              └──────┬─────────┬───────┘
                     │         │
                     ▼         ▼
        ┌────────────────┐  ┌──────────────────┐
        │   Gold Layer   │  │  ML Experiments   │
        │  (Aggregated)  │  │  (MLflow)         │
        │   4 tables     │  │  3 models trained │
        └───────┬────────┘  └────────┬──────────┘
                │                    │
                │         ┌──────────▼──────────┐
                │         │  Model Registry     │
                │         │  (Best model saved) │
                │         └──────────┬──────────┘
                │                    │
                ▼                    ▼
        ┌────────────────────────────────────┐
        │   Gold: Scored Predictions         │
        │   (fraud_probability + risk_cat)   │
        └───────────────┬────────────────────┘
                        │
                        ▼
              ┌────────────────────────┐
              │   Power BI Dashboard   │
              │   (Fraud Analytics)    │
              └────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Platform | Microsoft Fabric (Lakehouse, Data Factory, Power BI) |
| Compute | Apache Spark (PySpark) |
| Storage | Delta Lake on OneLake |
| ML Tracking | MLflow (Experiments + Model Registry) |
| Models | Logistic Regression, Random Forest, XGBoost |
| Orchestration | Fabric Data Factory Pipelines |
| Visualization | Power BI |
| Version Control | GitHub (Fabric Git Integration) |
| Language | Python, PySpark, SQL |

## Dataset

**Credit Card Fraud Detection Dataset 2023** from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)

- **568,630** credit card transactions by European cardholders
- **31 columns**: `id`, `V1`–`V28` (PCA-transformed), `Amount`, `Class`
- **Class**: 0 = Legitimate, 1 = Fraudulent
- **Balanced dataset**: 50/50 split (pre-balanced by dataset creator)
- Data ingested programmatically via **Kaggle REST API** — no manual uploads

## Project Structure

```
Credit-Fraud-Detection/
│
├── 01_bronze_ingestion/          # Raw data ingestion from Kaggle API → Delta table
├── 02_silver_transformation/     # Data cleaning, type casting, feature engineering
├── 03_gold_aggregation/          # Business-ready aggregated tables for Power BI
├── 04_eda_analysis/              # Exploratory data analysis with visualizations
├── 05_model_training/            # Train 3 models with MLflow experiment tracking
├── 06_model_scoring/             # Load best model, score all transactions
├── fraud_lakehouse/              # Fabric Lakehouse metadata
├── fraud-detection-pipeline/     # Data Factory pipeline (orchestration)
└── README.md
```

## Medallion Architecture

### Bronze Layer
- Raw CSV ingested from Kaggle REST API using `KaggleApi` Python SDK
- Stored as Delta table with ingestion metadata (`ingestion_timestamp`, `data_source`, `source_endpoint`)
- No transformations applied — preserves original data for auditability

### Silver Layer
- Data cleaning: deduplication, null checks, type casting
- **9 engineered features**:
  - `Amount_log` — log-transformed transaction amount to reduce skewness
  - `Amount_category` — binned into Low / Medium / High / Very High
  - `V1_abs` through `V5_abs` — absolute values of top PCA components
  - `V1_V2_interaction` — interaction between top two features
  - `V1_Amount_interaction` — interaction between V1 and log amount
  - `Amount_percentile` — percentile rank of each transaction
  - `Amount_quartile` — quartile assignment
- Data quality assertions: zero nulls, zero duplicates, zero negative amounts

### Gold Layer
Four business-ready Delta tables:

| Table | Description |
|-------|-------------|
| `gold_fraud_by_amount` | Fraud rate and counts by amount category |
| `gold_fraud_by_quartile` | Fraud rate by spending quartile |
| `gold_kpis` | Single-row KPI table for Power BI cards |
| `gold_feature_summary` | Average feature values by fraud/non-fraud class |

## Machine Learning

### Experiment Tracking with MLflow

Three classification models trained and compared using MLflow:

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9712 | 0.9586 | 0.9852 | 0.9717 | 0.9952 |
| **Random Forest** | **0.9995** | **0.9992** | **0.9997** | **0.9995** | **1.0000** |
| XGBoost | 0.9994 | 0.9991 | 0.9996 | 0.9994 | 1.0000 |

**Best Model**: Random Forest (200 estimators, max depth 15)

### What was logged to MLflow for each run:
- **Parameters**: model type, hyperparameters, number of features
- **Metrics**: accuracy, precision, recall, F1, AUC-ROC, average precision
- **Artifacts**: confusion matrix CSV, feature importance CSV
- **Model**: serialized model object

### Top Predictive Features (Random Forest)
1. V14 — 19.48%
2. V4 — 13.44%
3. V10 — 13.38%
4. V17 — 9.34%
5. V12 — 8.78%

### Model Scoring
- Best model loaded from **MLflow Model Registry**
- All 568,630 transactions scored with:
  - `fraud_prediction` (0 or 1)
  - `fraud_probability` (0.0 to 1.0)
  - `risk_category` (Low / Medium / High)
- Predictions written to `gold_fraud_scores` Delta table

## Pipeline Orchestration

Automated **Data Factory pipeline** chains all notebooks:

```
Bronze Ingestion → Silver Transformation → Gold Aggregation → Model Scoring
```

- Scheduled for daily execution
- Error handling on each activity
- End-to-end runtime: ~10 minutes

## Power BI Dashboard

Four-page interactive dashboard connected via Fabric semantic model:

1. **Executive Summary** — KPI cards (total transactions, fraud count, fraud rate, fraud amount)
2. **Fraud Analysis** — Fraud rate by amount category and quartile
3. **Model Performance** — AUC-ROC, precision, recall, F1 score metrics
4. **Transaction Scores** — Detailed scored transactions with risk category filtering

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| API ingestion over CSV upload | Demonstrates production-style data ingestion |
| Medallion architecture | Industry-standard pattern for data quality progression |
| No SMOTE applied | Dataset is pre-balanced at 50/50 — verified class distribution before applying any resampling |
| MLflow over manual tracking | Native Fabric integration, reproducibility, model versioning |
| Multiple models compared | Shows systematic model selection, not just picking one algorithm |
| Gold layer for Power BI | Pre-aggregated tables reduce dashboard query time |
| Delta Lake for all layers | ACID transactions, time travel, schema enforcement |

## How to Run

### Prerequisites
- Microsoft Fabric workspace with trial or paid capacity
- Kaggle account and API token (`kaggle.json`)
- GitHub account for Git integration

### Steps
1. Create a Fabric workspace and Lakehouse
2. Create an MLflow Experiment: `fraud-detection-experiment`
3. Import notebooks into the workspace
4. Set your Kaggle credentials in `01_bronze_ingestion`
5. Run notebooks sequentially: 01 → 02 → 03 → 04 → 05 → 06
6. Create a Data Factory pipeline to chain notebooks
7. Build Power BI report from the Gold layer semantic model

## Skills Demonstrated

- **Data Engineering**: API ingestion, medallion architecture, Delta Lake, PySpark transformations, data quality checks, pipeline orchestration
- **Data Science**: Feature engineering, model training, hyperparameter tuning, class imbalance analysis, model comparison, MLflow experiment tracking, model registry
- **Business Intelligence**: Semantic modeling, Power BI dashboard design, KPI reporting
- **DevOps**: Git integration with Fabric, automated pipelines, documentation

## Author

**Raj Kumar Manala**

Built as a portfolio project targeting Azure Cloud Data Engineering and Microsoft Fabric Engineering roles.
