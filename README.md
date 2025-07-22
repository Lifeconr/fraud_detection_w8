# Fraud Detection for E-commerce and Bank Transactions

## Overview
This project implements Task 1 of the fraud detection challenge for Adey Innovations Inc. It preprocesses e-commerce (`Fraud_Data.csv`) and bank transaction (`creditcard.csv`) datasets, performs exploratory data analysis (EDA), merges with geolocation data (`IpAddress_to_Country.csv`), and engineers features for fraud detection.


## Setup
1. **Create Directory Structure**:
   ```bash
   mkdir -p fraud_detection/data/raw fraud_detection/data/processed fraud_detection/src fraud_detection/logs


## Place Datasets:
Copy Fraud_Data.csv, IpAddress_to_Country.csv, and creditcard.csv to fraud_detection/data/raw/.


Install Dependencies: pip install -r requirements.txt

## Directory Structure

  data/raw/: Input datasets.
  data/processed/: Preprocessed data and EDA plots.
  src/: Scripts for preprocessing, EDA, and feature engineering.
  requirements.txt: Dependencies.

## Outputs

Preprocessed CSVs: data/processed/fraud_data_processed.csv, data/processed/creditcard_processed.csv.
Feature-engineered CSVs: data/processed/fraud_data_features.csv, data/processed/creditcard_features.csv.
EDA plots: Histograms, boxplots, and class distribution plots  

Run Scripts:
```bash
python src/data_preprocessing.py
python src/eda.py
python src/feature_engineering.py 




