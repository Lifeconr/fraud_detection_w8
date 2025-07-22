# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

def load_data(fraud_path, ip_country_path, creditcard_path):
    """
    Loads the three required datasets from specified paths.
    """
    try:
        df_fraud = pd.read_csv(fraud_path)
        df_ip_country = pd.read_csv(ip_country_path)
        df_creditcard = pd.read_csv(creditcard_path)
        print("Datasets loaded successfully.")
        return df_fraud, df_ip_country, df_creditcard
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Please ensure files are in the correct directory.")
        raise

def clean_ecommerce_data(df_fraud):
    """
    Cleans the e-commerce fraud data:
    - Removes duplicates.
    - Corrects data types for 'signup_time' and 'purchase_time'.
    """
    print(f"E-commerce data before dropping duplicates: {df_fraud.shape}")
    df_fraud.drop_duplicates(inplace=True)
    print(f"E-commerce data after dropping duplicates: {df_fraud.shape}")

    df_fraud['signup_time'] = pd.to_datetime(df_fraud['signup_time'])
    df_fraud['purchase_time'] = pd.to_datetime(df_fraud['purchase_time'])
    print("Data types for e-commerce data updated.")
    return df_fraud

def clean_creditcard_data(df_creditcard):
    """
    Cleans the credit card fraud data:
    - Removes duplicates.
    """
    print(f"Credit card data before dropping duplicates: {df_creditcard.shape}")
    df_creditcard.drop_duplicates(inplace=True)
    print(f"Credit card data after dropping duplicates: {df_creditcard.shape}")
    return df_creditcard

def setup_ecommerce_preprocessor(X_fraud):
    """
    Sets up the ColumnTransformer for e-commerce data preprocessing.
    Identifies numerical and categorical features and defines scaling/encoding.
    """
    categorical_features = ['source', 'browser', 'sex', 'country']
    numerical_features = X_fraud.select_dtypes(include=np.number).columns.tolist()

    # Ensure 'country' is in categorical features if it exists in X_fraud
    if 'country' not in X_fraud.columns:
        print("Warning: 'country' column not found in X_fraud. It might not have been merged yet.")
        # Remove 'country' from categorical_features if it's not present
        categorical_features = [f for f in categorical_features if f in X_fraud.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any, though none expected after dropping target/IDs
    )
    print("E-commerce data preprocessing pipeline setup complete.")
    return preprocessor, categorical_features, numerical_features

def setup_creditcard_preprocessor(X_credit):
    """
    Sets up the ColumnTransformer for credit card data preprocessing.
    All features are numerical and will be scaled.
    """
    numerical_features = X_credit.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features)
        ]
    )
    print("Credit card data preprocessing pipeline setup complete.")
    return preprocessor, numerical_features
