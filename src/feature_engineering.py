# src/feature_engineering.py

import pandas as pd
import numpy as np
from utils import ip_to_int, get_country


def merge_ip_to_country(df_fraud, df_ip_country):
    """
    Merges the e-commerce fraud data with IP address to country mapping.
    Converts IP strings to integers and then maps them to countries.
    """
    print("\nMapping IP addresses to countries for e-commerce data...")
    df_fraud['ip_address_int'] = df_fraud['ip_address'].apply(ip_to_int)

    # Sort ip_country for efficient merging (binary search like behavior in get_country)
    df_ip_country['lower_bound_ip_address'] = df_ip_country['lower_bound_ip_address'].astype(int)
    df_ip_country['upper_bound_ip_address'] = df_ip_country['upper_bound_ip_address'].astype(int)
    df_ip_country.sort_values(by='lower_bound_ip_address', inplace=True)

    df_fraud['country'] = df_fraud['ip_address_int'].apply(lambda x: get_country(x, df_ip_country))
    print("IP to Country mapping complete.")
    print(df_fraud[['ip_address', 'country', 'class']].head())
    return df_fraud

def engineer_ecommerce_features(df_fraud):
    """
    Engineers time-based and transaction frequency/velocity features for e-commerce data.
    """
    print("\nEngineering features for E-commerce data...")

    # Time-Based features
    df_fraud['time_since_signup'] = (df_fraud['purchase_time'] - df_fraud['signup_time']).dt.total_seconds()
    df_fraud['hour_of_day'] = df_fraud['purchase_time'].dt.hour
    df_fraud['day_of_week'] = df_fraud['purchase_time'].dt.dayofweek # Monday=0, Sunday=6

    # Transaction frequency and velocity (simplified approach for demonstration)
    # Sort by user_id and purchase_time to ensure correct cumulative counts/diffs
    df_fraud.sort_values(by=['user_id', 'purchase_time'], inplace=True)

    # Count previous transactions for the same user/device/ip
    df_fraud['user_transaction_count'] = df_fraud.groupby('user_id').cumcount()
    df_fraud['device_transaction_count'] = df_fraud.groupby('device_id').cumcount()
    df_fraud['ip_transaction_count'] = df_fraud.groupby('ip_address_int').cumcount()

    # Calculate time difference between consecutive transactions for user/device/ip
    df_fraud['time_diff_user'] = df_fraud.groupby('user_id')['purchase_time'].diff().dt.total_seconds().fillna(0)
    df_fraud['time_diff_device'] = df_fraud.groupby('device_id')['purchase_time'].diff().dt.total_seconds().fillna(0)
    df_fraud['time_diff_ip'] = df_fraud.groupby('ip_address_int')['purchase_time'].diff().dt.total_seconds().fillna(0)

    print("E-commerce feature engineering complete.")
    return df_fraud

