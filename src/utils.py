# src/utils.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Create logs directory
log_dir = Path('logs')
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame to CSV."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info(f"Saved data to {file_path}")

def ip_to_int(ip_val):
    """
    Converts a float or string-represented numeric IP address to an integer.
    Handles NaN values and malformed inputs by returning NaN.
    """
    if pd.isna(ip_val):
        return np.nan

    try:
        return int(float(ip_val))
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert IP '{ip_val}' to integer. Error: {e}")
        return np.nan

def get_country(ip_int, ip_country_df):
    """
    Maps an integer IP address to a country using a pre-sorted IP-to-country DataFrame.
    Assumes ip_country_df is sorted by 'lower_bound_ip_address'.
    Returns 'Unknown' if the IP is not found in any range.
    """
    if pd.isna(ip_int):
        return np.nan

    idx = ip_country_df['lower_bound_ip_address'].searchsorted(ip_int, side='right') - 1

    if 0 <= idx < len(ip_country_df) and ip_int <= ip_country_df.iloc[idx]['upper_bound_ip_address']:
        return ip_country_df.iloc[idx]['country']
    
    return 'Unknown'

