# src/data_loader.py
import pandas as pd
from src.utils import setup_logging
from src.config import config  # Import config here

logger = setup_logging(config.LOG_FILE_PATH)  # Pass config.LOG_FILE_PATH


def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: File not found at path: {file_path}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        raise
