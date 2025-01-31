# src/utils.py
import logging


def setup_logging(log_file_path):  # Modified: Pass log_file_path as argument
    """Sets up logging to a file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(),  # Output to console as well
        ],
    )
    return logging.getLogger(__name__)  # Return the logger instance


def ensure_non_negative(df, numerical_cols):
    """Ensures numerical columns in a DataFrame are non-negative (clips negative values)."""
    for col in numerical_cols:
        if df[col].dtype in ["int64", "float64"]:  # Check if it's a numerical column
            df[col] = df[col].clip(lower=0)  # Clip values to be >= 0
    return df
