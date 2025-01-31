# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


class Config:
    DATA_FILE_PATH = os.getenv("DATA_FILE_PATH")

    SYNTHETIC_DATA_OUTPUT_PATH = os.getenv("SYNTHETIC_DATA_OUTPUT_PATH")
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH")
    SYNTHETIC_DATA_MULTIPLIER = int(
        os.getenv("SYNTHETIC_DATA_MULTIPLIER", 1)
    )  # Default to 1 if not set
    MODEL_TYPE = os.getenv("MODEL_TYPE", "GAN").upper()  # Default to GAN if not set
    LATENT_DIM = int(os.getenv("LATENT_DIM", 128))
    NAME_COLUMN = os.getenv("NAME_COLUMN")
    EPOCHS = int(os.getenv("EPOCHS", 100))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.0002))
    ALPHANUMERIC_UNIQUE = (
        os.getenv("ALPHANUMERIC_UNIQUE", "False").lower() == "true"
    )  # Default to False if not set

    # Validation and error handling for config parameters
    if MODEL_TYPE not in ["GAN", "VAE"]:
        raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}. Must be 'GAN' or 'VAE'.")
    if not DATA_FILE_PATH:
        raise ValueError("DATA_FILE_PATH not set in .env")
    if not SYNTHETIC_DATA_OUTPUT_PATH:
        raise ValueError("SYNTHETIC_DATA_OUTPUT_PATH not set in .env")
    if not LOG_FILE_PATH:
        raise ValueError("LOG_FILE_PATH not set in .env")
    if not NAME_COLUMN:
        raise ValueError("NAME_COLUMN not set in .env file.")


config = Config()  # Instantiate the configuration object
