# main.py
from src.synthetic_data_generator import generate_synthetic_data
from src.config import config
from src.utils import setup_logging

logger = setup_logging(config.LOG_FILE_PATH)

if __name__ == "__main__":
    try:
        logger.info("Starting synthetic data generation application.")
        generate_synthetic_data(
            model_type=config.MODEL_TYPE,
            data_file_path=config.DATA_FILE_PATH,
            output_path=config.SYNTHETIC_DATA_OUTPUT_PATH,
            latent_dim=config.LATENT_DIM,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
        )
        logger.info("Synthetic data generation process finished successfully.")
    except Exception as e:
        logger.error(
            f"An error occurred during synthetic data generation: {e}", exc_info=True
        )
