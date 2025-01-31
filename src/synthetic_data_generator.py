import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.config import config
from src.utils import setup_logging, ensure_non_negative
from src.data_loader import load_data
from src.data_preprocessor import DataPreprocessor
from src.models.gans import train_gan, Generator
from src.models.vae import train_vae, VAE

logger = setup_logging(config.LOG_FILE_PATH)


def generate_synthetic_data(
    model_type,
    data_file_path,
    output_path,
    latent_dim,
    epochs,
    batch_size,
    learning_rate,
):
    """Generates synthetic data using GAN or VAE."""
    logger.info("Starting synthetic data generation process...")

    # 1. Load and Preprocess Data
    logger.info("Loading and preprocessing data...")
    original_df = load_data(data_file_path)

    # --- Debug prints for original data ---
    logger.info("Original Data (First 5 Rows):")
    logger.info(original_df.head())

    preprocessor = DataPreprocessor()
    preprocessor.fit(original_df)
    processed_data = preprocessor.transform(original_df)

    # --- Debug prints for processed data ---
    logger.info("Processed Data (First 5 Rows):")
    logger.info(processed_data.head())

    # Exclude name column and specific handling for alphanumeric columns before creating tensor
    feature_columns = [
        col
        for col in processed_data.columns
        if col != config.NAME_COLUMN and not col.endswith(("_letter", "_number"))
    ]
    for col in original_df.columns:
        if preprocessor.is_alphanumeric(original_df[col]):
            feature_columns.extend([col + "_letter", col + "_number"])
    tensor_data = torch.tensor(
        processed_data[feature_columns].values, dtype=torch.float32
    )

    train_dataset = TensorDataset(tensor_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. Train Model (GAN or VAE)
    logger.info(f"Training {model_type} model...")
    if model_type == "GAN":
        generator, _ = train_gan(
            train_loader, len(feature_columns), latent_dim, epochs, learning_rate
        )
        trained_model = generator
    elif model_type == "VAE":
        logger.info(
            f"Feature dimension (feature_dim) value for VAE: {len(feature_columns)}"
        )
        vae_model = train_vae(
            train_loader, len(feature_columns), latent_dim, epochs, learning_rate
        )
        trained_model = vae_model
    else:
        raise ValueError(f"Invalid model type: {model_type}. Choose 'GAN' or 'VAE'.")

    # 3. Generate Synthetic Data
    logger.info("Generating synthetic data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    trained_model.eval()

    num_original_samples = len(original_df)
    synthetic_data_multiplier = config.SYNTHETIC_DATA_MULTIPLIER
    num_synthetic_samples = num_original_samples * synthetic_data_multiplier
    logger.info(
        f"Generating {num_synthetic_samples} synthetic samples (multiplier: {synthetic_data_multiplier}x)."
    )

    synthetic_data_processed = []
    with torch.no_grad():
        for _ in range(
            num_synthetic_samples // batch_size
            + (1 if num_synthetic_samples % batch_size else 0)
        ):
            noise = torch.randn(batch_size, latent_dim).to(device)
            if model_type == "GAN":
                synthetic_batch_processed = trained_model(noise).cpu().numpy()
            elif model_type == "VAE":
                synthetic_batch_processed, _, _ = trained_model(noise)
                synthetic_batch_processed = synthetic_batch_processed.cpu().numpy()
            synthetic_data_processed.extend(synthetic_batch_processed)

    synthetic_data_processed = synthetic_data_processed[:num_synthetic_samples]

    # Create a DataFrame with the appropriate column names
    synthetic_df_processed = pd.DataFrame(
        synthetic_data_processed, columns=feature_columns
    )

    # --- Debug prints for synthetic data before inverse transform ---
    logger.info("Synthetic Data Before Inverse Transform (First 5 Rows):")
    logger.info(synthetic_df_processed.head())

    # Log alphanumeric columns before inverse transform
    for col in preprocessor.alphanumeric_columns:
        if (
            col + "_letter" in synthetic_df_processed.columns
            and col + "_number" in synthetic_df_processed.columns
        ):
            logger.info(
                f"Synthetic Data Before Inverse - Alphanumeric Column '{col}_letter' (First 5 Rows):"
            )
            logger.info(synthetic_df_processed[col + "_letter"].head())
            logger.info(
                f"Synthetic Data Before Inverse - Alphanumeric Column '{col}_number' (First 5 Rows):"
            )
            logger.info(synthetic_df_processed[col + "_number"].head())

    # Clip numerical columns to be non-negative before inverse transform (if needed as a backup)
    for col in preprocessor.numerical_columns:
        if col in synthetic_df_processed.columns:
            synthetic_df_processed[col] = synthetic_df_processed[col].clip(lower=0)

    # 4. Inverse Transform to Original Scale and Format
    logger.info("Inverse transforming synthetic data...")

    synthetic_df_original = preprocessor.inverse_transform(synthetic_df_processed)

    # --- Debug prints for synthetic data after inverse transform ---
    logger.info("Synthetic Data After Inverse Transform (First 5 Rows):")
    logger.info(synthetic_df_original.head())

    # Log alphanumeric columns after inverse transform
    for col in preprocessor.alphanumeric_columns:
        if col in synthetic_df_original.columns:
            logger.info(
                f"Synthetic Data After Inverse - Alphanumeric Column '{col}' (First 5 Rows):"
            )
            logger.info(synthetic_df_original[col].head())

    # --- Ensure the NAME_COLUMN is in the output ---
    if config.NAME_COLUMN not in synthetic_df_original.columns:
        logger.warning(
            f"'{config.NAME_COLUMN}' not found in synthetic_df_original. Adding it back."
        )
        synthetic_df_original[config.NAME_COLUMN] = [
            preprocessor.fake.name() for _ in range(len(synthetic_df_original))
        ]

    # Ensure the final DataFrame has the correct column order
    synthetic_df_original = synthetic_df_original.reindex(columns=original_df.columns)
    # --- End NAME_COLUMN handling ---

    # 5. Save Synthetic Data
    logger.info(f"Saving synthetic data to: {output_path}")
    synthetic_df_original.to_csv(output_path, index=False)
    logger.info("Synthetic data generation complete.")


if __name__ == "__main__":
    generate_synthetic_data(
        model_type=config.MODEL_TYPE,
        data_file_path=config.DATA_FILE_PATH,
        output_path=config.SYNTHETIC_DATA_OUTPUT_PATH,
        latent_dim=config.LATENT_DIM,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
    )
