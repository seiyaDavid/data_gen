import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from src.utils import setup_logging, ensure_non_negative
from src.config import config
from faker import Faker

logger = setup_logging(config.LOG_FILE_PATH)


class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.date_columns = []
        self.name_columns = []
        self.numerical_columns = []
        self.alphanumeric_columns = []
        self.original_columns = []
        self.original_data_types = {}
        self.fake = Faker()

    def fit(self, df):
        """Analyzes data types and fits preprocessing steps."""
        self.original_columns = df.columns.tolist()
        for col in df.columns:
            self.original_data_types[col] = df[col].dtype
            if col == config.NAME_COLUMN:
                self.name_columns.append(col)
                logger.info(f"Column '{col}' detected as the name column.")
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                self.date_columns.append(col)
                logger.info(f"Column '{col}' detected as date.")
            elif pd.api.types.is_numeric_dtype(df[col]):
                self.numerical_columns.append(col)
                logger.info(f"Column '{col}' detected as numerical.")
            elif self.is_alphanumeric(df[col]):
                self.alphanumeric_columns.append(col)
                logger.info(f"Column '{col}' detected as alphanumeric.")
            else:
                self.name_columns.append(col)
                logger.info(f"Column '{col}' detected as name/categorical.")

        # Fit Scalers for numerical columns
        for col in self.numerical_columns:
            self.scalers[col] = MinMaxScaler()
            self.scalers[col].fit(df[[col]])
            logger.info(f"Fitted MinMaxScaler for numerical column: {col}")

        # Fit Scalers for date columns (after converting to numerical)
        for col in self.date_columns:
            self.scalers[col] = MinMaxScaler()
            df[col] = pd.to_datetime(df[col])
            df[col + "_dayofweek_sin"] = np.sin(2 * np.pi * df[col].dt.dayofweek / 7)
            df[col + "_dayofweek_cos"] = np.cos(2 * np.pi * df[col].dt.dayofweek / 7)
            df[col + "_month_sin"] = np.sin(2 * np.pi * df[col].dt.month / 12)
            df[col + "_month_cos"] = np.cos(2 * np.pi * df[col].dt.month / 12)
            df[col + "_day_sin"] = np.sin(2 * np.pi * df[col].dt.day / 31)
            df[col + "_day_cos"] = np.cos(2 * np.pi * df[col].dt.day / 31)
            self.scalers[col].fit(df[[col]])

        # Fit LabelEncoders for name/categorical columns (excluding the specified NAME_COLUMN)
        for col in self.name_columns:
            if col != config.NAME_COLUMN:
                self.label_encoders[col] = LabelEncoder()
                df[col] = df[col].astype(str)
                self.label_encoders[col].fit(df[col])
                logger.info(f"Fitted LabelEncoder for name/categorical column: {col}")

        # Fit LabelEncoders and Scalers for alphanumeric columns
        for col in self.alphanumeric_columns:
            letter_part = df[col].str.extract("([A-Za-z]+)", expand=False)
            number_part = pd.to_numeric(
                df[col].str.extract("(\d+)", expand=False), errors="coerce"
            )

            self.label_encoders[col + "_letter"] = LabelEncoder()
            self.label_encoders[col + "_letter"].fit(letter_part)

            self.scalers[col + "_number"] = MinMaxScaler()
            self.scalers[col + "_number"].fit(number_part.values.reshape(-1, 1))

    def transform(self, df):
        """Transforms the data using fitted preprocessing steps."""
        df_transformed = df.copy()

        # Transform Numerical columns
        for col in self.numerical_columns:
            if col in self.scalers:
                df_transformed[col] = self.scalers[col].transform(df_transformed[[col]])
                logger.debug(f"Transformed numerical column: {col}")

        # Transform Date columns
        for col in self.date_columns:
            df_transformed[col] = pd.to_datetime(df_transformed[col])
            df_transformed[col + "_dayofweek_sin"] = np.sin(
                2 * np.pi * df_transformed[col].dt.dayofweek / 7
            )
            df_transformed[col + "_dayofweek_cos"] = np.cos(
                2 * np.pi * df_transformed[col].dt.dayofweek / 7
            )
            df_transformed[col + "_month_sin"] = np.sin(
                2 * np.pi * df_transformed[col].dt.month / 12
            )
            df_transformed[col + "_month_cos"] = np.cos(
                2 * np.pi * df_transformed[col].dt.month / 12
            )
            df_transformed[col + "_day_sin"] = np.sin(
                2 * np.pi * df_transformed[col].dt.day / 31
            )
            df_transformed[col + "_day_cos"] = np.cos(
                2 * np.pi * df_transformed[col].dt.day / 31
            )

            # Convert to numerical (Unix timestamp) and scale
            if col in self.scalers:
                df_transformed[col] = (
                    pd.to_datetime(df_transformed[col]).astype("int64") // 10**9
                )
                logger.debug(
                    f"Date column: {col}, Min (scaled): {df_transformed[col].min()}, Max (scaled): {df_transformed[col].max()}"
                )
                df_transformed[col] = self.scalers[col].transform(df_transformed[[col]])

        # Transform Name/categorical columns (excluding the specified NAME_COLUMN)
        for col in self.name_columns:
            if col != config.NAME_COLUMN:
                if col in self.label_encoders:
                    df_transformed[col] = df_transformed[col].astype(str)
                    df_transformed[col] = self.label_encoders[col].transform(
                        df_transformed[col]
                    )
                    logger.debug(
                        f"Transformed name/categorical column: {col} using Label Encoding."
                    )

        # Transform Alphanumeric Columns
        for col in self.alphanumeric_columns:
            letter_part = df_transformed[col].str.extract("([A-Za-z]+)", expand=False)
            number_part = pd.to_numeric(
                df_transformed[col].str.extract("(\d+)", expand=False), errors="coerce"
            )

            if col + "_letter" in self.label_encoders:
                df_transformed[col + "_letter"] = self.label_encoders[
                    col + "_letter"
                ].transform(letter_part)
            if col + "_number" in self.scalers:
                df_transformed[col + "_number"] = self.scalers[
                    col + "_number"
                ].transform(number_part.values.reshape(-1, 1))

            # Drop the original alphanumeric column
            df_transformed.drop(col, axis=1, inplace=True)

        logger.info(f"Columns in df_transformed: {df_transformed.columns}")

        df_transformed = ensure_non_negative(
            df_transformed, self.numerical_columns + self.date_columns
        )

        return df_transformed

    def inverse_transform(self, synthetic_df):
        """Inverse transforms the synthetic data back to original scale and format."""
        synthetic_df_original = synthetic_df.copy()

        # Inverse transform Numerical columns
        for col in self.numerical_columns:
            if col in self.scalers:
                logger.debug(
                    f"Column: {col}, Before inverse: min={synthetic_df_original[col].min()}, max={synthetic_df_original[col].max()}"
                )
                synthetic_df_original[col] = self.scalers[col].inverse_transform(
                    synthetic_df_original[[col]]
                )
                logger.debug(
                    f"Column: {col}, After inverse: min={synthetic_df_original[col].min()}, max={synthetic_df_original[col].max()}"
                )
                synthetic_df_original = ensure_non_negative(
                    synthetic_df_original, [col]
                )

        # Inverse transform Date columns
        for col in self.date_columns:
            if col in self.scalers:
                synthetic_df_original[col] = self.scalers[col].inverse_transform(
                    synthetic_df_original[[col]]
                )
                synthetic_df_original[col] = pd.to_datetime(
                    synthetic_df_original[col], unit="s"
                )
                synthetic_df_original[col + "_dayofweek_sin"] = np.sin(
                    2 * np.pi * synthetic_df_original[col].dt.dayofweek / 7
                )
                synthetic_df_original[col + "_dayofweek_cos"] = np.cos(
                    2 * np.pi * synthetic_df_original[col].dt.dayofweek / 7
                )
                synthetic_df_original[col + "_month_sin"] = np.sin(
                    2 * np.pi * synthetic_df_original[col].dt.month / 12
                )
                synthetic_df_original[col + "_month_cos"] = np.cos(
                    2 * np.pi * synthetic_df_original[col].dt.month / 12
                )
                synthetic_df_original[col + "_day_sin"] = np.sin(
                    2 * np.pi * synthetic_df_original[col].dt.day / 31
                )
                synthetic_df_original[col + "_day_cos"] = np.cos(
                    2 * np.pi * synthetic_df_original[col].dt.day / 31
                )
                synthetic_df_original[col] = synthetic_df_original[col].dt.strftime(
                    "%Y-%m-%d"
                )
                logger.debug(
                    f"Inverse transformed date column: {col} back to datetime."
                )

        # Inverse transform Alphanumeric columns
        for col in self.alphanumeric_columns:
            if (
                col + "_letter" in self.label_encoders
                and col + "_number" in self.scalers
            ):
                # Clip synthetic labels for letter part to the known range
                min_label = 0
                max_label = len(self.label_encoders[col + "_letter"].classes_) - 1
                synthetic_df_original[col + "_letter"] = synthetic_df_original[
                    col + "_letter"
                ].clip(lower=min_label, upper=max_label)

                # Inverse transform for letter part
                letter_part_transformed = self.label_encoders[
                    col + "_letter"
                ].inverse_transform(
                    np.round(synthetic_df_original[col + "_letter"]).astype(int)
                )

                # Debug print before combining letter and number parts
                logger.debug(
                    f"'{col}' Letter part (inverse transformed): {letter_part_transformed[:5]}"
                )

                # Inverse transform for number part
                number_part_series = synthetic_df_original[col + "_number"]
                number_part_transformed = self.scalers[
                    col + "_number"
                ].inverse_transform(number_part_series.values.reshape(-1, 1))
                number_part_transformed = (
                    np.round(number_part_transformed).astype(int).flatten()
                )

                # Debug print before combining letter and number parts
                logger.debug(
                    f"'{col}' Number part (inverse transformed): {number_part_transformed[:5]}"
                )

                # Combine letter and number parts
                if config.ALPHANUMERIC_UNIQUE:
                    # Ensure unique alphanumeric combinations
                    alphanumeric_combinations = set()
                    synthetic_values = []
                    max_attempts = 100  # Set a maximum number of attempts
                    for l, n in zip(letter_part_transformed, number_part_transformed):
                        alphanumeric = f"{l}{n}"
                        attempts = 0
                        while (
                            alphanumeric in alphanumeric_combinations
                            and attempts < max_attempts
                        ):
                            # Regenerate letter part if combination is not unique
                            l = self.label_encoders[col + "_letter"].inverse_transform(
                                [
                                    np.random.randint(
                                        0,
                                        len(
                                            self.label_encoders[
                                                col + "_letter"
                                            ].classes_
                                        ),
                                    )
                                ]
                            )[0]
                            alphanumeric = f"{l}{n}"
                            attempts += 1  # Increment the counter
                        alphanumeric_combinations.add(alphanumeric)
                        synthetic_values.append(alphanumeric)
                    synthetic_df_original[col] = synthetic_values
                else:
                    # Allow non-unique alphanumeric combinations
                    synthetic_df_original[col] = [
                        f"{l}{n}"
                        for l, n in zip(
                            letter_part_transformed, number_part_transformed
                        )
                    ]

                # Debug print after combining letter and number parts
                logger.debug(
                    f"'{col}' Combined alphanumeric (synthetic): {synthetic_df_original[col].head()}"
                )

                # Drop the temporary columns
                synthetic_df_original.drop(
                    [col + "_letter", col + "_number"], axis=1, inplace=True
                )

        # Inverse transform Name/categorical columns (excluding the specified NAME_COLUMN)
        for col in self.name_columns:
            if col != config.NAME_COLUMN:
                if col in self.label_encoders:
                    min_label = 0
                    max_label = len(self.label_encoders[col].classes_) - 1
                    synthetic_df_original[col] = synthetic_df_original[col].clip(
                        lower=min_label, upper=max_label
                    )
                    synthetic_df_original[col] = self.label_encoders[
                        col
                    ].inverse_transform(synthetic_df_original[col].round().astype(int))
                    logger.debug(
                        f"Inverse transformed name/categorical column: {col} from Label Encoding."
                    )

        # Generate fake names for the specified NAME_COLUMN
        if config.NAME_COLUMN in self.original_columns:
            num_samples = len(synthetic_df_original)
            synthetic_df_original[config.NAME_COLUMN] = [
                self.fake.name() for _ in range(num_samples)
            ]
            logger.debug(f"Generated fake names for column: {config.NAME_COLUMN}")

        # Enforce Original Data Types
        for col in synthetic_df_original.columns:
            if col in self.original_data_types:
                if self.original_data_types[col] == "int64":
                    synthetic_df_original[col] = pd.to_numeric(
                        synthetic_df_original[col], errors="coerce"
                    )
                    synthetic_df_original[col] = (
                        synthetic_df_original[col].fillna(0).round().astype("int64")
                    )
                else:
                    try:
                        synthetic_df_original[col] = synthetic_df_original[col].astype(
                            self.original_data_types[col]
                        )
                    except ValueError as e:
                        logger.warning(
                            f"Could not convert column '{col}' to type '{self.original_data_types[col]}': {e}"
                        )

        return synthetic_df_original

    def is_alphanumeric(self, column):
        """Checks if a column contains alphanumeric values."""
        return column.astype(str).str.match(r"^[A-Za-z]+\d+$").any()

    def get_feature_dimensions(self):
        """Returns the input feature dimension for models."""
        date_columns_len = len(self.date_columns) * 7
        name_columns_len = 0
        for col in self.name_columns:
            if col != config.NAME_COLUMN:
                name_columns_len += 1
        alphanumeric_columns_len = len(self.alphanumeric_columns) * 2

        return (
            len(self.numerical_columns)
            + date_columns_len
            + name_columns_len
            + alphanumeric_columns_len
        )

    def get_original_columns(self):
        """Returns the original column names."""
        return self.original_columns
