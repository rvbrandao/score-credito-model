import pandas as pd

from training.config import DATA_PATH
from training.config import FEATURE_COLUMNS
from training.config import TARGET_COLUMN


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    dataframe = pd.read_csv(DATA_PATH)

    expected_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_columns = sorted(expected_columns - set(dataframe.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            f"Dataset is missing required columns: {missing_text}"
        )

    feature_frame = dataframe[list(FEATURE_COLUMNS)]
    target_series = dataframe[TARGET_COLUMN]
    return feature_frame, target_series

