import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from src.utils.config_resolver import load_config
from src.data.load_data import load_flights
from src.data.train_test_split import train_test_split_flight
from src.features.add_features import add_features
from src.features.soc_estimation import compute_soc
from src.utils.path_resolver import DATA_PROCESSED


def main():
    print("Loading raw data...")
    df = load_flights()

    print("Applying feature engineering...")
    df = add_features(df)
    df = compute_soc(df)

    # Load config (train/test split ratio)
    cfg = load_config("mlr.yaml")
    split_ratio = cfg.get("train_split", 0.8)

    print(f"Splitting dataset into train/test (ratio={split_ratio})...")
    train_df, test_df = train_test_split_flight(df, split_ratio=split_ratio)

    # Output paths
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_path = DATA_PROCESSED / "train_dataset.csv"
    test_path = DATA_PROCESSED / "test_dataset.csv"

    print("Saving datasets...")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train dataset saved to: {train_path}")
    print(f"Test dataset saved to:  {test_path}")
    print("Done!")


if __name__ == "__main__":
    main()
