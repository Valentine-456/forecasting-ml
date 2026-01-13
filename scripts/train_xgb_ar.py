from src.features.add_autoregressive import add_battery_current_ar
from src.features.add_is_flying import add_is_flying
from src.utils.config_resolver import load_config
from src.data.load_data import load_flights
from src.data.train_test_split import train_test_split_flight
from src.features.add_features import add_features
from src.features.soc_estimation import compute_soc
from src.models.xgboost import train_xgb, save_xgb
from src.utils.metrics import print_metrics


def main():
    print("Loading data...")
    df = load_flights()

    print("Engineering features...")
    df = add_features(df)
    df = compute_soc(df)
    df = add_is_flying(df)
    df = add_battery_current_ar(df)

    cfg = load_config("XGB_AR_lag1.yaml")
    features = cfg["features"]
    split = cfg["train_split"]
    model_name = cfg["model"]["name"]

    print("Splitting train/test by flight...")
    train_df, test_df = train_test_split_flight(df, split_ratio=split)

    print("Training XGBoost model...")
    model, current_metrics, soc_metrics = train_xgb(train_df, test_df, features)

    print("Saving model...")
    save_xgb(model, name=model_name)

    print("Evaluation:", print_metrics(model_name, current_metrics, soc_metrics))

    print("Done!")


if __name__ == "__main__":
    main()
