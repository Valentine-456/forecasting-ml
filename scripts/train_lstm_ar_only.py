from src.data.filter_dummy_flights import filter_dummy_flights
from src.data.load_data import load_flights
from src.data.train_test_split import train_val_test_split_flight
from src.features.add_features import add_features
from src.features.add_is_flying import add_is_flying
from src.features.soc_estimation import compute_soc
from src.models.lstm import save_lstm, train_lstm
from src.utils.config_resolver import load_config
from src.utils.metrics import print_metrics


def main():
    print("Loading data...")
    df = load_flights()

    print("Engineering features...")
    df = filter_dummy_flights(df)
    df = add_features(df)
    df = compute_soc(df)
    df = add_is_flying(df)

    cfg = load_config("LSTM_AR_ONLY.yaml")
    features = cfg["features"]
    model_name = cfg["model"]["name"]

    print("Splitting train/val/test by flight...")
    train_df, val_df, test_df = train_val_test_split_flight(
        df,
        train_ratio=cfg["train_split"],
        val_ratio=cfg["validation_split"],
    )

    print("Training LSTM model...")
    model, scalers, current_metrics, soc_metrics = train_lstm(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        features=features,
        cfg=cfg,
    )

    print("Saving model...")
    save_lstm(model, scalers, cfg=cfg, features=features, name=model_name)

    print_metrics(model_name, current_metrics, soc_metrics)
    print("Done!")


if __name__ == "__main__":
    main()
