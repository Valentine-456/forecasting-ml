from src.utils.config_resolver import load_config
from src.data.load_data import load_flights
from src.data.train_test_split import train_test_split_flight
from src.features.add_features import add_features
from src.features.soc_estimation import compute_soc
from src.models.mlr import train_mlr, save_mlr
from src.utils.metrics import print_metrics

def main():
    print("Loading data...")
    df = load_flights()

    print("Engineering features...")
    df = add_features(df)
    df = compute_soc(df)

    cfg = load_config("mlr.yaml")
    features = cfg["features"]
    split = cfg["train_split"]

    print("Splitting train/test by flight...")
    train_df, test_df = train_test_split_flight(df, split_ratio=split)

    print("Training MLR model...")
    model, ent_metrics, soc_metrics = train_mlr(train_df, test_df, features)

    model, current_metrics, soc_metrics = train_mlr(train_df, test_df, features)

    print("Saving model...")
    model_name = save_mlr(model)

    print("Evaluation:", print_metrics(model_name, current_metrics, soc_metrics))

    print("Done!")

if __name__ == "__main__":
    main()
