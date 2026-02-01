import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.utils.metrics import battery_current_metrics, battery_soc_metrics
from src.utils.path_resolver import MODELS_DIR

def train_mlr(train_df, test_df, features):
    X_train = train_df[features].fillna(0)
    y_train = train_df["battery_current"]

    X_test = test_df[features].fillna(0)
    y_test = test_df["battery_current"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    test_df["battery_current_pred"] = preds
    current_metrics = battery_current_metrics(test_df)
    soc_metrics = battery_soc_metrics(test_df)
    
    return model, current_metrics, soc_metrics

def save_mlr(model, name="mlr_battery_current"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"Model saved at {path}")
    return name