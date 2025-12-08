import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.utils.path_resolver import MODELS_DIR

def train_mlr(train_df, test_df, features):
    X_train = train_df[features].fillna(0)
    y_train = train_df["battery_current"]

    X_test = test_df[features].fillna(0)
    y_test = test_df["battery_current"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    return model, {"mse": mse}

def save_mlr(model, name="mlr_battery_current.pkl"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / name
    joblib.dump(model, path)
    print(f"Model saved at {path}")
