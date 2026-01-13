import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from src.utils.metrics import battery_current_metrics, battery_soc_metrics
from src.utils.path_resolver import MODELS_DIR

def train_xgb(train_df, test_df, features):
    X_train = train_df[features].fillna(0)
    y_train = train_df["battery_current"]

    X_test = test_df[features].fillna(0)
    y_test = test_df["battery_current"]

    model = XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    weights = np.where(train_df["is_flying"] == 1, 1.0, 300.0)

    model.fit(
        X_train,
        y_train,
        sample_weight=weights,
    )

    preds = model.predict(X_test)
    current_metrics = battery_current_metrics(y_test, preds)
    soc_metrics = battery_soc_metrics(test_df, preds)
    return model, current_metrics, soc_metrics

def save_xgb(model, name="xgb_battery_current.pkl"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / name
    joblib.dump(model, path)
    print(f"XGBoost model saved at {path}")
    return name
