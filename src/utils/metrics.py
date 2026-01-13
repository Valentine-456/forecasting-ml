import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def battery_current_metrics(y_test, preds):
    y_t = y_test[y_test > 1.0]
    y_p = preds[y_test > 1.0]

    if len(y_t) == 0:
        return {"mape": np.nan, "r2": np.nan}

    r2 = r2_score(y_test, preds)

    mape = np.mean(np.abs(y_t - y_p) / np.abs(y_t)) * 100

    return {"mape": round(mape, 2), "r2": round(r2, 4)}


def battery_soc_metrics(y_test, preds):
    df = y_test.copy()
    df["battery_current_pred"] = preds

    df["dt_sec"] = df.groupby("flight")["time"].diff().fillna(0.0)

    df["mAh_step_real"] = (df["dt_sec"] / 3600.0) * (df["battery_current"] * 1000)
    df["mAh_step_pred"] = (df["dt_sec"] / 3600.0) * (df["battery_current_pred"] * 1000)

    df["cum_mAh_real"] = df.groupby("flight")["mAh_step_real"].cumsum()
    df["cum_mAh_pred"] = df.groupby("flight")["mAh_step_pred"].cumsum()

    flight_errors = []
    for f_id, group in df.groupby("flight"):
        real_total = group["cum_mAh_real"].iloc[-1]
        pred_total = group["cum_mAh_pred"].iloc[-1]
        
        abs_err = abs(real_total - pred_total)
        relative_err = (abs_err / max(real_total, 1.0)) * 100
        drift = np.mean(np.abs(group["cum_mAh_real"] - group["cum_mAh_pred"]))
        
        flight_errors.append({
            "flight": f_id,
            "abs_err": abs_err,
            "relative_err": relative_err,
            "drift": drift
        })

    error_df = pd.DataFrame(flight_errors)

    return {
        "avg_final_err_relative": error_df["relative_err"].mean(),
        "max_final_err_mah": error_df["abs_err"].max(),
        "avg_trajectory_drift": error_df["drift"].mean()
    }

def print_metrics(model_name, curr_metrics, soc_metrics):
    print(f" {model_name.upper()} metrics: ")
    
    print(f" CURRENT (Instantaneous Accuracy)")
    print(f" R2 Score:         {curr_metrics['r2']:>10.4f}")
    print(f" MAPE:    {curr_metrics['mape']:>10.2f} %")
    
    print(f"\n SOC (Cumulative Endurance)")
    print(f" Final Energy Error: {soc_metrics['avg_final_err_relative']:>10.2f} %")
    print(f" Max mAh Deviation: {soc_metrics['max_final_err_mah']:>9.1f} mAh")
    print(f" Avg Curve Drift:   {soc_metrics['avg_trajectory_drift']:>9.1f} mAh")