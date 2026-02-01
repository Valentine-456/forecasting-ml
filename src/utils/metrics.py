import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def battery_current_metrics(test_df: pd.DataFrame):
    mape_list = []
    r2_list = []

    for _f, g in test_df.groupby("flight", sort=False):
        g = g.dropna(subset=["battery_current_pred"]).copy()
        if len(g) == 0:
            continue

        y = g["battery_current"].to_numpy()
        p = g["battery_current_pred"].to_numpy()

        mask = y > 1.0
        if mask.sum() == 0:
            mape_list.append(np.nan)
            r2_list.append(np.nan)
            continue

        denom = np.clip(np.abs(y[mask]), 1e-9, None)
        mape = float(np.mean(np.abs(y[mask] - p[mask]) / denom) * 100.0)
        mape_list.append(mape)

        if np.allclose(y, y[0]):
            r2_list.append(np.nan)
        else:
            r2_list.append(float(r2_score(y, p)))

    mape_avg = np.nanmean(mape_list) if len(mape_list) else np.nan
    r2_avg = np.nanmean(r2_list) if len(r2_list) else np.nan

    return {"mape": round(float(mape_avg), 2), "r2": round(float(r2_avg), 4)}


def battery_soc_metrics(test_df: pd.DataFrame):
    df = test_df.dropna(subset=["battery_current_pred"]).copy()
    if df.empty:
        return {
            "avg_final_err_relative": np.nan,
            "max_final_err_mah": np.nan,
            "avg_trajectory_drift": np.nan,
        }

    df = df.sort_values(["flight", "time"])

    df["dt_sec"] = df.groupby("flight")["time"].diff().fillna(0.0)

    df["mAh_step_real"] = (df["dt_sec"] / 3600.0) * (df["battery_current"] * 1000.0)
    df["mAh_step_pred"] = (df["dt_sec"] / 3600.0) * (df["battery_current_pred"] * 1000.0)

    df["cum_mAh_real"] = df.groupby("flight")["mAh_step_real"].cumsum()
    df["cum_mAh_pred"] = df.groupby("flight")["mAh_step_pred"].cumsum()

    rel_errs = []
    abs_errs = []
    drifts = []

    for _f, g in df.groupby("flight", sort=False):
        real_total = float(g["cum_mAh_real"].iloc[-1])
        pred_total = float(g["cum_mAh_pred"].iloc[-1])

        abs_err = abs(real_total - pred_total)
        den = max(abs(real_total), 1.0)
        rel_err = (abs_err / den) * 100.0

        drift = float(np.mean(np.abs(g["cum_mAh_real"] - g["cum_mAh_pred"])))

        abs_errs.append(abs_err)
        rel_errs.append(rel_err)
        drifts.append(drift)

    return {
        "avg_final_err_relative": float(np.nanmean(rel_errs)) if len(rel_errs) else np.nan,
        "max_final_err_mah": float(np.nanmax(abs_errs)) if len(abs_errs) else np.nan,
        "avg_trajectory_drift": float(np.nanmean(drifts)) if len(drifts) else np.nan,
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