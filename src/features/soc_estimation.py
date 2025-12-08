def compute_soc(df):
    df = df.copy()

    df["dt_sec"] = df.groupby("flight")["time"].diff().fillna(0.0)

    df["mAh_used"] = (df["dt_sec"] / 3600.0) * (df["battery_current"] * 1000)
    df["cum_mAh_used"] = df.groupby("flight")["mAh_used"].cumsum()
    df["total_mAh_used"] = df.groupby("flight")["cum_mAh_used"].transform("max")

    df["soc_coulomb_mAh"] = df["total_mAh_used"] - df["cum_mAh_used"]

    return df
