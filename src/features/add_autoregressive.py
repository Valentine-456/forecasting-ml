def add_battery_current_ar(df):
    df = df.copy()

    df["battery_current_lag1"] = (
        df.groupby("flight")["battery_current"]
          .shift(1)
          .fillna(0.0)
    )

    return df
