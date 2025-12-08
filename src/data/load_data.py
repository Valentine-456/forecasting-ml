import pandas as pd
import numpy as np
from src.utils.path_resolver import DATA_RAW

def comma_to_numeric(x):
    s = x.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def load_flights(filename="flights.csv"):
    path = DATA_RAW / filename
    df = pd.read_csv(path, low_memory=False)

    if "altitude" in df.columns:
        if df["altitude"].dtype == "object":
            df["altitude"] = comma_to_numeric(df["altitude"])

    df = df.sort_values(["flight", "time"]).reset_index(drop=True)
    df["flight_frame"] = df.groupby("flight").cumcount() + 1

    return df
