from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Scalers:
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    scaled_features: List[str]   
    not_scaled_features: List[str]     # is_flying

    def to_dict(self) -> Dict:
        return {
            "x_mean": self.x_scaler.mean_.tolist(),
            "x_scale": self.x_scaler.scale_.tolist(),
            "y_mean": self.y_scaler.mean_.tolist(),
            "y_scale": self.y_scaler.scale_.tolist(),
            "scaled_features": self.scaled_features,
            "not_scaled_features": self.not_scaled_features,
        }
    
    @staticmethod
    def from_dict(d: Dict) -> "Scalers":
        xs = StandardScaler()
        ys = StandardScaler()

        #  x scaler
        x_mean = np.array(d.get("x_mean", []), dtype=np.float64)
        x_scale = np.array(d.get("x_scale", []), dtype=np.float64)

        xs.mean_ = x_mean
        xs.scale_ = x_scale
        xs.var_ = x_scale ** 2
        xs.n_features_in_ = len(x_mean)

        # Y scaler
        y_mean = np.array(d["y_mean"], dtype=np.float64)
        y_scale = np.array(d["y_scale"], dtype=np.float64)

        ys.mean_ = y_mean
        ys.scale_ = y_scale
        ys.var_ = y_scale ** 2
        ys.n_features_in_ = 1

        return Scalers(
            x_scaler=xs,
            y_scaler=ys,
            scaled_features=list(d.get("scaled_features", [])),
            not_scaled_features=list(d.get("not_scaled_features", [])),
        )

def fit_scalers(
    train_df: pd.DataFrame,
    features: List[str],
    target_col: str,
    passthrough: Optional[List[str]] = None,
) -> Scalers:
    passthrough = passthrough or []
    scaled_features = [f for f in features if f not in passthrough]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    if scaled_features:
        x_scaler.fit(train_df[scaled_features].fillna(0.0).to_numpy())
    else:
        x_scaler.mean_ = np.array([])
        x_scaler.scale_ = np.array([])

    y_scaler.fit(train_df[[target_col]].to_numpy())

    return Scalers(
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        scaled_features=scaled_features,
        not_scaled_features=passthrough,
    )


def apply_scaling(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    scalers: Scalers,
) -> pd.DataFrame:
    df2 = df.copy()

    if scalers.scaled_features:
        df2[scalers.scaled_features] = scalers.x_scaler.transform(
            df2[scalers.scaled_features].fillna(0.0).to_numpy()
        )

    for f in scalers.not_scaled_features:
        if f not in df2.columns:
            df2[f] = 0.0

    df2[[target_col]] = scalers.y_scaler.transform(df2[[target_col]].to_numpy())

    return df2

