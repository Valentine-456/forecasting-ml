from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FlightWindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "battery_current",
        seq_len: int = 100,
    ):
        self.samples: List[Tuple[np.ndarray, float, int, float]] = []

        for flight_id, g in df.groupby("flight"):
            g = g.sort_values("time").reset_index(drop=True)
            if len(g) <= seq_len:
                continue

            X = g[feature_cols].to_numpy(dtype=np.float32)
            y = g[target_col].to_numpy(dtype=np.float32)
            t = g["time"].to_numpy(dtype=np.float32)

            for i in range(seq_len, len(g)):
                self.samples.append((X[i - seq_len:i], float(y[i]), int(flight_id), float(t[i])))

        if not self.samples:
            raise ValueError("No LSTM samples created")
        

    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int):
        Xseq, y, flight_id, time_val = self.samples[idx]
        return (
            torch.from_numpy(Xseq),                  # [L, F]
            torch.tensor([y], dtype=torch.float32),  # [1]
            flight_id,
            time_val,
        )
    

    