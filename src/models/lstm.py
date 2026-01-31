

from email.headerregistry import DateHeader
from typing import Dict, List
import numpy as np
import pandas as pd
import torch

from src.models.RNN.flight_window import FlightWindowDataset
from src.models.RNN.lstm_model import LSTMModel
from src.models.RNN.scale_features import Scalers, apply_scaling, fit_scalers
from src.utils.metrics import battery_current_metrics, battery_soc_metrics
from src.utils.path_resolver import MODELS_DIR
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_on_df(
    model: nn.Module,
    df: pd.DataFrame,
    features: List[str],
    scalers: Scalers,
    seq_len: int,
    device: str,
    target_col: str = "battery_current",
) -> pd.DataFrame:
    model.eval()

    df_scaled = apply_scaling(df, features, target_col, scalers)
    ds = FlightWindowDataset(df_scaled, features, target_col=target_col, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    preds_scaled = []
    y_scaled = []
    flight_ids = []
    times = []

    for X, y, f_id, t in loader:
        X = X.to(device)
        pred = model(X).cpu().numpy().reshape(-1)
        preds_scaled.append(pred)

        y_scaled.append(y.numpy().reshape(-1))
        flight_ids.extend(f_id.detach().cpu().tolist())
        times.extend(t.detach().cpu().tolist())


    preds_scaled = np.concatenate(preds_scaled)
    y_scaled = np.concatenate(y_scaled)

    preds_A = scalers.y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
    y_A = scalers.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)

    return (
        pd.DataFrame(
            {"flight": flight_ids, "time": times, "battery_current": y_A, "battery_current_pred": preds_A}
        )
        .sort_values(["flight", "time"])
        .reset_index(drop=True)
    )

def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    cfg: Dict,
    target_col: str = "battery_current",
):
    print("Reading config")
    seq_len = int(cfg["model"]["seq_len"])
    hidden_size = int(cfg["model"]["hidden_size"])
    num_layers = int(cfg["model"]["num_layers"])
    dropout = float(cfg["model"]["dropout"])

    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["lr"])
    epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"]["patience"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Scaling and grouping features")
    passthrough = ["is_flying"] if "is_flying" in features else []
    scalers = fit_scalers(train_df, features, target_col, passthrough=passthrough)

    train_scaled = apply_scaling(train_df, features, target_col, scalers)
    val_scaled = apply_scaling(val_df, features, target_col, scalers)

    train_ds = FlightWindowDataset(train_scaled, features, target_col=target_col, seq_len=seq_len)
    val_ds = FlightWindowDataset(val_scaled, features, target_col=target_col, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LSTMModel(
        input_dim=len(features),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    bad = 0

    idx_fly = features.index("is_flying") if "is_flying" in features else None

    for ep in range(1, epochs + 1):
        model.train()
        tr_sum = 0.0
        tr_n = 0

        # MODEL TRAINING

        for X, y, _f, _t in train_loader:
            X = X.to(device)
            y = y.to(device)

            if idx_fly is not None:
                is_flying_last = X[:, -1, idx_fly]
                w = torch.where(is_flying_last > 0.5, 1.0, 300.0).to(device)
            else:
                w = torch.ones(X.size(0), device=device)

            opt.zero_grad()
            pred = model(X)

            per = torch.nn.functional.smooth_l1_loss(pred, y, reduction="none").squeeze(1)  
            loss = (per * w).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_sum += float(loss.item()) * X.size(0)
            tr_n += X.size(0)

        tr_loss = tr_sum / max(tr_n, 1)

        # VALIDATION
        model.eval()
        va_sum = 0.0
        va_n = 0
        with torch.no_grad():
            for X, y, _f, _t in val_loader:
                X = X.to(device)
                y = y.to(device)

                if idx_fly is not None:
                    is_flying_last = X[:, -1, idx_fly]
                    w = torch.where(is_flying_last > 0.5, 1.0, 300.0).to(device)
                else:
                    w = torch.ones(X.size(0), device=device)

                pred = model(X)
                per = torch.nn.functional.smooth_l1_loss(pred, y, reduction="none").squeeze(1)
                loss = (per * w).mean()

                va_sum += float(loss.item()) * X.size(0)
                va_n += X.size(0)

        va_loss = va_sum / max(va_n, 1)
        print(f"epoch {ep:02d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f}")

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # TESTING
    test_eval = predict_on_df(model, test_df, features, scalers, seq_len=seq_len, device=device, target_col=target_col)

    current_metrics = battery_current_metrics(test_eval["battery_current"], test_eval["battery_current_pred"])
    soc_metrics = battery_soc_metrics(test_eval, test_eval["battery_current_pred"])

    return model, scalers, current_metrics, soc_metrics


def save_lstm(model: nn.Module, scalers: Scalers, cfg: Dict, features: List[str], name: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_state_dict": model.state_dict(),
        "model_config": cfg["model"],
        "features": features,
        "scalers": scalers.to_dict(),
    }
    path = MODELS_DIR / f"{name}.pth"
    torch.save(bundle, path)
    print(f"LSTM saved at {path}")