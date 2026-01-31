import numpy as np

def train_test_split_flight(df, split_ratio=0.8, seed=42):
    flights = df["flight"].unique()
    np.random.shuffle(flights)

    split = int(len(flights) * split_ratio)
    train_ids = flights[:split]
    test_ids = flights[split:]

    train_df = df[df["flight"].isin(train_ids)]
    test_df = df[df["flight"].isin(test_ids)]

    return train_df, test_df


def train_val_test_split_flight(
    df,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
):

    flights = df["flight"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(flights)

    n = len(flights)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = flights[:n_train]
    val_ids = flights[n_train:n_train + n_val]
    test_ids = flights[n_train + n_val:]

    train_df = df[df["flight"].isin(train_ids)].copy()
    val_df   = df[df["flight"].isin(val_ids)].copy()
    test_df  = df[df["flight"].isin(test_ids)].copy()

    return train_df, val_df, test_df
