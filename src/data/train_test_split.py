import numpy as np

def train_test_split_flight(df, split_ratio=0.8):
    flights = df["flight"].unique()
    np.random.shuffle(flights)

    split = int(len(flights) * split_ratio)
    train_ids = flights[:split]
    test_ids = flights[split:]

    train_df = df[df["flight"].isin(train_ids)]
    test_df = df[df["flight"].isin(test_ids)]

    return train_df, test_df
