def add_is_flying(df):
    df = df.copy()
    df['is_flying'] = ((df['speed_h_ms'] > 0.1) | (df['alt'] > 1)).astype(int)
    return df