import numpy as np

def add_features(df):
    df = df.copy()

    # Lat/lon conversions
    df["lat_rad"] = np.deg2rad(df["position_y"])
    df["lat_m"] = (df["position_y"] -
                   df.groupby("flight")["position_y"].transform("first")) * 111111
    df["lon_m"] = ((df["position_x"] -
                    df.groupby("flight")["position_x"].transform("first"))
                    * 111111 * np.cos(df["lat_rad"]))

    # Relative altitude
    df["alt"] = df["position_z"] - df.groupby("flight")["position_z"].transform("first")

    # Horizontal speed
    df["speed_h_ms"] = np.sqrt(df["velocity_x"]**2 + df["velocity_y"]**2)

    # Heading and wind effect
    heading = np.degrees(np.arctan2(df["velocity_y"], df["velocity_x"])) % 360
    wind_angle_diff = np.deg2rad(df["wind_angle"] - heading)
    df["wind_effect"] = df["wind_speed"] * np.cos(wind_angle_diff)

    # Position differences
    df["d_position_x"] = df.groupby("flight")["lat_m"].diff().fillna(0)
    df["d_position_y"] = df.groupby("flight")["lon_m"].diff().fillna(0)
    df["d_position_z"] = df.groupby("flight")["position_z"].diff().fillna(0)
    df["d_position_h"] = np.sqrt(df["d_position_x"]**2 + df["d_position_y"]**2)

    return df
