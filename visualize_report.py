import h5py
import pandas as pd
import matplotlib.pyplot as plt

def load_run_data(member, activity, run, file_path="dataset.hdf5"):
    """
    Load raw data for a given member, activity, and run from the HDF5 file.
    """
    with h5py.File(file_path, "r") as h5f:
        # Construct the path to the dataset
        dset = h5f[f"Raw data/{member}/{activity}/{run}/raw data"]
        # Retrieve the column names stored as an attribute
        columns = dset.attrs["column_names"]
        # Create DataFrame from the dataset
        df = pd.DataFrame(dset[...], columns=columns)
        df.columns = df.columns.str.strip()
    return df

def preprocess_df(df, window_size=5):
    """
    Apply a moving average filter to the acceleration columns using a rolling window.
    """
    df_filtered = df.copy()
    columns_to_smooth = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)"
    ]
    for col in columns_to_smooth:
        if col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].rolling(window=window_size, min_periods=1, center=True).mean()
    return df_filtered

# Choose one jumping run and one walking run (adjust member, activity, and run names as needed)
jumping_member = "ivan"
jumping_activity = "jumping"
jumping_run = "run_1"  # example run

walking_member = "ivan"
walking_activity = "walking"
walking_run = "run_1"  # example run

# Load raw data from the HDF5 file
df_jump_raw = load_run_data(jumping_member, jumping_activity, jumping_run)
df_walk_raw = load_run_data(walking_member, walking_activity, walking_run)

# Apply the moving average filter to obtain the "filtered" data
df_jump_filtered = preprocess_df(df_jump_raw, window_size=5)
df_walk_filtered = preprocess_df(df_walk_raw, window_size=5)

# Plot the raw vs. filtered time series for one acceleration axis ("Linear Acceleration x (m/s^2)")
plt.figure(figsize=(14, 8))

# Plot for Jumping run
plt.subplot(2, 1, 1)
plt.plot(df_jump_raw["Time (s)"], df_jump_raw["Linear Acceleration x (m/s^2)"], label="Raw", alpha=0.7)
plt.plot(df_jump_filtered["Time (s)"], df_jump_filtered["Linear Acceleration x (m/s^2)"], label="Filtered", linewidth=2)
plt.title("Jumping Run (Raw vs. Filtered)")
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration x (m/s^2)")
plt.legend()

# Plot for Walking run
plt.subplot(2, 1, 2)
plt.plot(df_walk_raw["Time (s)"], df_walk_raw["Linear Acceleration x (m/s^2)"], label="Raw", alpha=0.7)
plt.plot(df_walk_filtered["Time (s)"], df_walk_filtered["Linear Acceleration x (m/s^2)"], label="Filtered", linewidth=2)
plt.title("Walking Run (Raw vs. Filtered)")
plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration x (m/s^2)")
plt.legend()

plt.tight_layout()
plt.show()
