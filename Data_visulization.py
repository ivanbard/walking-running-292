import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

all_dfs = []
hdf5_file = "dataset.hdf5"

with h5py.File(hdf5_file, "r") as h5f:
    raw_group = h5f["Raw data"]
    
    #go through all the members
    for member in raw_group.keys():
        member_group = raw_group[member]
        
        #go through all the activities
        for activity in member_group.keys():
            activity_group = member_group[activity]
            
            #go through all the run folders
            for run in activity_group.keys():
                run_group = activity_group[run]
                
                dset = run_group["raw data"] #load dataset
                columns = dset.attrs["column_names"]
                
                df = pd.DataFrame(dset[...], columns=columns)
                df.columns = df.columns.str.strip()
                
                df["Activity"] = activity.capitalize() #capitalize activity labels
                df["Person"] = member 
                
                all_dfs.append(df)

df_combined = pd.concat(all_dfs, ignore_index=True)

# Select relevant columns
acceleration_columns = ["Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)"]

# --- 📌 1️⃣ TIME-SERIES PLOT (Acceleration over Time) ---
plt.figure(figsize=(12, 6))

# Plot each person's data
for person in df_combined["Person"].unique():
    for activity in ["Jumping", "Walking"]:
        subset = df_combined[(df_combined["Person"] == person) & (df_combined["Activity"] == activity)]
        if "Time (s)" in subset.columns:  # Ensure time exists
            plt.plot(subset["Time (s)"], subset["Linear Acceleration x (m/s^2)"], label=f"{person} - {activity} X")
            plt.plot(subset["Time (s)"], subset["Linear Acceleration y (m/s^2)"], label=f"{person} - {activity} Y")
            plt.plot(subset["Time (s)"], subset["Linear Acceleration z (m/s^2)"], label=f"{person} - {activity} Z")

plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Time-Series Comparison of Jumping vs. Walking")
plt.legend()
plt.grid()
plt.show()

# --- 📌 2️⃣ BAR CHART (Mean Acceleration Comparison) ---
# Melt DataFrame for seaborn
df_melted = df_combined.melt(id_vars=["Activity", "Person"], value_vars=acceleration_columns, var_name="Axis", value_name="Acceleration")

# Compute mean acceleration per person, activity, and axis
mean_acceleration = df_melted.groupby(["Person", "Activity", "Axis"]).mean().reset_index()

# Create bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x="Axis", y="Acceleration", hue="Activity", data=mean_acceleration, ci=None)

# Labeling
plt.xlabel("Acceleration Axis")
plt.ylabel("Mean Acceleration (m/s²)")
plt.title("Comparison of Jumping vs. Walking for All Participants")
plt.legend(title="Activity")

# Show plot
plt.show()
