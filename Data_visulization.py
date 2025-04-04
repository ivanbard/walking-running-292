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
                
                df["Activity"] = activity.capitalize()  #capitalize activity labels
                df["Person"] = member 
                df["Run"] = run  # add run identifier
                
                all_dfs.append(df)

df_combined = pd.concat(all_dfs, ignore_index=True)

#overall time plot with evevryhting
plt.figure(figsize=(12, 6))
for person in df_combined["Person"].unique():
    for activity in ["Jumping", "Walking"]:
        subset = df_combined[(df_combined["Person"] == person) & (df_combined["Activity"] == activity)]
        if "Time (s)" in subset.columns:
            plt.plot(subset["Time (s)"], subset["Linear Acceleration x (m/s^2)"], label=f"{person} - {activity} X", alpha=0.7)
            plt.plot(subset["Time (s)"], subset["Linear Acceleration y (m/s^2)"], label=f"{person} - {activity} Y", alpha=0.7)
            plt.plot(subset["Time (s)"], subset["Linear Acceleration z (m/s^2)"], label=f"{person} - {activity} Z", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Time-Series Comparison of Jumping vs. Walking (All Runs)")
plt.legend()
plt.grid()
plt.show()

#bar chart to compare average accels across all runs
df_melted = df_combined.melt(id_vars=["Activity", "Person"], 
                             value_vars=["Linear Acceleration x (m/s^2)", 
                                         "Linear Acceleration y (m/s^2)", 
                                         "Linear Acceleration z (m/s^2)"],
                             var_name="Axis", value_name="Acceleration")
mean_acceleration = df_melted.groupby(["Person", "Activity", "Axis"]).mean().reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x="Axis", y="Acceleration", hue="Activity", data=mean_acceleration, ci=None)
plt.xlabel("Acceleration Axis")
plt.ylabel("Mean Acceleration (m/s²)")
plt.title("Mean Acceleration Comparison (All Runs)")
plt.legend(title="Activity")
plt.show()

#dataframe for printing one jump
single_jump = df_combined[(df_combined["Person"]=="armaan") & 
                          (df_combined["Activity"]=="Jumping") & 
                          (df_combined["Run"]=="run_1")]

#dataframe for printing one walk
single_walk = df_combined[(df_combined["Person"]=="armaan") & 
                          (df_combined["Activity"]=="Walking") & 
                          (df_combined["Run"]=="run_1")]

#combined plot
plt.figure(figsize=(10, 5))
plt.plot(single_jump["Time (s)"], single_jump["Linear Acceleration x (m/s^2)"], label="X", linewidth=2)
plt.plot(single_jump["Time (s)"], single_jump["Linear Acceleration y (m/s^2)"], label="Y", linewidth=2)
plt.plot(single_jump["Time (s)"], single_jump["Linear Acceleration z (m/s^2)"], label="Z", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Jumping Run (Combined XYZ) - armaan, run_1")
plt.legend()
plt.grid()
plt.show()

#separate jump data
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axs[0].plot(single_jump["Time (s)"], single_jump["Linear Acceleration x (m/s^2)"], color="r", linewidth=2)
axs[0].set_ylabel("X Acceleration")
axs[0].set_title("Jumping Run - X Axis (armaan, run_1)")
axs[0].grid(True)

axs[1].plot(single_jump["Time (s)"], single_jump["Linear Acceleration y (m/s^2)"], color="g", linewidth=2)
axs[1].set_ylabel("Y Acceleration")
axs[1].set_title("Jumping Run - Y Axis (armaan, run_1)")
axs[1].grid(True)

axs[2].plot(single_jump["Time (s)"], single_jump["Linear Acceleration z (m/s^2)"], color="b", linewidth=2)
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Z Acceleration")
axs[2].set_title("Jumping Run - Z Axis (armaan, run_1)")
axs[2].grid(True)

plt.tight_layout()
plt.show()

#combined walking plot
plt.figure(figsize=(10, 5))
plt.plot(single_walk["Time (s)"], single_walk["Linear Acceleration x (m/s^2)"], label="X", linewidth=2)
plt.plot(single_walk["Time (s)"], single_walk["Linear Acceleration y (m/s^2)"], label="Y", linewidth=2)
plt.plot(single_walk["Time (s)"], single_walk["Linear Acceleration z (m/s^2)"], label="Z", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Walking Run (Combined XYZ) - armaan, run_1")
plt.legend()
plt.grid()
plt.show()

#separate jumping plots
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axs[0].plot(single_walk["Time (s)"], single_walk["Linear Acceleration x (m/s^2)"], color="r", linewidth=2)
axs[0].set_ylabel("X Acceleration")
axs[0].set_title("Walking Run - X Axis (armaan, run_1)")
axs[0].grid(True)

axs[1].plot(single_walk["Time (s)"], single_walk["Linear Acceleration y (m/s^2)"], color="g", linewidth=2)
axs[1].set_ylabel("Y Acceleration")
axs[1].set_title("Walking Run - Y Axis (armaan, run_1)")
axs[1].grid(True)

axs[2].plot(single_walk["Time (s)"], single_walk["Linear Acceleration z (m/s^2)"], color="b", linewidth=2)
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Z Acceleration")
axs[2].set_title("Walking Run - Z Axis (armaan, run_1)")
axs[2].grid(True)

plt.tight_layout()
plt.show()
