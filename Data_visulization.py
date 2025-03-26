import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load all 6 CSV files
jumping_jackson = pd.read_csv("Armaan_Jumping_Left_Pocket.csv")
walking_jackson = pd.read_csv("Armaan_Walking_Left_Pocket.csv")

jumping_general = pd.read_csv("Ivan_Jumping_Left_Pocket.csv")
walking_general = pd.read_csv("Ivan_Walking_Left_Pocket.csv")

jumping_pri = pd.read_csv("Letchu_Jumping_Left_Pocket.csv")
walking_pri = pd.read_csv("Letchu_Walking_Left_Pocket.csv")

# Clean column names
for df in [jumping_jackson, walking_jackson, jumping_general, walking_general, jumping_pri, walking_pri]:
    df.columns = df.columns.str.strip()

# Add labels for activity & person
jumping_jackson["Activity"] = "Jumping"
walking_jackson["Activity"] = "Walking"

jumping_general["Activity"] = "Jumping"
walking_general["Activity"] = "Walking"

jumping_pri["Activity"] = "Jumping"
walking_pri["Activity"] = "Walking"

jumping_jackson["Person"] = "Jackson"
walking_jackson["Person"] = "Jackson"

jumping_general["Person"] = "General"
walking_general["Person"] = "General"

jumping_pri["Person"] = "Pri"
walking_pri["Person"] = "Pri"

# Combine all datasets
df_combined = pd.concat([jumping_jackson, walking_jackson, jumping_general, walking_general, jumping_pri, walking_pri])

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
