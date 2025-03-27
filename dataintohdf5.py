import h5py
import pandas as pd

members = {
    "armaan": {
        "jumping": {
            "run_1": "raw-data/armaan/jumping/jump left back/Raw Data.csv",
            "run_2": "raw-data/armaan/jumping/jump left hand/Raw Data.csv",
            "run_3": "raw-data/armaan/jumping/jump left pocket/Raw Data.csv",
            "run_4": "raw-data/armaan/jumping/jump right back/Raw Data.csv",
            "run_5": "raw-data/armaan/jumping/jump right hand/Raw Data.csv",
            "run_6": "raw-data/armaan/jumping/jump right pocket/Raw Data.csv",
        },
        "walking": {
            "run_1": "raw-data/armaan/walking/walk left back/Raw Data.csv",
            "run_2": "raw-data/armaan/walking/walk left hand/Raw Data.csv",
            "run_3": "raw-data/armaan/walking/walk left pocket/Raw Data.csv",
            "run_4": "raw-data/armaan/walking/walk right back/Raw Data.csv",
            "run_5": "raw-data/armaan/walking/walk right hand/Raw Data.csv",
            "run_6": "raw-data/armaan/walking/walk right pocket/Raw Data.csv",
        }
    },
    "ivan": {
        "jumping": {
            "run_1": "raw-data/ivan/jumping/jumping back left pocket/Raw Data.csv",
            "run_2": "raw-data/ivan/jumping/jumping back right pocket/Raw Data.csv",
            "run_3": "raw-data/ivan/jumping/jumping left front pocket - may need redoing/Raw Data.csv",
            "run_4": "raw-data/ivan/jumping/jumping left hand vertical/Raw Data.csv",
            "run_5": "raw-data/ivan/jumping/jumping right hand vertical/Raw Data.csv",
            "run_6": "raw-data/ivan/jumping/jumping right pocket/Raw Data.csv",
        },
        "walking": {
            "run_1": "raw-data/ivan/walking/walking back left pocket/Raw Data.csv",
            "run_2": "raw-data/ivan/walking/walking back right pocket/Raw Data.csv",
            "run_3": "raw-data/ivan/walking/walking left front pocket/Raw Data.csv",
            "run_4": "raw-data/ivan/walking/walking left hand vertical/Raw Data.csv",
            "run_5": "raw-data/ivan/walking/walking right hand vertical/Raw Data.csv",
            "run_6": "raw-data/ivan/walking/walking right pocket/Raw Data.csv",
        }
    },
    "letchu": {
        "jumping": {
            "run_1": "raw-data/letchu/jumping/jump left hand/Raw Data.csv",
            "run_2": "raw-data/letchu/jumping/jump left pocket/Raw Data.csv",
            "run_3": "raw-data/letchu/jumping/jump right hand/Raw Data.csv",
            "run_4": "raw-data/letchu/jumping/jump right pocket/Raw Data.csv",
        },
        "walking": {
            "run_1": "raw-data/letchu/walking/walk left hand/Raw Data.csv",
            "run_2": "raw-data/letchu/walking/walk left pocket/Raw Data.csv",
            "run_3": "raw-data/letchu/walking/walk right hand/Raw Data.csv",
            "run_4": "raw-data/letchu/walking/walk right pocket/Raw Data.csv",
        }
    }
}


with h5py.File("dataset.hdf5", "w") as h5f:
    raw_data_group = h5f.create_group("Raw data")
    
    for member_name, activities in members.items():
        member_group = raw_data_group.create_group(member_name)
        
        for activity_type, runs in activities.items(): #activities: jump and walk
            activity_group = member_group.create_group(activity_type)
            
            for run_name, csv_path in runs.items():
                run_group = activity_group.create_group(run_name)
                
                df = pd.read_csv(csv_path)

                #add filter to keep all runs to 45 seconds
                if "Time (s)" in df.columns:
                    df = df[df["Time (s)"] <= 45]

                data_array = df.to_numpy()
                
                dset = run_group.create_dataset("raw data", data=data_array)
                dset.attrs["column_names"] = df.columns.tolist() #save column names


def print_hdf5_structure(group, indent=0): #used for checking h5 file structure
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print("    " * indent + f"Group: {key}")
            print_hdf5_structure(item, indent + 1)
        else:
            print("    " * indent + f"Dataset: {key}, shape: {item.shape}, dtype: {item.dtype}")

with h5py.File("dataset.hdf5", "r") as h5f:
    print("HDF5 file structure:")
    print_hdf5_structure(h5f)