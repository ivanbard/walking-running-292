import h5py
import pandas as py

members = {
    "armaan": {
        "jumping": {
            "run_1": "member1/jumping/run_1/raw data.csv",
            "run_2": "member1/jumping/run_2/raw data.csv",
        },
        "walking": {
            "run_1": "member1/walking/run_1/raw data.csv",
            "run_2": "member1/walking/run_2/raw data.csv",
        }
    },
    "ivan": {
        "jumping": {
            "run_1": "ivan/jumping/jumping back left pocket/Raw Data.csv",
            "run_2": "ivan/jumping/jumping back right pocket/Raw Data.csv",
            "run_3": "ivan/jumping/jumping left front pocket - may need redoing/Raw Data.csv",
            "run_4": "ivan/jumping/jumping left hand vertical/Raw Data.csv",
            "run_5": "ivan/jumping/jumping right hand vertical/Raw Data.csv",
            "run_6": "ivan/jumping/jumping right pocket/Raw Data.csv",
        },
        "walking": {
            "run_1": "ivan/walking/walking back left pocket/Raw Data.csv",
            "run_2": "ivan/walking/walking back right pocket/Raw Data.csv",
            "run_3": "ivan/walking/walking left front pocket/Raw Data.csv",
            "run_4": "ivan/walking/walking left hand vertical/Raw Data.csv",
            "run_5": "ivan/walking/walking right hand vertical/Raw Data.csv",
            "run_6": "ivan/walking/walking right pocket/Raw Data.csv",
        }
    },
    "letchu": {
        "jumping": {
            "run_1": "member3/jumping/run_1/raw data.csv",
            "run_2": "member3/jumping/run_2/raw data.csv",
        },
        "walking": {
            "run_1": "member3/walking/run_1/raw data.csv",
            "run_2": "member3/walking/run_2/raw data.csv",
        }
    }
}


with h5py.File("my_dataset.hdf5", "w") as h5f:
    raw_data_group = h5f.create_group("Raw data")
    
    for member_name, activities in members.items():
        member_group = raw_data_group.create_group(member_name)
        
        for activity_type, runs in activities.items(): #activities: jump and walk
            activity_group = member_group.create_group(activity_type)
            
            for run_name, csv_path in runs.items():
                run_group = activity_group.create_group(run_name)
                
                df = pd.read_csv(csv_path)
                data_array = df.to_numpy()
                
                dset = run_group.create_dataset("raw data", data=data_array)
                dset.attrs["column_names"] = df.columns.tolist() #save column names
