import h5py
import pandas as py

csv_fies = {"armaan":" ",
            "ivan": " ",
            "letchu": " "
            }

with h5py.File("dataset.hdf5", "w") as h5:
    raw_data = h5.create_group("Raw Data")

    for member_name, csv_path in csv_fies.items():
        df = pd.read_csv(csv_path)

        member_group = raw_data.create_group(member_name)
        data_array = df.to_numpy()
        member_group.create_dataset("data", data=data_array)