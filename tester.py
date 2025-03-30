import h5py #GPT script for checking hdf5 file structure
import pandas as pd

def print_hdf5_structure(group, indent=0):
    """
    Recursively prints the structure of an HDF5 group.
    """
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print("    " * indent + f"Group: {key}")
            print_hdf5_structure(item, indent + 1)
        else:
            print("    " * indent + f"Dataset: {key}, shape: {item.shape}, dtype: {item.dtype}")

hdf5_file = "dataset.hdf5"

# Open the file and print its structure
with h5py.File(hdf5_file, "r") as h5f:
    print("Full HDF5 File Structure:")
    print_hdf5_structure(h5f)

    # Now check the segmented data
    if "Segmented data" in h5f:
        segmented_group = h5f["Segmented data"]
        for set_name in ["Train", "Test"]:
            if set_name in segmented_group:
                print(f"\nStructure under 'Segmented data/{set_name}':")
                print_hdf5_structure(segmented_group[set_name])
                
                # Load the features dataset from the segmented group
                if "features" in segmented_group[set_name]:
                    dset = segmented_group[set_name]["features"]
                    # Convert the structured array into a pandas DataFrame
                    columns = dset.attrs["column_names"]
                    df = pd.DataFrame(dset[...])
                    
                    # For any byte strings, decode them (if needed)
                    for col in df.select_dtypes(include=[object]).columns:
                        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                    
                    print(f"\nFirst few rows of '{set_name}' features:")
                    print(df.head())
            else:
                print(f"'Segmented data/{set_name}' group not found.")
