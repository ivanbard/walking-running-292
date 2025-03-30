import h5py  # processed data will need to be visualized and compared with the raw
import pandas as pd
import numpy as np

def preprocess_df(df, window_size=5):
    if df.isnull().values.any():
        df = df.interpolate(method='linear')  # fill data using interpolation
    
    columns_to_smooth = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)"
    ]
    
    for col in columns_to_smooth:  # moving average filter on each column
        if col in df.columns:
            df[col] = df[col].rolling(window=window_size, min_periods=1, center=True).mean()

    df = df.reset_index(drop=True)
    return df

hdf5_file = "dataset.hdf5"
with h5py.File(hdf5_file, "r+") as h5f:
    raw_group = h5f["Raw data"]  # open raw data for processing
    
    if "Pre-processed data" not in h5f:
        preproc_group = h5f.create_group("Pre-processed data")
    else:
        preproc_group = h5f["Pre-processed data"]

    for member in raw_group.keys():  # go through raw data of all members
        member_raw = raw_group[member]
        if member not in preproc_group:
            member_preproc = preproc_group.create_group(member)
        else:
            member_preproc = preproc_group[member]
        
        for activity in member_raw.keys():  # go through all activities
            activity_raw = member_raw[activity]
            if activity not in member_preproc:
                activity_preproc = member_preproc.create_group(activity)
            else:
                activity_preproc = member_preproc[activity]
            
            for run in activity_raw.keys():  # go through all the runs
                run_raw = activity_raw[run]
                dset = run_raw["raw data"]
                columns = dset.attrs["column_names"]
                
                df = pd.DataFrame(dset[...], columns=columns)  # convert data set to dataframe
                df.columns = df.columns.str.strip()

                #label data with jumping or walking
                df["Activity"] = activity.capitalize()
                df["Activity"] = df["Activity"].map({"Walking": 0, "Jumping": 1})
                df["Person"] = member
                
                df_preprocessed = preprocess_df(df, window_size=5)  # call processing func

                #convert object columns to fixed-length strings to avoid dtype 'O' error
                for col in df_preprocessed.select_dtypes(include=['object']).columns:
                    df_preprocessed[col] = df_preprocessed[col].astype('S')
                
                if run in activity_preproc:
                    del activity_preproc[run]  # delete previous processed data if it already exists
                run_preproc = activity_preproc.create_group(run)
                data_array = df_preprocessed.to_records(index=False)
                dset_preproc = run_preproc.create_dataset("raw data", data=data_array)
                
                dset_preproc.attrs["column_names"] = list(df_preprocessed.columns)
                print(f"Processed and stored {member} -> {activity} -> {run}")