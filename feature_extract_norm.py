import h5py
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def extract_features(df):
    features = {}
    axes = [
        "Linear Acceleration x (m/s^2)",
        "Linear Acceleration y (m/s^2)",
        "Linear Acceleration z (m/s^2)"
    ]
    
    for axis in axes:
        if axis in df.columns:
            signal = df[axis].values
            
            features[f"{axis}_mean"] = np.mean(signal)
            features[f"{axis}_std"] = np.std(signal)
            features[f"{axis}_max"] = np.max(signal)
            features[f"{axis}_min"] = np.min(signal)
            features[f"{axis}_range"] = np.max(signal) - np.min(signal)
            features[f"{axis}_median"] = np.median(signal)
            features[f"{axis}_variance"] = np.var(signal)
            features[f"{axis}_skewness"] = skew(signal)
            features[f"{axis}_kurtosis"] = kurtosis(signal)
            features[f"{axis}_rms"] = np.sqrt(np.mean(signal**2))
    
    return features

hdf5_file = "dataset.hdf5"

feature_rows = []

with h5py.File(hdf5_file, "r") as h5f:
    preproc_group = h5f["Pre-processed data"]
    
    for member in preproc_group.keys():
        member_group = preproc_group[member]
        for activity in member_group.keys():
            activity_group = member_group[activity]
            for run in activity_group.keys():
                run_group = activity_group[run]
                dset = run_group["raw data"]
                
                columns = dset.attrs["column_names"]
                df = pd.DataFrame(dset[...], columns=columns)
                df.columns = df.columns.str.strip()
                
                #decode byte strings if need be
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                
                feats_dict = extract_features(df)
                feats_dict["Activity"] = df["Activity"].iloc[0]
                feats_dict["Person"] = df["Person"].iloc[0]
                feats_dict["Run"] = run  #track what run we are on
                
                feature_rows.append(feats_dict)

features_df = pd.DataFrame(feature_rows)
label_cols = ["Activity", "Person", "Run"]
feature_cols = [c for c in features_df.columns if c not in label_cols]

scaler = StandardScaler()
features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols])

train_df, test_df = train_test_split(features_df, test_size=0.1, random_state=42, stratify=features_df["Activity"])

with h5py.File(hdf5_file, "r+") as h5f:
    if "Segmented data" not in h5f:
        segmented_group = h5f.create_group("Segmented data")
    else:
        segmented_group = h5f["Segmented data"]
    
    #create/overwrite the train and test groups in hdf5
    if "Train" in segmented_group:
        del segmented_group["Train"]
    train_group = segmented_group.create_group("Train")
    
    if "Test" in segmented_group:
        del segmented_group["Test"]
    test_group = segmented_group.create_group("Test")
    
    def store_df_as_hdf5(df, parent_group, name):
        #covert to one type to prevent errors
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('S')
        
        data_array = df.to_records(index=False)
        
        dset = parent_group.create_dataset(name, data=data_array)
        dset.attrs["column_names"] = list(df.columns)
    
    store_df_as_hdf5(train_df, train_group, "features")
    store_df_as_hdf5(test_df, test_group, "features")
    
    print("train and test feature sets stored under 'Segmented data' group")