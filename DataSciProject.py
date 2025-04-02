
# THIS FILE IS USED TO INITIALIZE AND WRITE ALL DATA TO AN HDF5 FILE FOR EFFICIENT USE
# assume all raw csv data is given in the form root/MEMBER/POSITION_ACTION.csv
# after all data is properly loaded into the hdf5 file, csv data can be deleted

from glob import glob

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# define constants
DATA_PATH = "data.h5"
DATA_PATH_RAW = "raw"
DATA_PATH_PREPROCESS = "preprocess"
DATA_PATH_TRAIN_X = "segmented/training/x"
DATA_PATH_TRAIN_Y = "segmented/training/y"
DATA_PATH_TEST_X = "segmented/testing/x"
DATA_PATH_TEST_Y = "segmented/testing/y"

MA_CSV = 34
EWM_CSV = 12
MA_RT = 10
EWM_RT = 4

TEST_SPLIT = 0.1
INTERVAL = 5  # 5-second intervals for testing/training
OVERLAP = 2.5 # overlap between the different intervals of data
COLUMNS = [
    "Linear Acceleration x (m/s^2)",
    "Linear Acceleration y (m/s^2)",
    "Linear Acceleration z (m/s^2)",
    "Absolute acceleration (m/s^2)"
]


# RELOADS ALL DATA, OVERWRITING EXISTING DATA
def load_data(path: str):
    print(f"[DEBUG] Loading data from {path} into hdf5 file {DATA_PATH}.")

    # go through the given directory and find any and all csv files stored in it
    features = []
    with pd.HDFStore(DATA_PATH, mode='w') as h5f:
        for filename in glob(path + '/**/*.csv', recursive=True):
            print(f"[DEBUG] Processing {filename}.")

            # use name format to identify different parameters
            attributes = filename.split('\\')
            member = attributes[-1].lower()
            action, pos = map(str.lower, attributes[2].split('.')[0].split('_'))
            file_id = '_'.join([action, pos])

            # create the meta data dict
            m_data = {
                "member": member,
                "position": pos,
                "action": action
            }

            # open the csv file using pandas
            df = pd.read_csv(filename)
            raw_key = f"{DATA_PATH_RAW}/{member}/{file_id}"

            # write it to the hdf5 file with metadata
            h5f.put(raw_key, df, format='table')
            h5f.get_storer(raw_key).attrs.metadata = m_data
            print(f"--> Finished writing raw data.")

            # process the raw data
            p_df = process_raw(df)
            processed_key = f"{DATA_PATH_PREPROCESS}/{member}/{file_id}"

            # write it to the hdf5 file with metadata
            h5f.put(processed_key, p_df, format='table')
            h5f.get_storer(processed_key).attrs.metadata = m_data
            print(f"--> Finished writing preprocessed data.")

            # extract features from the preprocessed data
            # first split the data into multiple intervals
            intervals = split_frame(p_df, INTERVAL, OVERLAP)
            features += [extract_features(df, action) for df in intervals]

        # create a data frame from all the feature intervals
        f_df = pd.DataFrame(features)
        data = f_df.drop(columns=['label'])
        labels = f_df['label']

        # plot all features against the label to visualize correlation
        # for col in f_df.columns:
        #     if col != 'label':
        #         plt.scatter(f_df[col], f_df['label'], label=col, alpha=0.7)
        #         plt.title(col)
        #         plt.show()

        x_train, x_test, y_train, y_test = train_test_split(
            data, labels,
            test_size=TEST_SPLIT,
            shuffle=True,
            random_state=0,
            stratify=labels  # Handle class imbalance
        )

        # store these new dataframes as segmented data in the hdf5 file
        print("[DEBUG] Wrote segmented training/testing data.")
        h5f.put(DATA_PATH_TRAIN_X, x_train)
        h5f.put(DATA_PATH_TRAIN_Y, y_train)
        h5f.put(DATA_PATH_TEST_X, x_test)
        h5f.put(DATA_PATH_TEST_Y, y_test)


def test_ext_csv(path: str, clf: Pipeline):
    if path.split('.')[-1] != 'csv': return
    df = pd.read_csv(path)
    return test_ext_df(df, clf, split=True, interval=INTERVAL)


def test_ext_df(df: pd.DataFrame, clf: Pipeline, rt=False, split=False, interval=-1):
    p_df = process_raw(df, rt)
    if p_df.empty: return -1

    f_df: pd.DataFrame
    if split:
        # ensure that a split interval was actually set
        if interval <= 0:
            raise ValueError("Interval not set.")

        # split the features and test
        f_df = pd.DataFrame(
            [extract_features(i) for i in split_frame(p_df, interval, 0)]
        )
    else:
        f_df = pd.DataFrame([extract_features(p_df)])

    # drop NaNs (don't interpolate since they are likely to just be at the very beginning of the dataset)
    f_df.dropna(inplace=True)

    # make sure data actually exists
    if f_df.empty: return -1
    return clf.predict(f_df)


def process_raw(df: pd.DataFrame, rt=False) -> pd.DataFrame:
    # fill all null/NaN values with interpolated values (use linear)
    interpolated = df.interpolate(method='linear')

    # run a moving average filter on the data
    filter_ma = interpolated.rolling(MA_RT if rt else MA_CSV).mean()
    filter_ma.dropna(inplace=True)

    # to make the signals smoother, run another exponential moving average filter
    filter_ewm = filter_ma.ewm(EWM_RT if rt else EWM_CSV).mean().dropna()
    filter_ewm.dropna(inplace=True)
    return filter_ewm


# splits the data frame into multiple intervals
def split_frame(df: pd.DataFrame, interval, overlap, col="Time (s)"):
    # make sure overlap is less than the interval
    if overlap >= interval: return
    offset = interval - overlap

    # find the max value in the column of the data set
    max_val = df[col].max()

    # loop through and split the data into different intervals
    intervals = []
    i_start = i_end = 0
    while (i_end := i_start + interval) <= max_val:
        intervals.append(get_interval(df, i_start, i_end))
        i_start += offset

    # loop through and split the data into different intervals
    return intervals


# returns an interval in a column of data between two values/points
def get_interval(df: pd.DataFrame, start, end, end_inclusive=False, col="Time (s)"):
    if end_inclusive: return df[(df[col] >= start) & (df[col] <= end)]
    return df[(df[col] >= start) & (df[col] < end)]


def extract_features(df: pd.DataFrame, action: str = None):
    y_accel_data = df['Linear Acceleration y (m/s^2)']
    z_accel_data = df['Linear Acceleration z (m/s^2)']
    abs_accel_data = df['Absolute acceleration (m/s^2)']

    feature_dict = {
        'lin_accel_y_min': y_accel_data.min(),
        'lin_accel_y_max': y_accel_data.max(),
        'lin_accel_y_range': y_accel_data.max() - y_accel_data.min(),
        'lin_accel_y_std': y_accel_data.std(),
        'lin_accel_y_var': y_accel_data.var(),

        'lin_accel_z_min': z_accel_data.min(),
        'lin_accel_z_max': z_accel_data.max(),
        'lin_accel_z_range': z_accel_data.max() - z_accel_data.min(),
        'lin_accel_z_std': z_accel_data.std(),
        'lin_accel_z_var': z_accel_data.var(),

        'abs_accel_min': abs_accel_data.min(),
        'abs_accel_max': abs_accel_data.max(),
        'abs_accel_range': abs_accel_data.max() - abs_accel_data.min(),
        'abs_accel_mean': abs_accel_data.mean(),
        'abs_accel_std': abs_accel_data.std(),
        'abs_accel_var': abs_accel_data.var(),
    }

    # add label if an action is specified
    if action: feature_dict['label'] = 1 if action == 'jumping' else 0
    return feature_dict