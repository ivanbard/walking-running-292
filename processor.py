import pandas as pd
import numpy as np


dataset = pd.read_csv(   )

dataset = dataset.iloc[:, 1:-1]

find_nan = np.where(pd.isna(dataset))

nan_locations = [(int(row), int(col)) for row, col in zip(find_nan[0], find_nan[1])]
print("NaN Location: ", nan_locations)

total_nan = dataset.isna().sum().sum()
print("Total NaN values per column:", total_nan)

find_dash = np.where(dataset == '-')

dash_locations = [(int(row), int(col)) for row, col in zip(find_dash[0], find_dash[1])]
print("Dash indices:",dash_locations)

total_dash = (dataset == '-').sum().sum()
print("Total NaN values per column:", total_dash)

dataset.mask(dataset == '-', other=np.nan, inplace=True)

updated_total_nan = dataset.isna().sum().sum()
print("Updated NaN values per column:", updated_total_nan)

dataset = dataset.astype('float64')

dataset.interpolate(method='linear', inplace=True)

print("Interpolated value at [17,0]:", dataset.iloc[17, 0])