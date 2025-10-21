# An example of K-Fold Cross Validation split

import pandas
import numpy
from sklearn.model_selection import KFold

# Configurable constants
NUM_SPLITS = 3

# Load the 2.1-Exercise.csv dataset
data = pandas.read_csv('../docs/2.1-Exercise.csv')

# Select feature columns (excluding Date, Time, and Target)
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[feature_columns].values

# Perform a K-Fold split and print results
kfold = KFold(n_splits=NUM_SPLITS)
split_data = kfold.split(X)

print("""\
The K-Fold method works by splitting off 'folds' of test data until every point has been used for testing.

The following output shows the result of splitting the 2.1-Exercise data.
A bar displaying the current train-test split as well as the actual data points are displayed for each split.
In the bar, "-" is a training point and "T" is a test point.
""")

print("Data shape: {}\n".format(X.shape))
print('K-Fold split (with n_splits = {}):\n'.format(NUM_SPLITS))

for train, test in split_data:
    output_train = ''
    output_test = ''

    bar = ["-"] * (len(train) + len(test))

    # Build our output for display from the resulting split
    for i in train:
        output_train = "{}({}: {}) ".format(output_train, i, X[i])

    for i in test:
        bar[i] = "T"
        output_test = "{}({}: {}) ".format(output_test, i, X[i])

    print("[ {} ]".format(" ".join(bar)))
    print("Train: {}".format(output_train))
    print("Test:  {}\n".format(output_test))
