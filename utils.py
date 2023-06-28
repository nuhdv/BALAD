import os
import glob
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler


def get_sub_seqs(x_arr, seq_len=100, stride=1):
    seq_starts = np.arange(0, x_arr.shape[0] - seq_len + 1, stride)
    x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])

    return x_seqs

def get_sub_seqs_label(y, seq_len=100, stride=1):
    seq_starts = np.arange(0, y.shape[0] - seq_len + 1, stride)
    ys = np.array([y[i:i + seq_len] for i in seq_starts])
    y = ys[:, -1]
    return y

def data_standardize(X_train, X_test, remove=False, verbose=False, max_clip=5, min_clip=-4):
    mini, maxi = X_train.min(), X_train.max()
    for col in X_train.columns:
        if maxi[col] != mini[col]:
            X_train[col] = (X_train[col] - mini[col]) / (maxi[col] - mini[col])
            X_test[col] = (X_test[col] - mini[col]) / (maxi[col] - mini[col])
            X_test[col] = np.clip(X_test[col], a_min=min_clip, a_max=max_clip)
        else:
            assert X_train[col].nunique() == 1
            if remove:
                if verbose:
                    print("Column {} has the same min and max value in train. Will remove this column".format(col))
                X_train = X_train.drop(col, axis=1)
                X_test = X_test.drop(col, axis=1)
            else:
                if verbose:
                    print("Column {} has the same min and max value in train. Will scale to 1".format(col))
                if mini[col] != 0:
                    X_train[col] = X_train[col] / mini[col]  # Redundant operation, just for consistency
                    X_test[col] = X_test[col] / mini[col]
                if verbose:
                    print("After transformation, train unique vals: {}, test unique vals: {}".format(
                    X_train[col].unique(),
                    X_test[col].unique()))
    X_train = X_train.values
    X_test = X_test.values
    return X_train, X_test

def data_normalization(x_train, x_test):
    x_train = x_train.values
    x_test = x_test.values

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test

def get_data_lst(data, data_root, norm='min-max'):

    machine_lst = [os.path.split(p)[1] for p in sorted(glob.glob(os.path.join(data_root, data, '*')))]

    all_data, all_label = [], []

    for ii, m in enumerate(machine_lst):
        train_path = glob.glob(os.path.join(data_root, data, m, '*train*.csv'))
        test_path = glob.glob(os.path.join(data_root, data, m, '*test*.csv'))

        assert len(train_path) == 1 and len(test_path) == 1
        train_path, test_path = train_path[0], test_path[0]

        train_df = pd.read_csv(train_path, sep=',', index_col=0)
        test_df = pd.read_csv(test_path, sep=',', index_col=0)
        labels = test_df['label'].values
        train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)

        if data == 'PSM':
            train_df.interpolate(inplace=True)
            train_df.bfill(inplace=True)
            test_df.interpolate(inplace=True)
            test_df.bfill(inplace=True)

        # standardize
        if norm == 'min-max':
            train, test = data_standardize(train_df, test_df)
        else:
            train, test = data_normalization(train_df, test_df)

        if 'MSL' in data or 'SMAP' in data:
            train = train[:, 0]
            test = test[:, 0]

        all_data.extend(test)
        all_label.extend(labels)

    all_data, all_label = np.array(all_data), np.array(all_label)

    if len(all_data.shape) == 1:
        all_data = all_data.reshape(len(all_data), 1)

    print(f'Counter: '
          f'all normal {Counter(all_label)[0]}, '
          f'all anomaly {Counter(all_label)[1]}, '
          )
    return all_data, all_label, data