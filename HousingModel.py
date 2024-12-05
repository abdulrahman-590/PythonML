import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

download_root = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
data_link = download_root + "datasets/housing/housing.tgz"
data_path = os.path.join("datasets", "housing")


def fetch_data(url, dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    tgz_path = os.path.join(dir_path, "housing.tgz")
    urllib.request.urlretrieve(url, tgz_path)
    data_tgz = tarfile.open(tgz_path)
    data_tgz.extractall(dir_path)
    data_tgz.close()


def load_data(path):
    csv_path = os.path.join(path, "housing.csv")
    return pd.read_csv(csv_path)


# method not recommended as the data will be random each time
def split_train_test_data(data, test_ratio): # train_test_split of sckikit almost does the same thing as this function
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[in_test_set], data.loc[in_test_set]


data = load_data(data_path)

# below line adds an index column

data_with_id = data.reset_index()

data_with_id["id"] = data["longitude"] * 1000 + data["latitude"]

train_set, test_set = split_train_test_by_id(data_with_id, 0.2, "id")
print(len(test_set))




#fetch_data(data_link, data_path)


# data.hist(bins=56, figsize=(20,15))
# plt.show()

# If data is small use stratified sampling split to avoid biasness

data["income_cat"] = pd.cut(data["median_income"],
                            bins=[0,1.5,3.0,4.5,6., np.inf],
                            labels=[1,2,3,4,5]
                            )



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["income_cat"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# for set_ in (strat_train_set, strat_test_set):
#     set_.drop("income_cat", axis=1, inplace=True)

data = strat_train_set.copy()
 
