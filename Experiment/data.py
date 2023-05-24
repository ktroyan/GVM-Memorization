import os
import sys
import argparse

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchsummary import summary

# Data manipulation
import csv
import pandas as pd

from utils import artists_to_class_numbers_bimap
pd.set_option('display.max_columns', None)

# Numerical operations without torch
import numpy as np
import random

# Models, datasets, metrics
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from scipy.stats import uniform

# Visualization
import matplotlib.pyplot as plt

# Tracking of experiments
from tqdm import trange, tqdm
import wandb

# Personal files dependencies
from config import config, CONSTANTS as C


def verify_class_balance(y_train, y_test, class_counts):
    # TODO: not sure if this is a correct way to check the class balance, think of better implementation of this function

    print("Performing a class balance check on the data sets...\n")
    print("y_train: ", y_train)
    print("y_test: ", y_test)

    nb_classes_train = len(np.unique(y_train))
    nb_classes_test = len(np.unique(y_test))
    print("Number of classes in y_train: ", nb_classes_train)
    print("Number of classes in y_test: ", nb_classes_test)
    if nb_classes_train == nb_classes_test:
        print("The train and test sets have the same number of different classes.")
        nb_classes = nb_classes_train
    else:
        raise ValueError("The train and test sets do NOT have the same number of different classes!")


    # count the number of different values in numpy arrays y_train and y_test
    train_counts = np.unique(y_train, return_counts=True)   # return a tuple with the unique classes and the count of samples per class
    test_counts = np.unique(y_test, return_counts=True)   # return a tuple with the unique classes and the count of samples per class

    print("Unique classes in train: ", train_counts[0])
    print("Unique classes in test: ", test_counts[0])

    print("Number of samples per class in train: ", train_counts[1])
    print("Number of samples per class in test: ", test_counts[1])

    # check that the train and test sets have the same different classes
    if not np.array_equal(train_counts[0], test_counts[0]):
        raise ValueError("The train and test sets have different classes!")
    
    # check if the number of samples per class in the train set is 80% and the number per class in the test set is 20%
    warning_flag_class_balance = False
    for i in range(nb_classes):
        # print(train_counts[1][i])
        # print(test_counts[1][i])

        if train_counts[1][i] != (int(config.train_test_ratio * class_counts)) or test_counts[1][i] != (int((10.0 - config.train_test_ratio*10)/10 * class_counts)):  # Note: * 10 and / 10 for numerical stability
            warning_flag_class_balance = True
            break
    
    if warning_flag_class_balance:
        print("WARNING: The train and test sets number of samples per class yields a poor class balance across sets!")
    else:
        print("The train and test sets number of samples per class yields a good class balance across sets.")

def balance_classes(X, y, sample_ids):
    X_train, X_test, y_train, y_test = [], [], [], []
    sample_ids_train, sample_ids_test = [], []

    # NOTE: we assume that the classes are ordered in the same way in the X and y arrays AND 
    # that the number of samples per artist/class before split is the same for all the artists/classes
    # TODO: implement for any number of samples per class in the original/complete dataset. Hence modify class_counts everywhere in this function.

    class_counts = y.value_counts()[0]  # get the number of samples per class (assuming same number for all classes!)
    print("Number of samples per class (assuming same number for all classes!): ", class_counts)
    n_train = int(class_counts * config.train_test_ratio)
    n_test = class_counts - n_train
    nb_classes = y.unique().size

    for class_index in range(nb_classes):
        k = class_index * class_counts
        X_train.extend(X[k:k + n_train])
        y_train.extend(y[k:k + n_train])
        sample_ids_train.extend(sample_ids[k:k + n_train])

        X_test.extend(X[k + n_train:k + n_train + n_test])
        y_test.extend(y[k + n_train:k + n_train + n_test])
        sample_ids_test.extend(sample_ids[k + n_train:k + n_train + n_test])

    # map to numpy arrays since X_train/X_test is a Python list of Torch Tensors
    X_train = list(map(np.asarray, X_train))
    X_test = list(map(np.asarray, X_test))

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    sample_ids_train = np.asarray(sample_ids_train)
    sample_ids_test = np.asarray(sample_ids_test)

    return X_train, X_test, y_train, y_test, sample_ids_train, sample_ids_test


def load_dataset(input_data_path, target_data_path):

    if config.explicit_train_input_path:
        input_data_path = config.explicit_train_input_path
        target_data_path = "/".join(config.explicit_train_input_path.split("/")[:-2]) + "/img_info.csv"

    
    # load input data (i.e., image embeddings) from Tensor in .pt file
    try:
        input_data_pt = torch.load(config.input_data_path)  # load the .pt file as a Tensor containing the generated images embeddings
        print(f"Dataset loaded.")
        print(f"Dataset of generated images' embeddings contain {len(input_data_pt)} samples.")
    except Exception as e:
        print(f"Error when loading the images' embeddings: {e}")

    # load saved target/label data (and additional info) from .csv file into pandas dataframe
    with open(config.target_data_path, 'rb') as f:
        target_data_df = pd.read_csv(f, delimiter=",", encoding="utf-8")


    # In case there is a way to specify the test set explicitly (that will replace the test set after the train-test split)
    if config.explicit_test_input_path:
        diff_realm_input_data_path = config.explicit_train_input_path
        is_filtered = config.explicit_train_input_path.split("/")[-1][0] == "f"
        if is_filtered:
            diff_realm_target_data_path = "/".join(config.explicit_train_input_path.split("/")[:-2]) + "/f_img_info.csv"
        else:
            diff_realm_target_data_path = "/".join(config.explicit_train_input_path.split("/")[:-2]) + "/img_info.csv"
        
        if not config.use_train_split_for_test:
            config.train_test_split = 1.0

        # load input data (i.e., image embeddings) from Tensor in .pt file
        try:
            diff_realm_input_data_pt = torch.load(diff_realm_input_data_path)  # load the .pt file as a Tensor containing the generated images embeddings
            
        except Exception as e:
            print(f"Error when loading the explicit test data: {e}")

        # load saved target/label data (and additional info) from .csv file into pandas dataframe
        with open(diff_realm_target_data_path, 'rb') as f:
            diff_realm_target_data_df = pd.read_csv(f, delimiter=",", encoding="utf-8")

    else:
        # load the different realm dataset if the flag True
        if config.train_test_diff_realm:
            if config.image_realm == "gen":
                diff_realm_input_data_path = config.input_data_path.replace(f"gen", "real").replace(f"sd_{config.sd_version}/", "")
                diff_realm_target_data_path = config.target_data_path.replace(f"gen", "real").replace(f"sd_{config.sd_version}/", "")
            elif config.image_realm == "real":
                diff_realm_input_data_path = config.input_data_path.replace("real", f"gen/sd_{config.sd_version}")
                diff_realm_target_data_path = config.target_data_path.replace("real", f"gen/sd_{config.sd_version}")

            print("OK diff_realm_target_data_path: ", diff_realm_input_data_path, diff_realm_target_data_path)


            # load input data (i.e., image embeddings) from Tensor in .pt file
            try:
                diff_realm_input_data_pt = torch.load(diff_realm_input_data_path)  # load the .pt file as a Tensor containing the generated images embeddings
                print(f"Different realm dataset loaded.")
                print(f"The different realm dataset of generated images' embeddings contain {len(diff_realm_input_data_pt)} samples.")
                
                assert len(diff_realm_input_data_pt[0]) == len(input_data_pt[0])
                print("The embedding size of the different realm datasets' samples is the same.")

            except Exception as e:
                print(f"Error when loading the different realm images' embeddings: {e}")
                print("Are the embedding sizes of the different realm datasets' samples the same?")

            # load saved target/label data (and additional info) from .csv file into pandas dataframe
            with open(diff_realm_target_data_path, 'rb') as f:
                diff_realm_target_data_df = pd.read_csv(f, delimiter=",", encoding="utf-8")

        elif not config.explicit_test_input_path:
            diff_realm_input_data_pt = None
            diff_realm_target_data_df = None

    if diff_realm_input_data_pt is not None and diff_realm_target_data_df is not None:
        # TODO: we slice for hacky purpose, but should change. Data are wrong! NAURYZBAYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
        diff_realm_input_data_pt = diff_realm_input_data_pt[:min(len(diff_realm_input_data_pt), len(diff_realm_target_data_df))]
        diff_realm_target_data_df = diff_realm_target_data_df[:min(len(diff_realm_input_data_pt), len(diff_realm_target_data_df))]


    return input_data_pt, target_data_df, diff_realm_input_data_pt, diff_realm_target_data_df

def rand_shuffle_balanced_classes(X_train, X_test, y_train, y_test, sample_ids_train, sample_ids_test):

    # randomly shuffle the numpy arrays X_train and y_train in the same way
    indices_train = np.random.permutation(len(X_train)) # generate a random permutation of indices
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]
    sample_ids_train = sample_ids_train[indices_train]

    # randomly shuffle the numpy arrays X_test and y_test in the same way
    indices_test = np.random.permutation(len(X_test)) # generate a random permutation of indices
    X_test = X_test[indices_test]
    y_test = y_test[indices_test]
    sample_ids_test = sample_ids_test[indices_test]

    return X_train, X_test, y_train, y_test, sample_ids_train, sample_ids_test

def rand_shuffle(X, y):

    # split the data randomly (and without class balancing across the splits/sets) into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=config.train_test_ratio, 
                                                        test_size=1-config.train_test_ratio, 
                                                        shuffle=config.rand_shuffle_data, 
                                                        random_state=config.seed)

    sample_ids_train = y_test.index.values
    sample_ids_test = y_train.index.values

    # convert the train and test sets back to numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return X_train, X_test, y_train, y_test, sample_ids_train, sample_ids_test

def get_dataset_splits(X, y, sample_ids):

    if config.balance_classes_across_sets:
        X_train, X_test, y_train, y_test, sample_ids_train, sample_ids_test = balance_classes(X, y, sample_ids)

        if config.rand_shuffle_data:

            X_train, X_test, y_train, y_test, sample_ids_train, sample_ids_test = rand_shuffle_balanced_classes(X_train, X_test, y_train, y_test, sample_ids_train, sample_ids_test)

    else:
        if config.rand_shuffle_data:
            X_train, X_test, y_train, y_test, sample_ids_train, sample_ids_test = rand_shuffle(X, y)

    if config.verify_class_balance:
        class_counts = y.value_counts()[0]  # get the number of samples per class (assuming same number for all classes!)
        verify_class_balance(y_train, y_test, class_counts)

    return X_train, X_test, y_train, y_test, sample_ids_train, sample_ids_test

def prepare_data(input_data_pt, target_data_df, input_data_diff_realm=None, target_data_diff_realm=None):
    
    if (config.train_test_diff_realm or config.explicit_train_input_path or config.explicit_test_input_path) and (input_data_diff_realm is not None) :     # TODO: change this as mpt correct
        X_diff_realm = np.asarray(input_data_diff_realm)
        target_data_df_diff_realm, _, _ = artists_to_class_numbers_bimap(target_data_diff_realm)
        y_diff_realm = np.asarray(target_data_df_diff_realm['artist_class_number'].copy())
        sample_ids_diff_realm = target_data_diff_realm['uid'].copy()

    else:
        X_diff_realm = None
        y_diff_realm = None
        sample_ids_diff_realm = None

    X = np.asarray(input_data_pt)
    target_data_df, artists_to_class_numbers, class_numbers_to_artists = artists_to_class_numbers_bimap(target_data_df)
    y = target_data_df['artist_class_number'].copy()
    sample_ids = target_data_df['uid'].copy()

    X_train, X_test, y_train, y_test, sample_ids_train, sample_ids_test = get_dataset_splits(X, y, sample_ids)

    print("Number of samples in X_train: ", len(X_train))
    print("Number of samples in y_train: ", len(y_train))

    if (config.train_test_diff_realm or config.explicit_train_input_path or config.explicit_test_input_path) and(X_diff_realm is not None):
        print("Number of samples in X_diff_realm (used for testing): ", len(X_diff_realm))
        print("Number of samples in y_diff_realm (used for testing): ", len(y_diff_realm))

    else:
        print("Number of samples in X_test: ", len(X_test))
        print("Number of samples in y_test: ", len(y_test))

    sample_size = X_train[1].shape[0]   # number of features of each sample (i.e., size of the embedding vector of each image)

    print("Split the dataset into (X,y) pairs of train and test sets.")

    dataset_splits = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    return sample_size, dataset_splits, sample_ids_train, sample_ids_test, artists_to_class_numbers, class_numbers_to_artists, X_diff_realm, y_diff_realm, sample_ids_diff_realm

def create_dataloaders(X_train, y_train, X_val=None, y_val=None):
    class ArtDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]

    # Note: X_train, y_train, X_val, y_val should all be numpy

    if X_train is not None and y_train is not None:
        # Convert the data to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)

        # Create a PyTorch Dataset
        train_dataset = ArtDataset(X_train, y_train)

        # Create a PyTorch DataLoader
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        # print("Number of batches in train loader: ", len(train_loader))

    else:
        train_loader = None

    if X_val is not None and y_val is not None:
        # Convert the data to PyTorch tensors
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val)

        # Create a PyTorch Dataset
        val_dataset = ArtDataset(X_train, y_train)

        # Create a PyTorch DataLoader
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        # print("Number of batches in val loader: ", len(val_loader))

    else:
        val_loader = None

    dataloaders = {'train_loader': train_loader, 'val_loader': val_loader}

    return dataloaders