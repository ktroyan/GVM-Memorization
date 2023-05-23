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

# Other files dependencies
from config import Configuration
from config import CONSTANTS as C


def seed_everything(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False   # True for deterministic
    torch.backends.cudnn.benchmark = True  # False for deterministic

def show_cuda_gpu_info():
    if torch.cuda.is_available():
        print("Cuda is available.")
        print(torch.cuda.current_device())
        # print(torch.cuda.device_count())
        # print(torch.cuda.get_device_name(0))
        # print(torch.cuda.get_device_capability(0))
        # print(torch.cuda.get_device_properties(0))
        # print(torch.cuda.memory_allocated(0))
        # print(torch.cuda.memory_cached(0))
        # print(torch.cuda.memory_summary(0))
    else:
        print("Cuda is not available.")

def load_dataset(input_data_path, target_data_path):

    if config.input_data_path:
        input_data_path = config.input_data_path
    if config.target_data_path:
        target_data_path = config.target_data_path

    # load input data (i.e., image embeddings) from Tensor in .pt file
    try:
        input_data_pt = torch.load(input_data_path)  # load the .pt file as a Tensor containing the generated images embeddings
        print(f"Dataset loaded.")
        print(f"Dataset of generated images' embeddings contain {len(input_data_pt)} samples.")
    except Exception as e:
        print(f"Error when loading the images' embeddings: {e}")

    # load saved target/label data (and additional info) from .csv file into pandas dataframe
    with open(target_data_path, 'rb') as f:
        target_data_df = pd.read_csv(f, delimiter=",", encoding="utf-8")

    # load the different realm dataset if the flag True
    if config.train_test_diff_realm:
        if config.image_realm == "gen":
            diff_realm_input_data_path = input_data_path.replace(f"gen_sd_{config.sd_version}", "real")
            diff_realm_target_data_path = target_data_path.replace(f"gen_sd_{config.sd_version}", "real")
        elif config.image_realm == "real":
            diff_realm_input_data_path = input_data_path.replace("real", f"gen_sd_{config.sd_version}")
            diff_realm_target_data_path = target_data_path.replace("real", f"gen_sd_{config.sd_version}")

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

    else:
        diff_realm_input_data_pt = None
        diff_realm_target_data_df = None

    return input_data_pt, target_data_df, diff_realm_input_data_pt, diff_realm_target_data_df

def set_header(results_path, data_writer, data_header):
    # write header of the csv file if there is no header yet
    if os.stat(results_path).st_size == 0:
        data_file_has_header = False
        data_writer.writerow(data_header)
    else:
        data_file_has_header = True
    return data_file_has_header

def setup_wandb(model, lr, optimizer, criterion):

    if config.use_wandb:

        wandb.init(project="GVM-project",
                config={
                        "model": config.pred_model_name,
                        "epochs": config.n_epochs,
                        # "batch_size": config.batch_size,  # uncomment if dataloaders are used
                        "learning_rate": lr,
                        "optimizer": optimizer,
                        "criterion": criterion,
                        "device": C.DEVICE,
                        "input_data_file": input_data_path,
                        "target_data_file": target_data_path
                        })

        # wandb.login()

        # set model to wandb
        wandb.watch(model, log_freq=100)

    else:
        wandb.init(mode="disabled")

def create_NN(nb_features, nb_classes):

    print("Number of features: ", nb_features)
    print("Number of classes: ", nb_classes)

    if config.pred_model_name == "linear_nn":
        
        class NNLogisticRegression(torch.nn.Module):    
            def __init__(self, nb_features, nb_classes):
                super(NNLogisticRegression, self).__init__()
                self.linear = torch.nn.Linear(nb_features, nb_classes)

            def forward(self, x):
                y_pred = self.linear(x)
                return y_pred
        
        model = NNLogisticRegression(nb_features, nb_classes)
        
    elif config.pred_model_name == "nn":
        
        class MLP(torch.nn.Module):    
            def __init__(self, nb_features, nb_classes):
                super(MLP, self).__init__()
                self.linear1 = torch.nn.Linear(nb_features, 512)
                self.linear2 = torch.nn.Linear(512, 256)
                self.hidden_layers = [self.linear1, self.linear2]
                self.linear_out = torch.nn.Linear(256, nb_classes)

                self.relu = torch.nn.ReLU()

            def forward(self, x):
                for layer in self.hidden_layers:
                    x = self.relu(layer(x))
                y_pred = self.linear_out(x)
                return y_pred
        
        model = MLP(nb_features, nb_classes)

    else:
        raise ValueError(f"When creating a NN model, got an unknown model name: {config.pred_model_name}")

    input_dims = (1, nb_features)   # without batch size
    summary(model.to(C.DEVICE), input_dims)
    return model

def create_model(nb_features=None, nb_classes=None):

    if config.pred_model_name == "logistic_regression":
        if config.add_verbose:
            verbose = 4
        else: 
            verbose = 0

        model = LogisticRegression(penalty='l2', 
                                   dual=False,
                                   tol=1e-4,
                                   C=1,
                                   solver='lbfgs',
                                   max_iter=500, 
                                   n_jobs=-1, 
                                   warm_start=False,
                                   verbose=verbose)

    elif config.pred_model_name == "xgboost":
        model = XGBClassifier(objective='multi:softmax', 
                              num_class=nb_classes, 
                              n_jobs=-1, 
                              tree_method="gpu_hist")
    
        # Set the XGBoost model parameters; see all: https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
        params_grid = {'base_score': 0.5, 
                        # 'colsample_bylevel': 1, 
                        # 'colsample_bytree': 1,
                        # 'gamma': 0, 
                        'learning_rate': 0.1, 
                        # 'max_delta_step': 0, 
                        'max_depth': 10,
                        # 'min_child_weight': 1, 
                        # 'missing': None, 
                        'n_estimators': 100, 
                        # 'nthread': -1,
                        # 'reg_alpha': 0,
                        # 'reg_lambda': 1,
                        # 'scale_pos_weight': 1, 
                        'seed': 121997, 
                        # 'subsample': 1
                        }
        model.set_params(**params_grid)
    
    elif config.pred_model_name == "linear_nn":
        model = create_NN(nb_features, nb_classes)
        model = model.to(C.DEVICE)

    elif config.pred_model_name == "nn":
        model = create_NN(nb_features, nb_classes)
        model = model.to(C.DEVICE)
    else:
        raise ValueError(f"Before creating the model, got an invalid model name: {config.pred_model_name}")

    return model

# create a new column/feature with the class numbers (obtained from the artist classes)
def artists_to_class_numbers_bimap(target_data_df):

    artists = target_data_df['artist'].copy()
    unique_artists = artists.unique()

    artists_to_class_numbers = {artist: i for i, artist in enumerate(unique_artists)}
    class_numbers_to_artists = {str(i): artist for i, artist in enumerate(unique_artists)}

    # create a column with the artist class numbers in the dataframe
    target_data_df['artist_class_number'] = target_data_df['artist'].map(artists_to_class_numbers)

    return target_data_df, artists_to_class_numbers, class_numbers_to_artists

def get_artists_from_class_numbers(predictions, class_numbers_to_artists):
    pred_artists = [class_numbers_to_artists[str(int(pred.item()))] for pred in predictions]
    return pred_artists

def get_artists_ranking(ranked_artist_classes, class_numbers_to_artists):
    return [[class_numbers_to_artists[str(artist_class.item())] for artist_class in sample] for sample in ranked_artist_classes]

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
    
    if config.train_test_diff_realm:
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
    print("Number of samples in y_train ", len(y_train))

    if config.train_test_diff_realm:
        print("Number of samples in X_diff_realm (used for testing): ", len(X_diff_realm))
        print("Number of samples in y_diff_realm (used for testing)", len(y_diff_realm))

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
        X_train = torch.LongTensor(X_train)
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
        X_val = torch.LongTensor(X_val)
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

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    return metrics

def compute_topk_accuracy(sorted_probabilities, class_ranking_indices, gt_labels, k_list=[1], method = "method1"):

    top_k_accuracy_dict = {}
    top_k_classes_dict = {}

    for index, k in enumerate(k_list):

        if method == "method1":
            top_k_classes = class_ranking_indices[:, :k]
            correct_predictions = torch.sum(top_k_classes == torch.Tensor(gt_labels).unsqueeze(1), dim=1)   # creare boolean Tensor and sume (True is 1 and False is 0)
            topk_accuracy = torch.mean(correct_predictions.float()).item()
        
        elif method == "method2":
            top_probs, top_k_classes = sorted_probabilities.cpu().topk(k, dim=-1)
            topk_correct = 0
            for i in range(k):
                top_one_labels = top_k_classes[:, i]
                correct = torch.sum(torch.Tensor(top_one_labels) == torch.Tensor(gt_labels))
                topk_correct += correct
            topk_accuracy = (topk_correct / gt_labels.shape[0]).item()
        
        top_k_accuracy_dict[str(k)] = topk_accuracy
        top_k_classes_dict[str(k)] = top_k_classes

    return top_k_accuracy_dict, top_k_classes_dict

def predict(model, X, y, X_val=None, y_val=None, train_loader=None, val_loader=None):

    if config.pred_model_name == "logistic_regression":
        y_pred = model.predict(X)
        
        metrics = compute_metrics(y, y_pred)
        print(f"Accuracy: {metrics['accuracy']*100:.3}%")  # for an equivalent to predict and compute accuracy metric: model.score(X, y)

        probabilities = model.predict_proba(X)
        probabilities = torch.Tensor(probabilities)

    elif config.pred_model_name == "xgboost":
        y_pred = model.predict(X)
        metrics = compute_metrics(y, y_pred)
        print(f"Accuracy: {metrics['accuracy']*100:.3}%")

        probabilities = model.predict_proba(X)
        probabilities = torch.Tensor(probabilities)

    elif config.pred_model_name in ["linear_nn", "nn"]:
        with torch.no_grad():
            logits = model(torch.Tensor(np.asarray(X)).to(C.DEVICE))
            probabilities = F.log_softmax(logits.cpu(), dim=1)    # dim=1 to compute along the class dimension; TODO: why simple softmax not working well? 
            _, y_pred = torch.max(probabilities, 1)

            # accuracy = (torch.Tensor(np.asarray(y_pred)) == torch.Tensor(np.asarray(y))).float().mean()
            # accuracy = accuracy.item()
            # print(f"Accuracy: {accuracy*100:.3}%")

            metrics = compute_metrics(torch.Tensor(np.asarray(y)), torch.Tensor(np.asarray(y_pred)))
            print(f"Accuracy: {metrics['accuracy']*100:.3}%")
            
    else:
        raise ValueError(f"Before predicting, got an invalid model name: {config.pred_model_name}")

    if config.multi_top_k:
        k_list = [1, 3, 5]
    else:
        k_list = [config.topk]

    # sort the predictions probabilities in descending order; get the ranking indices of classes/artists
    sorted_probabilities, class_rankings_indices = torch.sort(probabilities, descending=True)  # if element at i is j, then class j is the top-i prediction

    # get the top-k predicted classes and compute top-k accuracy
    top_k_accuracy_dict, top_k_classes_dict = compute_topk_accuracy(sorted_probabilities, class_rankings_indices, y, k_list)
    
    for k in k_list:
        print(f"Top {k} accuracy is {top_k_accuracy_dict[str(k)]*100:.3}%")
        
    y = torch.Tensor(np.asarray(y))
    y_pred = torch.Tensor(np.asarray(y_pred))

    return y, y_pred, metrics, top_k_classes_dict, top_k_accuracy_dict, class_rankings_indices, sorted_probabilities

def train(model, X_train, y_train, X_test, y_test, train_loader=None, val_loader=None, criterion=None, optimizer=None):
    
    print(f"Training the model {config.pred_model_name}...")

    if config.pred_model_name == "logistic_regression":

        if config.use_grid_search:
            nb_params_combinations = 5
            skf = StratifiedKFold(n_splits=config.k_folds, shuffle=False)

            params_grid = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])

            if config.add_verbose:
                verbose = 4
            else: 
                verbose = 0

            # see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV
            # model = GridSearchCV(model, 
            #                      param_distributions=params_grid, 
            #                      n_iter=nb_params_combinations, 
            #                      scoring='accuracy',
            #                      n_jobs=4, 
            #                      cv=skf.split(X_train,y_train),
            #                      verbose=verbose, 
            #                      random_state=121997)

            # see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
            model = RandomizedSearchCV(model,
                                       param_distributions=params_grid,
                                       n_iter=nb_params_combinations,
                                       scoring='accuracy',  # top_k_accuracy, balanced_accuracy, roc_auc, jaccard. See: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                                       n_jobs=4, 
                                       cv=skf.split(X_train,y_train), 
                                       verbose=verbose, 
                                       random_state=121997)
            
            # set the model with the best hyper-parameters found during the (random) grid search
            model.set_params(**model.best_params_)
        
            model.fit(X_train, y_train, verbose=verbose)
        
        else: 
            model.fit(X_train, y_train)   # here we could also pass an eval set with eval_set=[(X_val, y_val)]

    elif config.pred_model_name == "xgboost":
        if config.add_verbose:
            verbose = 4
        else: 
            verbose = 0

        if config.use_grid_search:     # see: https://www.kaggle.com/code/tilii7/hyperparameter-grid-search-with-xgboost/notebook
            nb_params_combinations = 5

            skf = StratifiedKFold(n_splits=config.k_folds, shuffle=False)

            params_grid = {
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [3, 4, 5]
                }
            
            # see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV
            # model = GridSearchCV(model, 
            #                      param_distributions=params_grid, 
            #                      n_iter=nb_params_combinations, 
            #                      scoring='accuracy',
            #                      n_jobs=4, 
            #                      cv=skf.split(X_train,y_train),
            #                      verbose=verbose, 
            #                      random_state=121997)

            # see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
            model = RandomizedSearchCV(model, 
                                       param_distributions=params_grid, 
                                       n_iter=nb_params_combinations, 
                                       scoring='accuracy',  # top_k_accuracy, balanced_accuracy, roc_auc, jaccard. See: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                                       n_jobs=4, 
                                       cv=skf.split(X_train,y_train), 
                                       verbose=verbose, 
                                       random_state=121997)

            if config.add_verbose:
                print('\n Best estimator:')
                print(model.best_estimator_)
                print('\n Best score:')
                print(model.best_score_)
                print('\n Best hyper-parameters:')
                print(model.best_params_)

            # set the model with the best hyper-parameters found during the (random) grid search
            model.set_params(**model.best_params_)

        model.fit(X_train, y_train, verbose=verbose)   # here we could also pass an eval set with eval_set=[(X_val, y_val)]

    elif config.pred_model_name in ["linear_nn", "nn"]:

        if config.use_tqdm:
            t = trange(config.n_epochs, desc='Loss', leave=True)
        else:
            t = range(config.n_epochs)

        # Train the model
        for epoch in t:  # number of epochs
            for X, y in train_loader:

                # Set data to GPU if possible
                X = X.to(C.DEVICE)
                y = y.to(C.DEVICE)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                logits = model(X.float())

                probabilities = F.log_softmax(logits, dim=1)    # TODO: why simple softmax not working well?

                # Compute the loss
                loss = criterion(probabilities, y)     # no need to apply softmax if criterion is PyTorch CrossEntropyLoss()

                # Backward pass
                loss.backward()

                # Optimization step (update weights/parameters)
                optimizer.step()

            if config.use_tqdm:
                t.set_description(f"loss: {loss.item()}")
            else:
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

            wandb.log({f"Epoch loss: ": loss.item()})

            if (epoch+1) % config.print_at_n_epochs == 0:
                print("Predicting on train set at epoch ", epoch+1, "...")
                _, _, metrics, top_k_classes_dict, top_k_accuracy_dict, class_ranking_indices, probabilities = predict(model, X_train, y_train)   # Sanity check: evaluate the model on the training set
                wandb.log({f"Every {config.print_at_n_epochs} epochs train accuracy: ": metrics['accuracy']*100})
                for k in top_k_accuracy_dict.keys():
                    wandb.log({f"Every {config.print_at_n_epochs} epochs train TOP-{k} accuracy: ": top_k_accuracy_dict[k]*100})
        
    else:
        raise ValueError(f"Before training, got an invalid model name: {config.pred_model_name}")
    
    print("\n")
    # print("Predicting on train set...")
    # y, y_pred, metrics top_k_classes_dict, top_k_accuracy_dict, class_ranking_indices, probabilities = predict(model, pred_model_name, X_train, y_train, val_loader=None, val_loader=None, k=topk)   # Sanity check: evaluate the model on the training set
    # print("Predicting on test set...")
    # y, y_pred, metrics, top_k_classes_dict, top_k_accuracy_dict, class_ranking_indices, probabilities = predict(model, pred_model_name, X_test, y_test, val_loader=None, val_loader=None, k=topk)    # Actual evaluation: evaluate the model on the test set

    return model

def run_training(model, dataset_splits):

    if config.pred_model_name in ['logistic_regression', 'xgboost']:
        trained_model = train(model, dataset_splits['X_train'], dataset_splits['y_train'], dataset_splits['X_test'], dataset_splits['y_test'])

    elif config.pred_model_name == 'linear_nn':
        dataloaders = create_dataloaders(dataset_splits['X_train'], dataset_splits['y_train'])
        
        nb_features = dataset_splits['X_train'].shape[1]  # length of the image embeddings

        learning_rate = 0.1 # use 0.1 for SGD and 0.0001 for Adam

        # Define the loss function and the optimizer for the NN model
        criterion = nn.NLLLoss()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # setup WandB
        setup_wandb(model, learning_rate, optimizer, criterion)

        trained_model = train(model, dataset_splits['X_train'], dataset_splits['y_train'], dataset_splits['X_test'], dataset_splits['y_test'], dataloaders['train_loader'], dataloaders['val_loader'], criterion, optimizer)


    elif config.pred_model_name == 'nn':
        dataloaders = create_dataloaders(dataset_splits['X_train'], dataset_splits['y_train'])

        nb_features = dataset_splits['X_train'].shape[1]  # length of the image embeddings

        learning_rate = 0.1 # use 0.1 for SGD and 0.0001 for Adam

        # Define the loss function and the optimizer
        criterion = nn.NLLLoss()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # setup WandB
        setup_wandb(model, learning_rate, optimizer, criterion)

        trained_model = train(model, dataset_splits['X_train'], dataset_splits['y_train'], dataset_splits['X_test'], dataset_splits['y_test'], dataloaders['train_loader'], dataloaders['val_loader'], criterion, optimizer)
    
    else:
        raise ValueError(f"Before starting training, got an invalid model name: {config.pred_model_name}")

    return trained_model

def save_artists_ranking(file_path, sample_ids, pred_artists, rankings, probabilities):

    # create list of list of strings; given 'probabilities' a Tensor of samples, convert the probabilities for artists within the samples to strings
    probabilities = [[str(prob.item()) for prob in sample_probs] for sample_probs in probabilities]

    with open(file_path, 'w', encoding="utf-8") as f:
        data_writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        data_writer.writerow(['original_sample_id', 'pred_artist', 'artists_ranking', 'artists_probabilities'])
        for sample_id, pred_artist, artists_ranking, artists_prob in zip(sample_ids, pred_artists, rankings, probabilities):
            data_writer.writerow([sample_id, pred_artist, ";".join(artists_ranking), ";".join(artists_prob)])  

def create_class_rankings(ranking_file_path, y_pred, sample_ids, class_ranking_indices, class_numbers_to_artists, probabilities):
    pred_artists = get_artists_from_class_numbers(y_pred, class_numbers_to_artists)   # convert class numbers predictions to explicit artist classes (i.e., names of the artists)
    rankings = get_artists_ranking(class_ranking_indices, class_numbers_to_artists)
    save_artists_ranking(ranking_file_path, sample_ids, pred_artists, rankings, probabilities)

def get_rounded_score(score, digit_precision=3):
    return round(score*100, digit_precision)

def run_evaluation(trained_pred_model, dataset_splits, sample_ids_train, sample_ids_test, nb_classes, class_numbers_to_artists, X_diff_realm=None, y_diff_realm=None, sample_ids_diff_realm=None, results_writer=None):

    if config.pred_model_name in ['linear_nn', 'nn']:
        dataloaders = create_dataloaders(dataset_splits['X_train'], dataset_splits['y_train'])    # create the dataloaders for NN model training
        train_loader = dataloaders['train_loader']
        val_loader = dataloaders['val_loader']
        X_val = None
        y_val = None

    elif config.pred_model_name in ['logistic_regression', 'xgboost']:
        train_loader = None
        val_loader = None

    else:
        raise ValueError(f"Before starting evaluation, got an invalid model name: {config.pred_model_name}")
    
    # predict on train set
    print("Predict on train set...")
    y_train, y_pred_train, metrics_train, top_k_classes_dict_train, top_k_accuracy_dict_train, class_ranking_indices_train, probabilities_train = predict(trained_pred_model, dataset_splits['X_train'], dataset_splits['y_train'], X_val=None, y_val=None, train_loader=train_loader, val_loader=val_loader)

    if class_ranking_indices_train is not None:
        train_ranking_file_path = './Experiment/Data/train_set_artists_ranking.csv'
        create_class_rankings(train_ranking_file_path, y_pred_train, sample_ids_train, class_ranking_indices_train, class_numbers_to_artists, probabilities_train)

    # predict on test set
    if config.train_test_diff_realm:
        print("Predict on test set of different realm...")
        print("Trained on realm: ", config.image_realm)
        print("Testing on realm: ", "real" if config.image_realm == "gen" else "gen")
        if config.image_realm == "gen":
            train_image_realm = "gen"
            test_image_realm = "real"
        else:
            train_image_realm = "real"
            test_image_realm = "gen"

        y_test, y_pred_test, metrics_test, top_k_classes_dict_test, top_k_accuracy_dict_test, class_ranking_indices_test, probabilities_test = predict(trained_pred_model, X_diff_realm, y_diff_realm)
        sample_ids_test = sample_ids_diff_realm

    else:
        print("Predict on test set...")
        y_test, y_pred_test, metrics_test, top_k_classes_dict_test, top_k_accuracy_dict_test, class_ranking_indices_test, probabilities_test = predict(trained_pred_model, dataset_splits['X_test'], dataset_splits['y_test'])
        
        train_image_realm = config.image_realm
        test_image_realm = config.image_realm

    if class_ranking_indices_test is not None:
        print("For test set, saving in file ranked artists for all samples.")
        test_ranking_file_path = './Experiment/Data/test_set_artists_ranking.csv'
        create_class_rankings(test_ranking_file_path, y_pred_test, sample_ids_test, class_ranking_indices_test, class_numbers_to_artists, probabilities_test)

    # save results
    if config.save_experiment_results:
        print("Saving experiment setting and results in a file...")

        if config.multi_top_k:

            results_writer.writerow([train_image_realm,
                                    test_image_realm,
                                    config.artist_category,
                                    config.clip_version.lower(), 
                                    str(config.sd_version).replace('.', '_'),
                                    config.pred_model_name,
                                    str(nb_classes),
                                    f"{str(get_rounded_score(metrics_train['accuracy']))}",
                                    f"{str(get_rounded_score(metrics_test['accuracy']))}",
                                    f"{str(get_rounded_score(top_k_accuracy_dict_test['1']))}",
                                    f"{str(get_rounded_score(top_k_accuracy_dict_test['3']))}",
                                    f"{str(get_rounded_score(top_k_accuracy_dict_test['5']))}",
                                    f"{str(get_rounded_score(metrics_test['precision']))}",
                                    f"{str(get_rounded_score(metrics_test['recall']))}",
                                    f"{str(get_rounded_score(metrics_test['f1']))}",
                                    ])
                  
        else:
            results_writer.writerow([train_image_realm,
                                    test_image_realm,
                                    config.artist_category,
                                    config.clip_version.lower(), 
                                    str(config.sd_version).replace('.', '_'),
                                    config.pred_model_name,
                                    str(nb_classes),
                                    f"{str(get_rounded_score(metrics_train['accuracy']))}",
                                    f"{str(get_rounded_score(metrics_test['accuracy']))}",
                                    None,
                                    None,
                                    None,
                                    f"{str(get_rounded_score(metrics_test['precision']))}",
                                    f"{str(get_rounded_score(metrics_test['recall']))}",
                                    f"{str(get_rounded_score(metrics_test['f1']))}",
                                    ])
    
    print("\n")

def run_experiment(dataset_splits, sample_ids_train, sample_ids_test, sample_size, nb_classes, class_numbers_to_artists, X_diff_realm=None, y_diff_realm=None, sample_ids_diff_realm=None, results_writer=None):
    if config.pred_model_name in ["logistic_regression", "xgboost", "nn", "linear_nn"]:
        
        # create the predictive model
        pred_model = create_model(sample_size, nb_classes)
        
        # train the predictive model
        trained_pred_model = run_training(pred_model, dataset_splits)

        if config.eval_mode:
            # evaluate the predictive model
            run_evaluation(trained_pred_model, dataset_splits, sample_ids_train, sample_ids_test, nb_classes, class_numbers_to_artists, X_diff_realm, y_diff_realm, sample_ids_diff_realm, results_writer)

    else:
        print("The predictive model name is not valid. Please choose one of the following: logistic_regression, xgboost, nn, linear_nn")

if __name__ == '__main__':

    # display information on cuda/GPU
    show_cuda_gpu_info()

    # get the config
    config = Configuration.parse_cmd()

    # seed everything for reproducibility
    seed_everything(config.seed)

    # format the command line arguments given
    config.sd_version = str(config.sd_version).replace('.', '_')
    config.clip_version = config.clip_version.lower()
    config.image_realm = config.image_realm.lower()

    data_folder_path = "./Experiment/Data/"

    # define the data folder path
    if config.artist_category == 'historical':
        data_files_folder_path = data_folder_path + "historical/"
    elif config.artist_category == 'artstation':
        data_files_folder_path = data_folder_path + "artstation/"

    if config.image_realm == 'real':
        input_data_path = f'{data_files_folder_path}{config.artist_category}_{config.image_realm}_{config.clip_version}_embeddings.pt'
        target_data_path = f'{data_files_folder_path}{config.artist_category}_{config.image_realm}_artworks.csv'

    elif config.image_realm == 'gen':
        input_data_path = f'{data_files_folder_path}{config.artist_category}_{config.image_realm}_sd_{config.sd_version}_{config.clip_version}_embeddings.pt'
        target_data_path = f'{data_files_folder_path}{config.artist_category}_{config.image_realm}_sd_{config.sd_version}_artworks.csv'

    if config.save_experiment_results:
        # open file and create writer to save the data
        results_path = f"{data_folder_path}experiments_results.csv"
        results_csv_file = open(results_path, 'a', encoding="utf-8")
        results_writer = csv.writer(results_csv_file, delimiter="\t", lineterminator="\n")
        results_header = ['train_realm', 'test_realm', 'source', 'encoder', 'sd_version', 'pred_model', 'nb_classes', 'train_accuracy', 'test_accuracy', 'top1', 'top3', 'top5', 'precision', 'recall', 'f1_score']
        results_file_has_header = set_header(results_path, results_writer, results_header)
    else:
        results_writer = None

    # load the dataset
    input_data_pt, target_data_df, diff_realm_input_data, diff_realm_target_data = load_dataset(input_data_path, target_data_path)
    
    # dataset_splits is a dict with the splitted dataset subsets: X_train, X_test, y_train, y_test
    sample_size, dataset_splits, sample_ids_train, sample_ids_test, artists_to_class_numbers, class_numbers_to_artists, X_diff_realm, y_diff_realm, sample_ids_diff_realm = prepare_data(input_data_pt, target_data_df, diff_realm_input_data, diff_realm_target_data)

    print("A sample (i.e., image vector embedding) is of size (i.e., a sample has this many features): ", sample_size)
    nb_classes = len(artists_to_class_numbers.keys())
    print(f"We have a {nb_classes} classes classification problem.")  

    if config.add_verbose:
        # print the <index> image embedding from the dataset as a sanity check
        index = 0
        print(f"CLIP embedding of the {index} image/artwork: {input_data_pt[index]}")
        print(f"Length of CLIP embedding of the {index} image/artwork (i.e., size of the embedding vector used as input to the predictive model): {len(input_data_pt[index])}")

    # run the experiment
    print("Starting the experiment run...")
    run_experiment(dataset_splits, sample_ids_train, sample_ids_test, sample_size, nb_classes, class_numbers_to_artists, X_diff_realm, y_diff_realm, sample_ids_diff_realm, results_writer)

    if config.save_experiment_results:
        # close csv file as nothing more to write for now
        results_csv_file.close()
        
        # load saved data csv file into pandas dataframe
        with open(results_path, 'r') as f:
            results_df = pd.read_csv(f, delimiter="\t")
        
        print(results_df.head())



