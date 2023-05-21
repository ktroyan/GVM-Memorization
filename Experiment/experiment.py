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
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error
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

def load_dataset(dataset_path):

    # load input data (i.e., image embeddings) from Tensor in .pt file
    try:
        input_data_pt = torch.load(dataset_path)  # load the .pt file as a Tensor containing the generated images embeddings
        print(f"Dataset loaded.")
        print(f"Dataset of generated images' embeddings contain {len(input_data_pt)} samples.")
    except Exception as e:
        print(f"Error when loading the images' embeddings: {e}")

    # load saved target/label data (and additional info) from csv file into pandas dataframe
    if image_source == "gen":
        with open(f'{data_folder_path}{image_source}_images_{sd_version}.csv', 'rb') as f:
            target_data_df = pd.read_csv(f, delimiter=",", encoding="utf-8")
    elif image_source == "real":
        with open(f'{data_folder_path}{image_source}_images.csv', 'rb') as f:
            target_data_df = pd.read_csv(f, delimiter=",", encoding="utf-8")
    else:
        print("To load the saved data, the image source is not valid. Please choose one of the following: 'real' or 'gen'")

    return input_data_pt, target_data_df

def set_header(data_writer, data_header):
    # write header of the csv file if there is no header yet
    if os.stat(new_dataset_path).st_size == 0:
        data_file_has_header = False
        data_writer.writerow(data_header)

def setup_wandb(model, LR, OPTIMIZER, CRITERION):

    if config.use_wandb:

        wandb.init(project="GVM-project",
                config={
                        "model": config.pred_model_name,
                        "epochs": config.n_epochs,
                        # "batch_size": config.batch_size,  # uncomment if dataloaders are used
                        "learning_rate": LR,
                        "optimizer": OPTIMIZER,
                        "criterion": CRITERION,
                        "device": C.DEVICE,
                        "top-k": "k=" + str(config.topk),
                        "dataset": dataset_file
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

def get_dataset_splits(X, y):

    if config.balance_classes_across_sets:
        X_train, X_test, y_train, y_test = [], [], [], []

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

            X_test.extend(X[k + n_train:k + n_train + n_test])
            y_test.extend(y[k + n_train:k + n_train + n_test])

        # map to numpy arrays since we X_train/X_test is a Python list
        X_train = list(map(np.asarray, X_train))
        X_test = list(map(np.asarray, X_test))

        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        if config.rand_shuffle_data:
            # randomly shuffle the numpy arrays X_train and y_train in the same way
            indices_train = np.random.permutation(len(X_train)) # generate a random permutation of indices
            X_train = X_train[indices_train]
            y_train = y_train[indices_train]

            # randomly shuffle the numpy arrays X_test and y_test in the same way
            indices_test = np.random.permutation(len(X_test)) # generate a random permutation of indices
            X_test = X_test[indices_test]
            y_test = y_test[indices_test]

    else:
        # split the data randomly (and without class balancing across the splits/sets) into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            train_size=config.train_test_ratio, 
                                                            test_size=1-config.train_test_ratio, 
                                                            shuffle=config.rand_shuffle_data, 
                                                            random_state=config.seed)

        # convert the train and test sets back to numpy arrays
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

    if config.verify_class_balance:
        class_counts = y.value_counts()[0]  # get the number of samples per class (assuming same number for all classes!)
        verify_class_balance(y_train, y_test, class_counts)

    return X_train, X_test, y_train, y_test

def prepare_data(input_data_pt, target_data_df):

    # X = np.array(input_data_pt)
    X = input_data_pt
    target_data_df, artists_to_class_numbers, class_numbers_to_artists = artists_to_class_numbers_bimap(target_data_df)

    y = target_data_df['artist_class_number'].copy()

    X_train, X_test, y_train, y_test = get_dataset_splits(X, y)

    print("Number of samples in X_train: ", len(X_train))
    print("Number of samples in y_train ", len(y_train))
    print("Number of samples in X_test: ", len(X_test))
    print("Number of samples in y_test: ", len(y_test))

    sample_size = X_train[1].shape[0]   # len(X_train[0]); number of features of each sample (i.e., size of the embedding vector of each image)

    print("Split the dataset into (X,y) pairs of train and test sets.")

    dataset_splits = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    return sample_size, dataset_splits, artists_to_class_numbers, class_numbers_to_artists

def create_dataloaders(dataset_splits):

    class ArtDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]

    X_train = dataset_splits['X_train']
    X_test = dataset_splits['X_test']
    y_train = np.asarray(dataset_splits['y_train'])
    y_test = np.asarray(dataset_splits['y_test'])

    # Convert the data to PyTorch tensors
    X_train, X_test = map(torch.LongTensor, (X_train, X_test))
    y_train, y_test = map(torch.LongTensor, (y_train, y_test))

    train_dataset = ArtDataset(X_train, y_train)
    test_dataset = ArtDataset(X_test, y_test)

    # Create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # print("Number of batches in train loader: ", len(train_loader))
    # print("Number of batches in test loader: ", len(test_loader))

    dataloaders = {'train_loader': train_loader, 'test_loader': test_loader}

    return dataloaders

def compute_topk_accuracy(probabilities, gt_labels, method = "method1"):

    # get the ranking of classes/artists
    sorted_classes, class_ranking_indices = torch.sort(probabilities, descending=True)  # if element at i is j, then class j is the top-i prediction

    if method == "method1":
        top_k_classes = class_ranking_indices[:, :config.topk]
        correct_predictions = torch.sum(top_k_classes == torch.Tensor(gt_labels).unsqueeze(1), dim=1)   # creare boolean Tensor and sume (True is 1 and False is 0)
        topk_accuracy = torch.mean(correct_predictions.float())
    
    elif method == "method2":
        top_probs, top_k_classes = probabilities.cpu().topk(config.topk, dim=-1)
        topk_correct = 0
        for i in range(config.topk):
            top_one_labels = top_k_classes[:, i]
            correct = torch.sum(torch.Tensor(top_one_labels) == torch.Tensor(gt_labels))
            topk_correct += correct
        topk_accuracy = (topk_correct / gt_labels.shape[0]).item()

    return topk_accuracy, top_k_classes, class_ranking_indices

def predict(model, X, y, X_test=None, y_test=None, train_loader=None, test_loader=None):

    if config.pred_model_name == "logistic_regression":
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        # accuracy = model.score(X, y)    # same as accuracy_score(y, y_pred)
        print(f"Accuracy: {accuracy*100:.3}%")

        probabilities = model.predict_proba(X)
        probabilities = torch.Tensor(probabilities)

    elif config.pred_model_name == "xgboost":
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy: {accuracy*100:.3}%")

        probabilities = model.predict_proba(X)
        probabilities = torch.Tensor(probabilities)

    elif config.pred_model_name in ["linear_nn", "nn"]:
        with torch.no_grad():
            logits = model(torch.Tensor(np.asarray(X)).to(C.DEVICE))
            probabilities = F.log_softmax(logits.cpu(), dim=1)    # dim=1 to compute along the class dimension; TODO: softmax or log_softmax? 
            _, y_pred = torch.max(probabilities, 1)
            accuracy = (torch.Tensor(np.asarray(y_pred)) == torch.Tensor(np.asarray(y))).float().mean()
            accuracy = accuracy.item()
            print(f"Accuracy: {accuracy*100:.3}%")
            
    else:
        raise ValueError(f"Before predicting, got an invalid model name: {config.pred_model_name}")

    # Get the top-k predicted classes and compute top-k accuracy
    topk_accuracy, top_k_pred_classes, class_ranking_indices = compute_topk_accuracy(probabilities, y)
    print(f"Top {config.topk} accuracy is {topk_accuracy*100:.3}%")
        
    y = torch.Tensor(np.asarray(y))
    y_pred = torch.Tensor(np.asarray(y_pred))

    return y, y_pred, accuracy, top_k_pred_classes, topk_accuracy, class_ranking_indices

def train(model, X_train, X_test, y_train, y_test, train_loader=None, test_loader=None, criterion=None, optimizer=None):
    
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

                probabilities = F.log_softmax(logits, dim=1)    # TODO: softmax or log_softmax?

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
                _, _, accuracy, top_k_pred_classes, topk_accuracy, class_ranking_indices = predict(model, X_train, y_train, train_loader=None, test_loader=None)   # Sanity check: evaluate the model on the training set
                wandb.log({f"Every {config.print_at_n_epochs} epochs train accuracy: ": accuracy*100})
                wandb.log({f"Every {config.print_at_n_epochs} epochs train TOP-{config.topk} accuracy: ": topk_accuracy*100})
        
    else:
        raise ValueError(f"Before training, got an invalid model name: {config.pred_model_name}")
    
    print("\n")
    # print("Predicting on train set...")
    # y, y_pred, accuracy, top_k_pred_classes, topk_accuracy, class_ranking_indices = predict(model, pred_model_name, X_train, y_train, train_loader=None, test_loader=None, k=topk)   # Sanity check: evaluate the model on the training set
    # print("Predicting on test set...")
    # y, y_pred, accuracy, top_k_pred_classes, topk_accuracy, class_ranking_indices = predict(model, pred_model_name, X_test, y_test, train_loader=None, test_loader=None, k=topk)    # Actual evaluation: evaluate the model on the test set

    return model

def run_training(model, dataset_splits):

    X_train = dataset_splits['X_train']
    X_test = dataset_splits['X_test']
    y_train = dataset_splits['y_train']
    y_test = dataset_splits['y_test']

    if config.pred_model_name == 'logistic_regression':
        trained_model = train(model, X_train, X_test, y_train, y_test)

    elif config.pred_model_name == 'xgboost':
        trained_model = train(model, X_train, X_test, y_train, y_test)

    elif config.pred_model_name == 'linear_nn':
        dataloaders = create_dataloaders(dataset_splits)
        train_loader = dataloaders['train_loader']
        test_loader = dataloaders['test_loader']
        
        nb_features = X_train.shape[1]  # length of the image embeddings

        learning_rate = 0.1 # use 0.1 for SGD and 0.0001 for Adam

        # Define the loss function and the optimizer for the NN model
        criterion = nn.NLLLoss()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # setup WandB
        setup_wandb(model, learning_rate, optimizer, criterion)

        trained_model = train(model, X_train, X_test, y_train, y_test, train_loader, test_loader, criterion, optimizer)


    elif config.pred_model_name == 'nn':
        dataloaders = create_dataloaders(dataset_splits)
        train_loader = dataloaders['train_loader']
        test_loader = dataloaders['test_loader']

        nb_features = X_train.shape[1]  # length of the image embeddings

        learning_rate = 0.1 # use 0.1 for SGD and 0.0001 for Adam

        # Define the loss function and the optimizer
        criterion = nn.NLLLoss()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # setup WandB
        setup_wandb(model, learning_rate, optimizer, criterion)

        trained_model = train(model, X_train, X_test, y_train, y_test, train_loader, test_loader, criterion, optimizer)
    
    else:
        raise ValueError(f"Before starting training, got an invalid model name: {config.pred_model_name}")

    return trained_model

def run_evaluation(trained_pred_model, dataset_splits, class_numbers_to_artists):

    if config.pred_model_name in ['linear_nn', 'nn']:
        # create the dataloaders for NN model training
        dataloaders = create_dataloaders(dataset_splits)
        train_loader = dataloaders['train_loader']
        test_loader = dataloaders['test_loader']
    elif config.pred_model_name in ['logistic_regression', 'xgboost']:
        train_loader = None
        test_loader = None
    else:
        raise ValueError(f"Before starting evaluation, got an invalid model name: {config.pred_model_name}")
    
    # predict on train set
    print("Predict on train set...")
    y_train, y_pred_train, accuracy_train, top_k_pred_classes, topk_accuracy, class_ranking_indices = predict(trained_pred_model, np.asarray(dataset_splits['X_train']), np.asarray(dataset_splits['y_train']), X_test=None, y_test=None, train_loader=train_loader, test_loader=test_loader)

    pred_artists = get_artists_from_class_numbers(y_pred_train, class_numbers_to_artists)   # convert class numbers predictions to explicit artist classes (i.e., names of the artists)
    # print("Train predictions (with explicit artist classes): ", pred_artists)      

    if class_ranking_indices is not None:
        print("For train set, saving in file ranked artists for all samples. ")
        rankings_train = get_artists_ranking(class_ranking_indices, class_numbers_to_artists)

        # save the artists ranking for the train set
        with open('./Experiment/Data/train_set_artists_ranking.csv', 'w', encoding="utf-8") as f:
            data_writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            data_writer.writerow(['pred_artist', 'artists_ranking'])
            for pred_artist, sample_ranking in zip(pred_artists, rankings_train):
                data_writer.writerow([pred_artist, ";".join(sample_ranking)])    

    # predict on test set
    print("Predict on test set...")
    y_test, y_pred_test, accuracy_test, top_k_pred_classes, topk_accuracy, class_ranking_indices = predict(trained_pred_model, np.asarray(dataset_splits['X_test']), np.asarray(dataset_splits['y_test']), X_test=None, y_test=None, train_loader=train_loader, test_loader=test_loader)
    pred_artists = get_artists_from_class_numbers(y_pred_test, class_numbers_to_artists)   # convert class numbers predictions to explicit artist classes (i.e., names of the artists)
    # print("Test predictions (with explicit artist classes): ", pred_artists)

    if class_ranking_indices is not None:
        print("For test set, saving in file ranked artists for all samples. ")
        rankings_test = get_artists_ranking(class_ranking_indices, class_numbers_to_artists)

        # save the artists ranking for the train set
        with open('./Experiment/Data/test_set_artists_ranking.csv', 'w', encoding="utf-8") as f:
            data_writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            data_writer.writerow(['pred_artist', 'artists_ranking'])
            for pred_artist, sample_ranking in zip(pred_artists, rankings_test):
                data_writer.writerow([pred_artist, ";".join(sample_ranking)])

    print("\n")  

def run_experiment(dataset_splits, sample_size, nb_classes, class_numbers_to_artists):
    if config.pred_model_name in ["logistic_regression", "xgboost", "nn", "linear_nn"]:
        
        # create the predictive model
        pred_model = create_model(sample_size, nb_classes)
        
        # train the predictive model
        trained_pred_model = run_training(pred_model, dataset_splits)

        if config.eval_mode:
            # evaluate the predictive model
            run_evaluation(trained_pred_model, dataset_splits, class_numbers_to_artists)

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
    sd_version = str(config.sd_version).replace('.', '_')
    clip_version = config.clip_version.lower()
    image_source = config.image_source.lower()

    # define the data folder path
    data_folder_path = "./Experiment/Data/"

    # define the path to embeddings dataset
    if image_source == "gen":
        if clip_version == "openai":
            dataset_file = f'sd_{sd_version}_{clip_version}_ViT-B-32.pt'
        elif clip_version == "laion2b":
            dataset_file = f'sd_{sd_version}_{clip_version}_s34b_b79k_ViT-B-32.pt'
        else:
            print("The CLIP version is not valid. Please choose one of the following: 'OpenAI' or 'Laion2b'")

    elif image_source == "real":
        if clip_version == "openai":
            dataset_file = f'{image_source}_{clip_version}_ViT-B-32.pt'
        elif clip_version == "laion2b":
            dataset_file = f'{image_source}_{clip_version}_s34b_b79k_ViT-B-32.pt'
        else:
            print("The CLIP version is not valid. Please choose one of the following: 'OpenAI' or 'Laion2b'")
    else:
        print("To define the name of the dataset file, the image source is not valid. Please choose one of the following: 'real' or 'gen'")

    model_input_data_path = f'{data_folder_path}{dataset_file}'

    # open file and create writer to save the data
    new_dataset_path = f"{data_folder_path}new_data.csv"
    data_csv_file = open(new_dataset_path, 'a', encoding="utf-8")
    data_writer = csv.writer(data_csv_file, delimiter="\t", lineterminator="\n")
    data_header = ['image_id', 'embedding', 'artist', 'sd_version', 'clip_version']
    set_header(data_writer, data_header)

    # load the dataset
    input_data_pt, target_data_df = load_dataset(model_input_data_path)

    # print the <index> image embedding from the dataset as a sanity check
    index = 0
    # print(f"CLIP embedding of the {index} image/artwork: {input_data[index]}")
    # print(f"Length of CLIP embedding of the {index} image/artwork (i.e., size of the embedding vector used as input to the predictive model): {len(input_data[index])}")

    # dataset_splits is a dict with the splitted dataset subsets: X_train, X_test, y_train, y_test
    sample_size, dataset_splits, artists_to_class_numbers, class_numbers_to_artists = prepare_data(input_data_pt, target_data_df)

    print("A sample (i.e., image vector embedding) is of size (i.e., a sample has this many features): ", sample_size)

    nb_classes = len(artists_to_class_numbers.keys())
    print(f"We have a {nb_classes} classes classification problem.")  

    # run the experiment
    print("Starting the experiment run...")
    run_experiment(dataset_splits, sample_size, nb_classes, class_numbers_to_artists)

    # close csv file as nothing more to write for now
    data_csv_file.close()
    
    # load saved data csv file into pandas dataframe
    with open(new_dataset_path, 'r') as f:
        new_dataset_df = pd.read_csv(f, delimiter="\t")



