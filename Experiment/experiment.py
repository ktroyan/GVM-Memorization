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

# Personal files dependencies
from data import create_dataloaders, load_dataset, prepare_data
from models import create_model
from utils import seed_everything, show_cuda_gpu_info, setup_wandb, set_header, compute_metrics, compute_topk_accuracy, create_class_rankings, get_rounded_score 
from config import config, CONSTANTS as C


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
    if (config.train_test_diff_realm or config.explicit_test_input_path) and not config.use_train_split_for_test:
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
                                    str(get_rounded_score(1/nb_classes)),
                                    f"{str(get_rounded_score(metrics_train['accuracy']))}",
                                    f"{str(get_rounded_score(metrics_test['accuracy']))}",
                                    f"{str(get_rounded_score(top_k_accuracy_dict_test['1']))}",
                                    f"{str(get_rounded_score(top_k_accuracy_dict_test['3']))}",
                                    f"{str(get_rounded_score(top_k_accuracy_dict_test['5']))}",
                                    f"{str(get_rounded_score(metrics_test['precision']))}",
                                    f"{str(get_rounded_score(metrics_test['recall']))}",
                                    f"{str(get_rounded_score(metrics_test['f1']))}",
                                    config.explicit_train_input_path,
                                    config.explicit_test_input_path,
                                    ])
            
        else:
            results_writer.writerow([train_image_realm,
                                    test_image_realm,
                                    config.artist_category,
                                    config.clip_version.lower(), 
                                    str(config.sd_version).replace('.', '_'),
                                    config.pred_model_name,
                                    str(nb_classes),
                                    str(get_rounded_score(1/nb_classes)),
                                    f"{str(get_rounded_score(metrics_train['accuracy']))}",
                                    f"{str(get_rounded_score(metrics_test['accuracy']))}",
                                    None,
                                    None,
                                    None,
                                    f"{str(get_rounded_score(metrics_test['precision']))}",
                                    f"{str(get_rounded_score(metrics_test['recall']))}",
                                    f"{str(get_rounded_score(metrics_test['f1']))}",
                                    config.explicit_train_input_path,
                                    config.explicit_test_input_path,
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

    # get the config
    config.parse_cmd()

    # display information on cuda/GPU
    show_cuda_gpu_info()

    # seed everything for reproducibility
    seed_everything(config.seed)

    # format the command line arguments given
    config.sd_version = str(config.sd_version).replace('.', '_')
    config.clip_version = config.clip_version.lower()
    config.image_realm = config.image_realm.lower()

    data_folder_path = "./Experiment/Data/"

    # define the data folder path
    if config.artist_category == 'historical':
        data_category_folder_path = data_folder_path + "historical/"
    elif config.artist_category == 'artstation':
        data_category_folder_path = data_folder_path + "artstation/"
    else:
        raise ValueError("The artist category is not valid. Please choose one of the following: historical, artstation")

    if config.image_realm == 'real':
        input_data_path = f'{data_category_folder_path}{config.image_realm}/{config.clip_version}/img_embeddings.pt'
        target_data_path = f'{data_category_folder_path}{config.image_realm}/img_info.csv'

    elif config.image_realm == 'gen' or config.image_realm == 'merged':
        input_data_path = f'{data_category_folder_path}{config.image_realm}/sd_{config.sd_version}/{config.clip_version}/img_embeddings.pt'
        target_data_path = f'{data_category_folder_path}{config.image_realm}/sd_{config.sd_version}/img_info.csv'

    if config.input_data_path is None and config.target_data_path is None:
        config.input_data_path = input_data_path
        config.target_data_path = target_data_path

    elif config.input_data_path is None or config.target_data_path is None:
        raise ValueError("One data path (i.e., input/target) was provided explicitly but not the other (i.e., target/input). Please provide explicitly both input and target data paths.")

    if config.save_experiment_results:
        # open file and create writer to save the data
        results_path = f"{data_folder_path}experiments_results.csv"
        results_csv_file = open(results_path, 'a', encoding="utf-8")
        results_writer = csv.writer(results_csv_file, delimiter="\t", lineterminator="\n")
        results_header = ['train_realm', 'test_realm', 'source', 'encoder', 'sd_version', 'pred_model', 'nb_classes', 'random_guess_accuracy', 'train_accuracy', 'test_accuracy', 'top1', 'top3', 'top5', 'precision', 'recall', 'f1_score', 'explicit_train_input_path', 'explicit_test_input_path']
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
    print(f"The accuracy of a random classifier is: {str(get_rounded_score(1/nb_classes))}%") 

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
        
        # print(results_df.head())



