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
from config import config, CONSTANTS as C

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

