# PyTorch
from torch import nn, optim
from torch.nn import functional as F

# Data manipulation
import pandas as pd
pd.set_option('display.max_columns', None)


# Models, datasets, metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from scipy.stats import uniform

# Tracking of experiments
from tqdm import trange
import wandb

# Personal files dependencies
from data import create_dataloaders
from evaluation import predict
import utils as U 
from config import config, CONSTANTS as C

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
        U.setup_wandb(model, learning_rate, optimizer, criterion)

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
        U.setup_wandb(model, learning_rate, optimizer, criterion)

        trained_model = train(model, dataset_splits['X_train'], dataset_splits['y_train'], dataset_splits['X_test'], dataset_splits['y_test'], dataloaders['train_loader'], dataloaders['val_loader'], criterion, optimizer)
    
    else:
        raise ValueError(f"Before starting training, got an invalid model name: {config.pred_model_name}")

    return trained_model

