# PyTorch
import torch
from torch.nn import functional as F

# Data manipulation
import pandas as pd
pd.set_option('display.max_columns', None)

# Numerical operations without torch
import numpy as np

# Personal files dependencies
from data import create_dataloaders
import utils as U
from config import config, CONSTANTS as C


def predict(model, X, y, X_val=None, y_val=None, train_loader=None, val_loader=None):

    if config.pred_model_name == "logistic_regression":
        y_pred = model.predict(X)
        
        metrics = U.compute_metrics(y, y_pred)
        print(f"Accuracy: {metrics['accuracy']*100:.3}%")  # for an equivalent to predict and compute accuracy metric: model.score(X, y)

        probabilities = model.predict_proba(X)
        probabilities = torch.Tensor(probabilities)

    elif config.pred_model_name == "xgboost":
        y_pred = model.predict(X)
        metrics = U.compute_metrics(y, y_pred)
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

            metrics = U.compute_metrics(torch.Tensor(np.asarray(y)), torch.Tensor(np.asarray(y_pred)))
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
    top_k_accuracy_dict, top_k_classes_dict = U.compute_topk_accuracy(sorted_probabilities, class_rankings_indices, y, k_list)
    
    for k in k_list:
        print(f"Top {k} accuracy is {top_k_accuracy_dict[str(k)]*100:.3}%")
        
    y = torch.Tensor(np.asarray(y))
    y_pred = torch.Tensor(np.asarray(y_pred))

    return y, y_pred, metrics, top_k_classes_dict, top_k_accuracy_dict, class_rankings_indices, sorted_probabilities

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
        U.create_class_rankings(train_ranking_file_path, y_pred_train, sample_ids_train, class_ranking_indices_train, class_numbers_to_artists, probabilities_train)

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
        U.create_class_rankings(test_ranking_file_path, y_pred_test, sample_ids_test, class_ranking_indices_test, class_numbers_to_artists, probabilities_test)

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
                                    str(U.get_rounded_score(1/nb_classes)),
                                    f"{str(U.get_rounded_score(metrics_train['accuracy']))}",
                                    f"{str(U.get_rounded_score(metrics_test['accuracy']))}",
                                    f"{str(U.get_rounded_score(top_k_accuracy_dict_test['1']))}",
                                    f"{str(U.get_rounded_score(top_k_accuracy_dict_test['3']))}",
                                    f"{str(U.get_rounded_score(top_k_accuracy_dict_test['5']))}",
                                    f"{str(U.get_rounded_score(metrics_test['precision']))}",
                                    f"{str(U.get_rounded_score(metrics_test['recall']))}",
                                    f"{str(U.get_rounded_score(metrics_test['f1']))}",
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
                                    str(U.get_rounded_score(1/nb_classes)),
                                    f"{str(U.get_rounded_score(metrics_train['accuracy']))}",
                                    f"{str(U.get_rounded_score(metrics_test['accuracy']))}",
                                    None,
                                    None,
                                    None,
                                    f"{str(U.get_rounded_score(metrics_test['precision']))}",
                                    f"{str(U.get_rounded_score(metrics_test['recall']))}",
                                    f"{str(U.get_rounded_score(metrics_test['f1']))}",
                                    config.explicit_train_input_path,
                                    config.explicit_test_input_path,
                                    ])
    
    print("\n")