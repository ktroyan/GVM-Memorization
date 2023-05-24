import os

# PyTorch
import torch

# Data manipulation
import csv
import pandas as pd
pd.set_option('display.max_columns', None)

# Numerical operations without torch
import numpy as np
import random

# Models, datasets, metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import wandb

# Personal files dependencies
from config import config, CONSTANTS as C

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
                        "input_data_file": config.input_data_path,
                        "target_data_file": config.target_data_path
                        })

        # wandb.login()

        # set model to wandb
        wandb.watch(model, log_freq=100)

    else:
        wandb.init(mode="disabled")

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
