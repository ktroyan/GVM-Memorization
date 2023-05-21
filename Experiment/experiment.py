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
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt

# Tracking of experiments
from tqdm import trange, tqdm
import wandb


def seed_everything(SEED=121997):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False   # True for deterministic
    torch.backends.cudnn.benchmark = True  # False for deterministic

def get_cuda_gpu_info():
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
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():     # for MacOS
        device = 'mps'
    else:
        print("Cuda is not available.")
        device = torch.device('cpu')

    return device

def load_dataset(dataset_path):
    dataset = torch.load(dataset_path)  # load the .pt file as a Tensor containing the generated images embeddings
    print(f"Dataset loaded.")
    print(f"Dataset of generated images embeddings contain {len(dataset)} samples.")
    return dataset

def set_header(data_writer, data_header):
    # write header of the csv file if there is no header yet
    if os.stat(new_dataset_path).st_size == 0:
        data_file_has_header = False
        data_writer.writerow(data_header)

def setup_wandb(model, model_name, N_EPOCHS, BATCH_SIZE, LR, OPTIMIZER, CRITERION, DEVICE, use_wandb):

    if use_wandb:

        wandb.init(project="GVM-project",
                config={
                        "model": model_name,
                        "epochs": N_EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LR,
                        "optimizer": OPTIMIZER,
                        "criterion": CRITERION,
                        "device": DEVICE,
                        "dataset": "TBD"
                        })

        # wandb.login()

        # set model to wandb
        wandb.watch(model, log_freq=100)

    else:
        wandb.init(mode="disabled")

def create_NN(model_name, nb_features, nb_classes):

    # print("Number of features: ", nb_features)
    # print("Number of classes: ", nb_classes)

    if model_name == "linear_nn":
        
        class NNLogisticRegression(torch.nn.Module):    
            def __init__(self, nb_features, nb_classes):
                super(NNLogisticRegression, self).__init__()
                self.linear = torch.nn.Linear(nb_features, nb_classes)

            def forward(self, x):
                y_pred = self.linear(x)
                return y_pred
        
        model = NNLogisticRegression(nb_features, nb_classes)
        
    elif model_name == "nn":
        
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

    input_dims = (1, nb_features)   # without batch size
    summary(model.to(DEVICE), input_dims)
    return model

def create_model(model_name, nb_features=None, nb_classes=None):

    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=10000, solver='lbfgs', penalty='l2', n_jobs=-1, verbose=1)

    elif model_name == "xgboost":

        model = xgb.XGBClassifier(objective='multi:softmax', num_class=nb_classes, n_jobs=-1, tree_method="gpu_hist", verbosity=1)
        
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
    
    elif model_name == "linear_nn":
        print("Number of features: ", nb_features)
        print("Number of classes: ", nb_classes)

        model = create_NN(model_name, nb_features, nb_classes)
        model = model.to(DEVICE)

    elif model_name == "nn":
        print("Number of features: ", nb_features)
        print("Number of classes: ", nb_classes)

        model = create_NN(model_name, nb_features, nb_classes)
        model = model.to(DEVICE)

    return model

def artists_to_class_numbers_bimap(gen_images_df):

    artists = gen_images_df['artist'].copy()
    unique_artists = artists.unique()

    artists_to_class_numbers = {artist: i for i, artist in enumerate(unique_artists)}
    class_numbers_to_artists = {str(i): artist for i, artist in enumerate(unique_artists)}

    # create a column with the artist class numbers in the dataframe
    gen_images_df['artist_class_number'] = gen_images_df['artist'].map(artists_to_class_numbers)

    return gen_images_df['artist_class_number'].copy(), artists_to_class_numbers, class_numbers_to_artists

def get_artists_from_class_numbers(predictions, class_numbers_to_artists):
    pred_artists = [class_numbers_to_artists[str(int(pred.item()))] for pred in predictions]
    return pred_artists

def get_artists_ranking(ranked_artist_classes, class_numbers_to_artists):
    return [[class_numbers_to_artists[str(artist_class.item())] for artist_class in sample] for sample in ranked_artist_classes]

def prepare_data(dataset, gen_images_df):

    X = np.array(dataset)

    y, artists_to_class_numbers, class_numbers_to_artists = artists_to_class_numbers_bimap(gen_images_df)

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=121997)

    print("Number of samples in X_train: ", len(X_train))
    print("Number of samples in y_train ", len(y_train))
    print("Number of samples in X_test: ", len(X_test))
    print("Number of samples in y_test: ", len(y_test))

    sample_size = X_train[1].shape[0]   # len(X_train[0]); number of features of each sample (i.e., size of the embedding vector of each image)

    print("Splitted the dataset into (X,y) pairs of train and test sets.")

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
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # TODO: set True
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print("Number of batches in train loader: ", len(train_loader))
    # print("Number of batches in test loader: ", len(test_loader))

    dataloaders = {'train_loader': train_loader, 'test_loader': test_loader}

    return dataloaders, batch_size

def compute_topk_accuracy(probabilities, gt_labels, k, method = "method1"):

    # get the ranking of classes/artists
    sorted_classes, class_ranking_indices = torch.sort(probabilities, descending=True)  # if element at i is j, then class j is the top-i prediction

    if method == "method1":
        top_k_classes = class_ranking_indices[:, :k]
        correct_predictions = torch.sum(top_k_classes == torch.Tensor(gt_labels).unsqueeze(1), dim=1)   # creare boolean Tensor and sume (True is 1 and False is 0)
        topk_accuracy = torch.mean(correct_predictions.float())
    
    elif method == "method2":
        top_probs, top_k_classes = probabilities.cpu().topk(k, dim=-1)
        topk_correct = 0
        for i in range(k):
            top_one_labels = top_k_classes[:, i]
            correct = torch.sum(torch.Tensor(top_one_labels) == torch.Tensor(gt_labels))
            topk_correct += correct
        topk_accuracy = (topk_correct / gt_labels.shape[0]).item()

    return topk_accuracy, top_k_classes, class_ranking_indices

def predict(model, model_name, X, y, X_test=None, y_test=None, train_loader=None, test_loader=None):

    if model_name == "logistic_regression":
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        # accuracy = model.score(X, y)    # same as accuracy_score(y, y_pred)
        print(f"Accuracy: {accuracy*100:.3}%")

    elif model_name == "xgboost":
        print("Predicting with XGBoost model...")

        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy: {accuracy*100:.3}%")

    elif model_name == "linear_nn" or model_name == "nn":
        k = 5
        with torch.no_grad():
            logits = model(torch.Tensor(np.asarray(X)).to(DEVICE))

            # TODO: softmax or log_softmax?
            probabilities = F.log_softmax(logits.cpu(), dim=1)    # dim=1 to compute along the class dimension. 
            _, y_pred = torch.max(probabilities, 1)
            accuracy = (torch.Tensor(np.asarray(y_pred)) == torch.Tensor(np.asarray(y))).float().mean()
            accuracy = accuracy.item()
            print(f"Accuracy: {accuracy*100:.3}%")
            
            topk_accuracy, top_k_pred_classes, class_ranking_indices = compute_topk_accuracy(probabilities, y, k=k)
            print(f"Top {k} accuracy is {topk_accuracy*100:.3}%")
            
    y = torch.Tensor(np.asarray(y))
    y_pred = torch.Tensor(np.asarray(y_pred))

    if model_name in ["linear_nn", "nn"]:
        return y, y_pred, accuracy, top_k_pred_classes, topk_accuracy, class_ranking_indices
    else:
        return y, y_pred, accuracy, None, None, None

def train(model, model_name, X_train, X_test, y_train, y_test, train_loader=None, test_loader=None, n_epochs=None, learning_rate=None, criterion=None, optimizer=None):
    
    if model_name == "logistic_regression":
        # Train the model
        model.fit(X_train, y_train)

    elif model_name == "xgboost":
        # Train the model
        model = model.fit(X_train, y_train)

    elif model_name == "linear_nn" or model_name == "nn":

        print_at_n_epochs = 50

        use_tqdm = True
        if use_tqdm:
            t = trange(n_epochs, desc='Loss', leave=True)
        else:
            t = range(n_epochs)

        # Train the model
        for epoch in t:  # number of epochs
            for X, y in train_loader:

                # Set data to GPU if possible
                X = X.to(DEVICE)
                y = y.to(DEVICE)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                logits = model(X.float())

                probabilities = F.log_softmax(logits, dim=1)    # TODO: softmax or log_softmax?

                # Compute the loss
                loss = criterion(probabilities, y)     # no need to apply softmax if using cross-entropy loss

                # Backward pass
                loss.backward()

                # Optimization step (update weights/parameters)
                optimizer.step()

            # print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
            t.set_description(f"loss: {loss.item()}")
            wandb.log({f"Epoch loss: ": loss.item()})

            if (epoch+1) % print_at_n_epochs == 0:
                print("Predicting on train set at epoch ", epoch+1)
                _, _, accuracy, top_k_pred_classes, topk_accuracy, class_ranking_indices = predict(model, model_name, X_train, y_train, train_loader=None, test_loader=None)
                wandb.log({f"Every {print_at_n_epochs} epochs train accuracy: ": accuracy*100})
                wandb.log({f"Every {print_at_n_epochs} epochs train TOP-K accuracy: ": topk_accuracy*100})
        
    print("\n")
    # print("Predicting on train set...")
    # y, y_pred, accuracy, top_k_pred_classes, topk_accuracy, class_ranking_indices = predict(model, model_name, X_train, y_train, train_loader=None, test_loader=None)   # Sanity check: evaluate the model on the training set
    # print("Predicting on test set...")
    # y, y_pred, accuracy, top_k_pred_classes, topk_accuracy, class_ranking_indices = predict(model, model_name, X_test, y_test, train_loader=None, test_loader=None)    # Actual evaluation: evaluate the model on the test set

    return model

def run_training(model, model_name, dataset_splits, use_wandb):

    X_train = np.asarray(dataset_splits['X_train'])
    X_test = np.asarray(dataset_splits['X_test'])
    y_train = np.asarray(dataset_splits['y_train'])
    y_test = np.asarray(dataset_splits['y_test'])

    if model_name == 'logistic_regression':
        trained_model = train(model, model_name, X_train, X_test, y_train, y_test)

    elif model_name == 'xgboost':
        trained_model = train(model, model_name, X_train, X_test, y_train, y_test)

    elif model_name == 'linear_nn':
        dataloaders, batch_size = create_dataloaders(dataset_splits)
        train_loader = dataloaders['train_loader']
        test_loader = dataloaders['test_loader']
        
        nb_features = X_train.shape[1]  # length of the image embeddings

        n_epochs = 400
        learning_rate = 0.0001 # use 0.1 for SGD and 0.0001 for Adam

        # Define the loss function and the optimizer for the NN model
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # setup WandB
        setup_wandb(model, model_name, n_epochs, batch_size, learning_rate, optimizer, criterion, DEVICE, use_wandb)

        trained_model = train(model, model_name, X_train, X_test, y_train, y_test, train_loader, test_loader, n_epochs, learning_rate, criterion, optimizer)


    elif model_name == 'nn':
        dataloaders, batch_size = create_dataloaders(dataset_splits)
        train_loader = dataloaders['train_loader']
        test_loader = dataloaders['test_loader']

        nb_features = X_train.shape[1]  # length of the image embeddings

        n_epochs = 400
        learning_rate = 0.1 # use 0.1 for SGD and 0.0001 for Adam

        # Define the loss function and the optimizer
        criterion = nn.NLLLoss()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # setup WandB
        setup_wandb(model, model_name, n_epochs, batch_size, learning_rate, optimizer, criterion, DEVICE, use_wandb)

        trained_model = train(model, model_name, X_train, X_test, y_train, y_test, train_loader, test_loader, n_epochs, learning_rate, criterion, optimizer)
        

    return trained_model

def run_evaluation(trained_pred_model, pred_model_name, dataset_splits, class_numbers_to_artists):

    if pred_model_name in ['linear_nn', 'nn']:
        # create the dataloaders for NN model training
        dataloaders, batch_size = create_dataloaders(dataset_splits)
        train_loader = dataloaders['train_loader']
        test_loader = dataloaders['test_loader']
    else:
        train_loader = None
        test_loader = None

    # predict on train set
    print("Predict on train set...")
    y_train, y_pred_train, accuracy_train, top_k_pred_classes, topk_accuracy, class_ranking_indices = predict(trained_pred_model, pred_model_name, np.asarray(dataset_splits['X_train']), np.asarray(dataset_splits['y_train']), X_test=None, y_test=None, train_loader=train_loader, test_loader=test_loader)

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
    y_test, y_pred_test, accuracy_test, top_k_pred_classes, topk_accuracy, class_ranking_indices = predict(trained_pred_model, pred_model_name, np.asarray(dataset_splits['X_test']), np.asarray(dataset_splits['y_test']), X_test=None, y_test=None, train_loader=train_loader, test_loader=test_loader)
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

def run_experiment(pred_model_name, dataset_splits, sample_size, nb_classes, class_numbers_to_artists, eval_mode=False, use_wandb=False):
    if pred_model_name in ["logistic_regression", "xgboost", "nn", "linear_nn"]:
        
        # create the predictive model
        pred_model = create_model(pred_model_name, sample_size, nb_classes)
        
        # train the predictive model
        trained_pred_model = run_training(pred_model, pred_model_name, dataset_splits, use_wandb)

        if eval_mode:
            # evaluate the predictive model
            run_evaluation(trained_pred_model, pred_model_name, dataset_splits, class_numbers_to_artists)

    else:
        print("The predictive model name is not valid. Please choose one of the following: logistic_regression, xgboost, nn, linear_nn")

if __name__ == '__main__':

    seed_everything()
    
    DEVICE = get_cuda_gpu_info()

    command_line_parser = argparse.ArgumentParser()

    command_line_parser.add_argument("--sd_version", type=float, default=1.5, help="Stable Diffusion version: 1.5 or 2.1")
    command_line_parser.add_argument("--clip_version", type=str, default="OpenAI", help="CLIP version: OpenAI or Laion2b")
    command_line_parser.add_argument("--pred_model_name", type=str, default="nn", help="Name of the predictive model: logistic_regression, xgboost, linear_nn, nn")
    command_line_parser.add_argument("--eval_mode", action='store_true', default=False, help="Whether to explicitly perform a model evaluation or not during the experiment run.")
    command_line_parser.add_argument("--use_wandb", action='store_true', default=False, help="Whether to explicitly perform a model evaluation or not during the experiment run.")

    args = command_line_parser.parse_args()

    # format the command line arguments given
    sd_version = str(args.sd_version).replace('.', '_')
    clip_version = args.clip_version.lower()
    pred_model_name = args.pred_model_name
    eval_mode = args.eval_mode
    use_wandb = args.use_wandb

    # define the data folder path
    data_folder_path = "./Experiment/Data/"

    # define the path to embeddings dataset
    if clip_version == "openai":
        dataset_path = f'{data_folder_path}sd_{sd_version}_{clip_version}_ViT-B-32.pt'
    elif clip_version == "laion2b":
        dataset_path = f'{data_folder_path}sd_{sd_version}_{clip_version}_s34b_b79k_ViT-B-32.pt'

    # load saved data csv file into pandas dataframe
    with open(f'{data_folder_path}gen_images_{sd_version}.csv', 'rb') as f:
        gen_images_df = pd.read_csv(f, delimiter=",", encoding="utf-8")

    # open file and create writer to save the data
    new_dataset_path = f"{data_folder_path}new_data.csv"
    data_csv_file = open(new_dataset_path, 'a', encoding="utf-8")
    data_writer = csv.writer(data_csv_file, delimiter="\t", lineterminator="\n")
    data_header = ['image_id', 'embedding', 'artist', 'sd_version', 'clip_version']
    set_header(data_writer, data_header)

    # load the dataset
    input_data = load_dataset(dataset_path)

    # print the <index> image embedding from the dataset as a sanity check
    index = 0
    # print(f"CLIP embedding of the {index} generated image: {input_data[index]}")
    # print(f"Length of CLIP embedding of the {index} generated image (i.e., size of the embedding vector used as input to the predictive model): {len(input_data[index])}")

    # dataset_splits is a dict with the splitted dataset subsets: X_train, X_test, y_train, y_test
    sample_size, dataset_splits, artists_to_class_numbers, class_numbers_to_artists = prepare_data(input_data, gen_images_df)

    print("A sample (i.e., image vector embedding) is of size (i.e., a sample has this many features): ", sample_size)

    nb_classes = len(artists_to_class_numbers.keys())
    print(f"We have a {nb_classes} classes classification problem.")  

    # run the experiment
    run_experiment(pred_model_name, dataset_splits, sample_size, nb_classes, class_numbers_to_artists, eval_mode=eval_mode, use_wandb=use_wandb)

    # close csv file as nothing more to write for now
    data_csv_file.close()
    
    # load saved data csv file into pandas dataframe
    with open(new_dataset_path, 'r') as f:
        new_dataset_df = pd.read_csv(f, delimiter="\t")



