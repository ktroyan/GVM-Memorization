# Data manipulation
import csv
import pandas as pd
pd.set_option('display.max_columns', None)

# Personal files dependencies
from data import load_dataset, prepare_data
from models import create_model
from training import run_training
from evaluation import run_evaluation
import utils as U 
from config import config, CONSTANTS as C


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
    U.show_cuda_gpu_info()

    # seed everything for reproducibility
    U.seed_everything(config.seed)

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
        results_file_has_header = U.set_header(results_path, results_writer, results_header)
    else:
        results_writer = None

    # load the dataset
    input_data_pt, target_data_df, diff_realm_input_data, diff_realm_target_data = load_dataset(input_data_path, target_data_path)
    
    # dataset_splits is a dict with the splitted dataset subsets: X_train, X_test, y_train, y_test
    sample_size, dataset_splits, sample_ids_train, sample_ids_test, artists_to_class_numbers, class_numbers_to_artists, X_diff_realm, y_diff_realm, sample_ids_diff_realm = prepare_data(input_data_pt, target_data_df, diff_realm_input_data, diff_realm_target_data)

    print("A sample (i.e., image vector embedding) is of size (i.e., a sample has this many features): ", sample_size)
    nb_classes = len(artists_to_class_numbers.keys())
    print(f"We have a {nb_classes} classes classification problem.") 
    print(f"The accuracy of a random classifier is: {str(U.get_rounded_score(1/nb_classes))}%") 

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



