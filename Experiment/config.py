import argparse
import json
import os
import pprint
import torch


class Constants(object):
    """
    This is a singleton (only a single instance of the class should exist in the program).
    """
    class __Constants:
        def __init__(self):

            if torch.cuda.is_available():
                self.DEVICE = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.DEVICE = torch.device('mps')
            else:
                print("Cuda is not available.")
                self.DEVICE = torch.device('cpu')

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)


CONSTANTS = Constants()

class Configuration(object):
    """Configuration parameters given via the commandline."""

    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_cmd():
        command_line_parser = argparse.ArgumentParser()

        # General
        command_line_parser.add_argument("--artist_category", type=str, default="historical", help="'historical' if historical artists, 'artstation' if artstation (i.e., online) artists.")
        command_line_parser.add_argument("--image_source", type=str, default="gen", help="'real' if real images/artworks, 'gen' if generated images/artworks.")
        command_line_parser.add_argument("--sd_version", type=float, default=1.5, help="Stable Diffusion version: 1.5 or 2.1")
        command_line_parser.add_argument("--clip_version", type=str, default="Laion2b", help="CLIP version: OpenAI or Laion2b")
        
        # Setup
        command_line_parser.add_argument('--tag', default='', help='A custom tag for this experiment.')
        command_line_parser.add_argument('--seed', type=int, default=121997, help='Seed for reproducibility, set randomness.')
        command_line_parser.add_argument("--use_wandb", action='store_true', default=False, help="Whether or not to use WandB during the experiment run.")

        # Experiment run
        command_line_parser.add_argument("--eval_mode", action='store_true', default=False, help="Whether or not to explicitly perform a model evaluation during the experiment run.")
        command_line_parser.add_argument("--topk", type=int, default=5, help="Top k classes to consider for the evaluation of the predictive model. Top-k metric will also be compute. E.g.: 5")
        command_line_parser.add_argument("--multi_top_k", action='store_true', default=False, help="Whether or not to compute top-k metric for several k (e.g., k = 1, 3, 5)")
        command_line_parser.add_argument("--use_grid_search", action='store_true', default=False, help="Whether or not to perform a (random) Grid Search for HP optimization of the sklearn models.")
        command_line_parser.add_argument("--k_folds", type=int, default=5, help="Number of folds to have for the Cross-Validation (CV) of sklearn models. E.g.: 5")
        command_line_parser.add_argument("--use_tqdm", action='store_true', default=False, help="Whether or not to use tqdm bar during NN training.")
        command_line_parser.add_argument("--add_verbose", action='store_true', default=False, help="Whether or not to have additional training logs printed in the terminal.")
        command_line_parser.add_argument("--save_experiment_results", action='store_true', default=False, help="Whether or not to save the experiment results (i.e., some of the config and the pred scores obtained.")

        # Data
        command_line_parser.add_argument('--train_test_ratio', type=float, default=0.8, help="The training/testing ratio to use for the given dataset.")
        command_line_parser.add_argument("--rand_shuffle_data", action='store_true', default=False, help="Whether or not to randomly shuffle the data before creating the train and test sets.")
        command_line_parser.add_argument('--balance_classes_across_sets', 
                                         action='store_true', 
                                         default=False, 
                                         help="""
                                         Whether or not to voluntarily balance the random sampling of the samples/classes when creating the train-test split. 
                                         This means that given the train-test split ratio train_test_ratio, there will be train_test_ratio of samples of 
                                         all classes/artists in the train set and (1-train_test_ratio) in the test set.
                                         This ensures that there is no sample class that has not been seen during training while present in the test set.
                                         """)
        command_line_parser.add_argument("--verify_class_balance", action='store_true', default=False, help="Whether or not to verify if the classes are well balanced across the samples split in sets. See the variable balance_classes_across_sets")
        command_line_parser.add_argument("--input_data_path", type=str, default=None, help="Path to the predictive model input data (i.e, Pytorch Tensor) file (.pt).")
        command_line_parser.add_argument("--target_data_path", type=str, default=None, help="Path to the predictive model target data file (.csv).")
        command_line_parser.add_argument("--train_gen_test_real", type=str, default=None, help="Path to the predictive model target data file (.csv).")
        command_line_parser.add_argument("--train_real_test_gen", type=str, default=None, help="Path to the predictive model target data file (.csv).")

        # Model
        command_line_parser.add_argument("--pred_model_name", type=str, default="nn", help="Name of the predictive model: logistic_regression, xgboost, linear_nn, nn.")
        command_line_parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loaders.')
        command_line_parser.add_argument('--n_epochs', type=int, default=200, help='Number of train epochs for the NN model.')
        command_line_parser.add_argument('--print_at_n_epochs', type=int, default=50, help='Print metrics to command line every print_at_n_epochs epochs during NN training.')

        config = command_line_parser.parse_args()
        return Configuration(vars(config))

    @staticmethod
    def from_json(json_path):
        """Load configurations from a JSON file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
            return Configuration(config)

    def to_json(self, json_path):
        """Dump configurations to a JSON file."""
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)