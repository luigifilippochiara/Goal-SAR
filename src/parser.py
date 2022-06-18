import os
import sys
import argparse

import torch
import yaml

sys.path.append('.')  # needed lo launch from .
sys.path.append('..')  # needed lo launch from ./scripts

from src.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        description='Goal-SAR: Goal-driven Self-Attentive Recurrent Networks '
                    'for Trajectory Prediction')

    ##############################################################
    # Dataset, test set and model
    ##############################################################
    parser.add_argument(
        '--dataset', '-d', default='sdd', type=str,
        choices=['eth5', 'sdd', 'ind'],
        help='Dataset selection: ETH-UCY, Stanford Drone, Intersection Drone, '
             'Forking Path')
    parser.add_argument(
        '--test_set', '-ts', default='sdd', type=str,
        choices=['eth', 'hotel', 'univ', 'zara1', 'zara2', 'sdd', 'ind'],
        help='Set this value to [eth, hotel, univ, zara1, zara2, sdd, '
             'ind] for ETH-univ, ETH-hotel, UCY-univ, UCY-zara01, '
             'UCY-zara02, Stanford Drone, Intersection Drone')
    parser.add_argument(
        '--model_name', '-mn', default='Goal_SAR', type=str,
        choices=['SAR',
                 'Goal_SAR'],
        help='Type of architecture to use')

    ##############################################################
    # Training/testing parameters
    ##############################################################
    parser.add_argument(
        '--phase', '-ph', default='train_test', type=str,
        choices=['pre-process', 'train', 'test', 'train_test'],
        help='Phase selection. During test phase you need to load a '
             'pre-trained model')
    parser.add_argument(
        '--load_checkpoint', '-lc', default=None, type=str,
        help="Load pre-trained model for testing or resume training. Specify "
             "the epoch to load or 'best' to load the best model. Default=None "
             "means do not load any model.")
    parser.add_argument(
        '--num_epochs', '-ne', default=300, type=int)
    parser.add_argument(
        '--batch_size', '-bs', default=32, type=int,
        help="Desired number of pedestrians in a batch - more or less. This "
             "is the ideal batch size, but some batches will be smaller and"
             "others bigger, depending on the dataset and scene.")
    parser.add_argument(
        '--save_every', '-se', default=None, type=int,
        help="Save model weights and outputs every save_every epochs. "
             "If None save every num_epochs//5 epochs.")
    # 1 for slow but accurate results, 20 for fast results
    parser.add_argument(
        '--validate_every', '-ve', default=20, type=int,
        help="Validate model every validate_every epochs")
    # 1 for slow but accurate results, 20 for fast results
    parser.add_argument(
        '--skip_ts_window', '-skip', default=5, type=int,
        help="When extracting trajectory fragments, skip skip_ts_window "
             "time-steps between two consecutive starting_frames. "
             "If skip_ts_window >= seq_length there is no overlapping. "
             "If skip_ts_window == 1 make full use of the data.")

    ##############################################################
    # Goal Module parameters
    ##############################################################
    parser.add_argument(
        '--sampler_temperature', default=1, type=float,
        help="Temperature of the map random sampler")
    parser.add_argument(
        '--use_ttst', default=True, type=str2bool, const=True, nargs='?',
        help="Use Test Time Sampling Trick")

    ##############################################################
    # Deep Learning strategies
    ##############################################################
    parser.add_argument(
        '--learning_rate', '-lr', default=1e-4, type=float)
    parser.add_argument(
        '--optimizer', default='Adam', type=str, choices=['Adam', 'SGD'],
        help='Optimizer selection')
    parser.add_argument(
        '--scheduler', default=None, type=str, choices=[
            'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', None],
        help='Learning rate scheduler')
    parser.add_argument(
        '--device', default="cuda:0", type=str, help='What device to use')
    parser.add_argument(
        '--clip', default=1, type=int, help="Gradient clip")
    parser.add_argument(
        '--data_augmentation', default=True, type=str2bool, const=True,
        nargs='?', help="Apply data augmentation to the train set.")
    parser.add_argument(
        '--shift_last_obs', default=True, type=str2bool, const=True, nargs='?',
        help="Shift batch trajectories based on last observation coordinates. "
             "Used for data normalization.")
    parser.add_argument(
        '--add_noise_traj', default=True, type=str2bool, const=True, nargs='?',
        help="Add random noise before traj decoder to promote variability")

    ##############################################################
    # General parameters
    ##############################################################
    parser.add_argument(
        '--down_factor', default=8, type=int,
        help="Image down scale factor for CNN")
    parser.add_argument(
        '--traj_normalization', default=16, type=int,
        nargs='?', help="Traj normalization constant.")
    parser.add_argument(
        '--start_validation', default=5, type=int,
        help="Validate the model starting from this epoch")
    parser.add_argument(
        '--num_test_samples', default=20, type=int,
        help="Number of test samples. Set to 1 for deterministic model")
    parser.add_argument(
        '--num_valid_samples', default=20, type=int,
        help="Number of validation samples")
    parser.add_argument(
        '--shuffle_train_batches', default=True, type=str2bool, const=True,
        nargs='?',
        help="Shuffle train batches. Set to False for deterministic behavior.")
    parser.add_argument(
        '--shuffle_test_batches', default=True, type=str2bool, const=True,
        nargs='?',
        help="Shuffle valid and test batches. Set to False for deterministic "
             "behavior.")

    ##############################################################
    # Debug parameters
    ##############################################################
    parser.add_argument(
        '--use_wandb', default=True, type=str2bool, const=True, nargs='?')
    parser.add_argument(
        '--num_test_runs', default=10, type=int,
        help="Number of test run to average")
    parser.add_argument(
        '--fast_debug', '-fd', default=False, type=str2bool, const=True,
        nargs='?', help="Set to True for fast debug mode")
    parser.add_argument(
        '--fast_debug_num', default=3, type=int,
        help="Number of batches in fast debug mode")
    parser.add_argument(
        '--compute_valid_nll', default=False, type=str2bool, const=True,
        nargs='?', help="Set to True to compute validation KDE-NLL")
    parser.add_argument(
        '--compute_test_nll', default=True, type=str2bool, const=True,
        nargs='?', help="Set to True to compute test KDE-NLL")
    parser.add_argument(
        '--reproducibility', '-r', default=True, type=str2bool, const=True,
        nargs='?', help="Set to True to set the seed for reproducibility")

    return parser


def load_args(cur_parser, parsed_args):
    """
    Load args from saved config file and confront them with parsed args.
    parsed_args are the entered parsed arguments for this run, while saved_args
    are the previously saved config arguments.

    The priority is:
    command line > saved configuration files > default values in script.
    """
    with open(parsed_args.config, 'r') as f:
        saved_args = yaml.full_load(f)
    for k in saved_args.keys():
        if k not in vars(parsed_args).keys():
            raise KeyError('WRONG ARG: {}'.format(k))
    assert set(saved_args) == set(vars(parsed_args)), \
        "Entered args and config saved args are different"
    cur_parser.set_defaults(**saved_args)
    return cur_parser.parse_args()


def save_args(args):
    """
    Save args to config file
    """
    args_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(args_dict, f)


def check_and_add_additional_args(args):
    """
    Add default paths, device and other additional args to parsed args
    """
    # set current device
    if args.device.startswith('cuda') and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.device = 'cpu'
        args.use_cuda = False
    # dataset and test_set checks
    if args.dataset == 'eth5':
        assert args.test_set in ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    else:
        # hard assignation of test set
        args.test_set = args.dataset
    if not args.save_every:
        args.save_every = int(args.num_epochs // 5)
    # set parameters for trajectories
    args = compute_term(args)
    # do not add noise in deterministic mode
    if args.num_test_samples == 1:
        args.add_noise_traj = False
    # change a few things in fast debug mode
    if args.fast_debug:
        args.num_test_runs = 1
        args.start_validation = 0
    # set directories
    args.base_dir = '.'  # base directory
    args.save_base_dir = 'output'  # for saving output and models
    args.save_dir = os.path.join(
        args.base_dir, args.save_base_dir, str(args.test_set))
    args.model_dir = os.path.join(
        args.save_dir, args.model_name)
    args.config = os.path.join(
        args.model_dir, 'config_' + args.phase + '.yaml')
    return args


def compute_term(args):
    """
    Set parameters for trajectories
    """
    args.obs_length = 8
    args.pred_length = 12
    args.seq_length = args.obs_length + args.pred_length
    return args


def print_args(args):
    """
    Print parsed args to screen
    """
    print("-"*62)
    print("|" + " "*25 + "PARAMETERS" + " "*25 + "|")
    print("-" * 62)
    for k, v in vars(args).items():
        print(f"| {k:25s}: {v}")
    print("-"*62 + "\n")


def main_parser():
    """
    Pipeline from_parsed args to args
    """
    # Parse input parameters
    parser = get_parser()
    parsed_args = parser.parse_args()
    parsed_args = check_and_add_additional_args(parsed_args)

    # configuration files are created and stored at the first run only
    # if args yaml file does not exist, save it
    if not os.path.exists(parsed_args.config):
        save_args(parsed_args)
    args = load_args(parser, parsed_args)
    # print args given to the model
    print_args(args)
    return args


if __name__ == '__main__':
    # Parse input parameters
    parser = get_parser()
    pars_args = parser.parse_args()
    pars_args = check_and_add_additional_args(pars_args)
    print_args(pars_args)
