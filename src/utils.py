"""
Simple and reusable util functions that can be used in different part of
the project (and in other projects too).
"""

import os
import time
import math
import random
import argparse
import datetime

import numpy
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('true', 't', 'yes', 'y', 'on', '1'):
        return True
    elif v.lower() in ('false', 'f', 'no', 'n', 'off', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def strOrNone(v):
    if v is None:
        return v
    elif v.lower() in ('none', 'no', 'n', 'false', 'f', 'off', '0'):
        return None
    else:
        return v


def is_power_of_two(n: int):
    """
    Return True if n is a power of 2
    """
    return (n & (n - 1) == 0) and n != 0


def previous_power_of_2(n: int):
    """
    Return the biggest power of 2 which is smaller than n
    """
    return 2 ** math.floor(math.log2(n))


def next_power_of_2(n: int):
    """
    Return the smallest power of 2 which is bigger than n
    """
    return 2 ** math.ceil(math.log2(n))


def add_dict_prefix(d_in: dict, prefix: str):
    """
    Add prefix sub string to a dictionary with string keys
    """
    d_out = {prefix + '_' + k: v for k, v in d_in.items()}
    return d_out


def add_dict_suffix(d_in: dict, suffix: str):
    """
    Add suffix sub string to a dictionary with string keys
    """
    d_out = {k + "_" + suffix: v for k, v in d_in.items()}
    return d_out


def maybe_makedir(path_to_create: str):
    """
    This function will create a directory, unless it exists already.

    Parameters
    ----------
    path_to_create : string
        A string path to a directory you'd like created.
    """
    if not os.path.isdir(path_to_create):
        os.makedirs(path_to_create)


def formatted_time(elapsed_time):
    """
    Given an elapsed time in seconds, return a string with a nice format
    """
    if elapsed_time >= 3600:
        return str(datetime.timedelta(seconds=int(elapsed_time)))
    elif elapsed_time >= 60:
        minutes, seconds = elapsed_time // 60, elapsed_time % 60
        return f"{minutes:.0f} min and {seconds:.0f} sec"
    else:
        return f"{elapsed_time:.2f} sec"


def formatted_bytes(bytes_number):
    """
    Given a number of bytes, return a string with a nice format
    """
    if bytes_number >= 1024 ** 4:
        return f"{bytes_number / 1024 ** 4:.1f} TB"
    if bytes_number >= 1024 ** 3:
        return f"{bytes_number / 1024 ** 3:.1f} GB"
    if bytes_number >= 1024 ** 2:
        return f"{bytes_number / 1024 ** 2:.1f} MB"
    if bytes_number >= 1024:
        return f"{bytes_number / 1024:.1f} kB"
    else:
        return f"{bytes_number:.0f} bytes"


def print_git_information(repo_base_dir='.'):
    """
    To know where you are in git and print branch and last commit to screen
    """
    try:
        import git
        repo = git.Repo(repo_base_dir)
        branch = repo.active_branch
        commit = repo.head.commit
        print("Git info. Current branch:", branch)
        print("Last commit:", commit, commit.message)
    except:
        print("An exception occurred while printing git information")


def timed_main(use_git: bool = True):
    """
    Decorator for main function. Start/end date-times and elapsed time.
    Accepts an argument use_git to print or not git information.
    """
    def decorator(function):
        def timed_func(*args, **kw):
            start = time.time()
            now = datetime.datetime.now().strftime("%d %B %Y at %H:%M")
            print('Program started', now)
            if use_git:
                print_git_information()
            result = function(*args, **kw)
            now = datetime.datetime.now().strftime("%d %B %Y at %H:%M")
            elapsed_time = time.time() - start
            print('\nProgram finished {}. Elapsed time: {}'
                  .format(now, formatted_time(elapsed_time)))
            return result
        return timed_func
    return decorator


def timed(unit='ms'):
    """
    Use this decorator factory if you want to time a method or function
    with a specific unit of measure.
    You cannot call it with @timed. Always use @timed().
    """
    assert unit in ['ns', 'ms', 's']

    def decorator(method):
        def timed_func(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if unit == 'ns':
                dt = round((te - ts) * 1e6, 1)
            elif unit == 'ms':
                dt = round((te - ts) * 1e3, 1)
            else:  # seconds
                dt = round(te - ts, 2)
            print(f"Function {method.__name__} time {dt} {unit}\n")
            return result
        return timed_func
    return decorator


def set_seed(seed_value, use_cuda: bool = True):
    """
    Set random, numpy, torch and cuda seeds for reproducibility
    """
    random.seed(seed_value)
    numpy.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed_all(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_trainable_layers(model, verbose=False):
    """
    Given a PyTorch model, returns trainable layers
    """
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            if verbose:
                print(name)
    return params_to_update


def freeze_model_component(model, component_name, mode='starts'):
    """
    Given a PyTorch model, freeze some given layers
    """
    for name, param in model.named_parameters():
        if mode == 'starts':
            if name.startswith(component_name):
                param.requires_grad = False
        elif mode == 'ends':
            if name.endswith(component_name):
                param.requires_grad = False
        elif mode == 'contains':
            if component_name in name:
                param.requires_grad = False
        else:
            raise NotImplementedError


def unfreeze_all_layers(model):
    """
    Given a PyTorch model, unfreeze all layers
    """
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model):
    """
    Given a PyTorch model, count all parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """
    Given a PyTorch model, count only trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(net, model_name):
    """
    Print network name and parameters
    """
    print(f"Model {model_name} is ready!")
    print("Total number of parameters:", count_parameters(net))
    print("Number of trainable parameters:", count_trainable_parameters(net))
    print(f"Number of non-trainable parameters: "
          f"{count_parameters(net) - count_trainable_parameters(net)}\n")
