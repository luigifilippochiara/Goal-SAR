from src.data_src.experiment_src.experiment_eth5 import Experiment_eth5
from src.data_src.experiment_src.experiment_ind import Experiment_ind
from src.data_src.experiment_src.experiment_sdd import Experiment_sdd


def create_experiment(dataset_name):
    if dataset_name.lower() == 'eth5':
        return Experiment_eth5
    elif dataset_name.lower() == 'sdd':
        return Experiment_sdd
    elif dataset_name.lower() == 'ind':
        return Experiment_ind
    else:
        raise NotImplementedError("Experiment object not available yet!")
