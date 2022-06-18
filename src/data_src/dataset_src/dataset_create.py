from src.data_src.dataset_src.dataset_eth5 import Dataset_eth5
from src.data_src.dataset_src.dataset_ind import Dataset_ind
from src.data_src.dataset_src.dataset_sdd import Dataset_sdd


def create_dataset(dataset_name):
    if dataset_name.lower() == 'eth5':
        return Dataset_eth5()
    elif dataset_name.lower() == 'sdd':
        return Dataset_sdd()
    elif dataset_name.lower() == 'ind':
        return Dataset_ind()
    else:
        raise NotImplementedError("Dataset object not available yet!")
