import os

from src.data_src.dataset_src.dataset_base import Dataset_base
from src.data_src.scene_src.scene_eth5 import Scene_eth5


class Dataset_eth5(Dataset_base):
    """
    eth5 dataset is composed of 5 scenes: ETH-univ, ETH-hotel,
    UCY-univ, UCY-zara1 and UCY-zara2
    """

    def __init__(self, verbose=False):
        super().__init__()
        self.name = "eth5"
        self.dataset_folder = os.path.join(self.path_to_root, "data", self.name)
        self.scenes = {
            'eth': Scene_eth5('eth', verbose),
            'hotel': Scene_eth5('hotel', verbose),
            'univ': Scene_eth5('univ', verbose),
            'zara1': Scene_eth5('zara1', verbose),
            'zara2': Scene_eth5('zara2', verbose),
        }
