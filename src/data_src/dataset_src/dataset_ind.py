import os

from src.data_src.dataset_src.dataset_base import Dataset_base
from src.data_src.scene_src.scene_ind import Scene_ind


class Dataset_ind(Dataset_base):
    """
    The data of this Intersection Drone Dataset is the original one,
    in meter coordinates, where non-pedestrian agents have already been
    filtered out.
    There is a total of 33 different recordings recorded at 4 unique locations.
    The standard protocol is to train on 3 locations and test on the
    remaining one. The test location contains recordings 00 to 06 and
    corresponds to scene ID4 in the original inD paper.
    AT the end there are 26 train scenes and 7 test scenes.
    """
    def __init__(self, verbose=False):
        super().__init__()
        self.name = "ind"
        self.dataset_folder = os.path.join(self.path_to_root, "data", self.name)
        self.test_scenes = {str(i).zfill(2) for i in range(0, 7)}
        self.train_scenes = {str(i).zfill(2) for i in range(7, 33)}
        self.scenes = {key: Scene_ind(key, verbose) for
                       key in self.train_scenes.union(self.test_scenes)}
