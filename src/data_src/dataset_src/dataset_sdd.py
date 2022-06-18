import os

from src.data_src.dataset_src.dataset_base import Dataset_base
from src.data_src.scene_src.scene_sdd import Scene_sdd


class Dataset_sdd(Dataset_base):
    """
    The data of this Stanford Drone Dataset is the original one,
    in pixel coordinates.
    The scenes are the ones defined by the TrajNet Challenge: there
    are 30 train scenes and 17 test scenes, for a total of 47
    different scenes recorded at 8 unique locations.
    """
    def __init__(self, verbose=False):
        super().__init__()
        self.name = "sdd"
        self.dataset_folder = os.path.join(self.path_to_root, "data", self.name)
        self.train_scenes = {
            'bookstore_0',
            'bookstore_0',
            'bookstore_1',
            'bookstore_2',
            'bookstore_3',
            #
            'coupa_3',
            #
            'deathCircle_0',
            'deathCircle_1',
            'deathCircle_2',
            'deathCircle_3',
            'deathCircle_4',
            #
            'gates_0',
            'gates_1',
            'gates_3',
            'gates_4',
            'gates_5',
            'gates_6',
            'gates_7',
            'gates_8',
            #
            'hyang_4',
            'hyang_5',
            'hyang_6',
            'hyang_7',
            'hyang_9',
            #
            'nexus_0',
            'nexus_1',
            'nexus_3',
            'nexus_4',
            'nexus_7',
            'nexus_8',
            'nexus_9',
        }
        self.test_scenes = {
            'coupa_0',
            'coupa_1',
            #
            'gates_2',
            #
            'hyang_0',
            'hyang_1',
            'hyang_3',
            'hyang_8',
            #
            'little_0',
            'little_1',
            'little_2',
            'little_3',
            #
            'nexus_5',
            'nexus_6',
            #
            'quad_0',
            'quad_1',
            'quad_2',
            'quad_3',
        }
        self.scenes = {key: Scene_sdd(key, verbose) for
                       key in self.train_scenes.union(self.test_scenes)}
