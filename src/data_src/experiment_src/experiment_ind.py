import os

from tqdm import tqdm

from src.data_src.dataset_src.dataset_ind import Dataset_ind
from src.data_src.experiment_src.experiment_base import Experiment_base


class Experiment_ind(Experiment_base):
    def __init__(self, test_set, args):
        super().__init__(args)
        self.dataset = Dataset_ind()
        self.test_set = test_set
        self.dataset_folder = self.dataset.dataset_folder

        self.protocol = "26_7"
        self.train_valid_strategy = "validate_on_test"
        self.downsample_frame_rate = 10

        self.set_name_to_scenes = {
            "train": self.dataset.train_scenes,
            "valid": self.dataset.test_scenes,
            "test": self.dataset.test_scenes}

        self._load_all()

    def _load_data_files(self, set_name):
        scene_names = self.set_name_to_scenes[set_name]
        set_name_data = []
        for scene_name in tqdm(scene_names):
            scene = self.dataset.scenes[scene_name]
            file_name = f"{scene_name}.csv"
            file_path = os.path.join(self.dataset_folder, scene_name, file_name)

            # load raw data table
            raw_data = scene._load_raw_data_table(file_path)

            # world to pixel
            raw_pixel_data = scene._make_pixel_coord_pandas(
                raw_data.copy())

            set_name_data.append({
                "file_path": file_path,
                "scene_name": scene_name,
                "downsample_frame_rate": self.downsample_frame_rate,
                "set_name": set_name,
                "raw_pixel_data": raw_pixel_data,
            })
        return set_name_data

    def _load_train_val_test(self):
        for set_name in ['train', 'valid', 'test']:
            self.data[set_name] = self._load_data_files(set_name)
