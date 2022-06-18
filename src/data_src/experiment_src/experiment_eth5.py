import os

from tqdm import tqdm

from src.data_src.experiment_src.experiment_base import Experiment_base
from src.data_src.dataset_src.dataset_eth5 import Dataset_eth5


class Experiment_eth5(Experiment_base):
    def __init__(self, test_set, args):
        super().__init__(args)
        self.dataset = Dataset_eth5()
        self.test_set = test_set
        self.test_set_dir = os.path.join(
            self.dataset.dataset_folder, test_set)

        self.protocol = "leave_one_out"
        self.train_valid_strategy = "already_split"
        self.downsample_frame_rate = 1

        self.data_file_name_to_scene = {
            "biwi_eth": 'eth',
            "biwi_hotel": 'hotel',
            "students001": 'univ',
            "students003": 'univ',
            "uni_examples": 'univ',
            "crowds_zara01": 'zara1',
            "crowds_zara02": 'zara2',
            "crowds_zara03": 'zara2',
        }

        self._load_all()

    def _load_data_files(self, set_name):
        phase_name = 'val' if set_name == 'valid' else set_name
        set_name_dir = os.path.join(self.test_set_dir, phase_name)
        set_name_data = []
        for file_name in tqdm(os.listdir(set_name_dir)):
            if file_name.endswith(".txt"):
                file_path = os.path.join(set_name_dir, file_name)
                file_name = file_name.replace(
                    "_train", "").replace("_val", "").replace(".txt", "")
                scene_name = self.data_file_name_to_scene[file_name]
                scene = self.dataset.scenes[scene_name]
                # load raw data table
                raw_world_data = scene._load_raw_data_table(file_path)
                # to pixel
                raw_pixel_data = scene._make_pixel_coord_pandas(
                    raw_world_data.copy())
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
