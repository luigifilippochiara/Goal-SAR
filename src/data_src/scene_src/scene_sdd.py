import os

import pandas as pd

from src.data_src.data_utils import world2pixel_pandas, pixel2world_pandas, \
    world2pixel_torch_multiagent, pixel2world_torch_multiagent
from src.data_src.scene_src.scene_base import Scene_base


class Scene_sdd(Scene_base):
    def __init__(self, scene_name, verbose=False):
        super().__init__()
        self.name = scene_name
        self.dataset_name = "sdd"
        self.dataset_folder = os.path.join(self.path_to_root, "data",
                                           self.dataset_name)
        self.scene_folder = os.path.join(self.dataset_folder, self.name)
        self.raw_scene_data_path = os.path.join(self.scene_folder,
                                                "annotations.txt")
        self.RGB_image_name = "reference.jpg"
        self.semantic_map_gt_name = f"{self.name}_mask.png"

        self.column_names = [
            'agent_id',
            'x_min',
            'y_min',
            'x_max',
            'y_max',
            'frame_id',
            'lost',
            'occluded',
            'generated',
            'label']
        self.column_dtype = {
            'agent_id': int,
            'x_min': int,
            'y_min': int,
            'x_max': int,
            'y_max': int,
            'frame_id': int,
            'lost': int,
            'occluded': int,
            'generated': int,
            'label': object,
        }

        self.frames_per_second = 30
        self.delta_frame = 1
        self.unit_of_measure = "pixel"

        self.load_scene_all(verbose)

    def _load_raw_data_table(self, path):
        # load .txt raw data table
        raw_data = pd.read_csv(path,
                               sep=None,
                               engine='python',
                               header=None,
                               names=self.column_names,
                               dtype=self.column_dtype)
        # 1. Calculate center point of bounding box
        raw_data["x_coord"] = ((raw_data["x_min"] + raw_data["x_max"]) / 2)
        raw_data["y_coord"] = ((raw_data["y_min"] + raw_data["y_max"]) / 2)
        # 2. Keep Pedestrians only
        # 6 classes are: 'Biker', 'Pedestrian', 'Skater', 'Cart', 'Car', 'Bus'
        raw_data = raw_data[raw_data['label'] == 'Pedestrian']
        # 3. Drop lost samples
        raw_data = raw_data[raw_data['lost'] == 0]
        # 4. Drop useless columns
        raw_data = raw_data.drop(columns=[
            'x_min', 'x_max', 'y_min', 'y_max',
            'occluded', 'generated', 'label', 'lost'])
        return raw_data

    def _make_pixel_coord_pandas(self, raw_world_data):
        raw_pixel_data = raw_world_data.copy()
        raw_pixel_data['y_coord'] *= - 1
        raw_pixel_data = world2pixel_pandas(raw_pixel_data, self.H_inv)
        raw_pixel_data[['x_coord', 'y_coord']] = raw_pixel_data[
            ['x_coord', 'y_coord']]
        return raw_pixel_data

    def _make_world_coord_pandas(self, raw_pixel_data):
        raw_world_data = pixel2world_pandas(raw_pixel_data, self.H)
        raw_world_data['y_coord'] *= - 1
        return raw_world_data

    def make_pixel_coord_torch(self, world_batch_coord):
        pixel_batch_coord = world_batch_coord.clone()
        pixel_batch_coord[:, :, 1] = - pixel_batch_coord[:, :, 1]
        pixel_coord = world2pixel_torch_multiagent(
            pixel_batch_coord, self.H_inv)
        return pixel_coord

    def make_world_coord_torch(self, pixel_batch_coord):
        world_batch_coord = pixel_batch_coord.clone()
        world_batch_coord = pixel2world_torch_multiagent(
            world_batch_coord, self.H)
        world_batch_coord[:, :, 1] = - world_batch_coord[:, :, 1]
        return world_batch_coord
