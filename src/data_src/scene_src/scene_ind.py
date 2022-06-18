import os

import pandas as pd

from src.data_src.scene_src.scene_base import Scene_base


class Scene_ind(Scene_base):
    def __init__(self, scene_name, verbose=False):
        super().__init__()
        self.name = scene_name
        self.dataset_name = "ind"
        self.dataset_folder = os.path.join(self.path_to_root, "data",
                                           self.dataset_name)
        self.scene_folder = os.path.join(self.dataset_folder, self.name)
        self.raw_scene_data_path = os.path.join(self.scene_folder,
                                                f"{scene_name}.csv")
        self.RGB_image_name = f"{scene_name}_background.jpg"
        self.semantic_map_gt_name = "scene_mask.png"

        self.column_names = [
            'agent_id',
            'frame_id',
            'x_coord',
            'y_coord',
            'heading',
            'xVelocity',
            'yVelocity',
            'xAcceleration',
            'yAcceleration',
            'class']
        self.column_dtype = {
            'agent_id': int,
            'frame_id': int,
            'x_coord': float,
            'y_coord': float,
            'heading': float,
            'xVelocity': float,
            'yVelocity': float,
            'xAcceleration': float,
            'yAcceleration': float,
            'class': str}

        self.frames_per_second = 25
        self.delta_frame = 1
        self.unit_of_measure = 'meter'

        self.has_H = False
        self.has_ortho_px_to_meter = True
        # used for meter <--> pixel conversion
        self.scale_down_factor = 12

        self.load_scene_all(verbose)

    def _load_raw_data_table(self, path):
        # load .csv raw data table with header
        raw_data = pd.read_csv(path,
                               engine='python',
                               header=None,
                               skiprows=1,
                               names=self.column_names,
                               dtype=self.column_dtype)
        columns_to_drop = [
            'heading',
            'xVelocity',
            'yVelocity',
            'xAcceleration',
            'yAcceleration',
            'class']
        raw_data = raw_data.drop(columns=columns_to_drop)
        return raw_data

    def _make_pixel_coord_pandas(self, raw_world_data):
        raw_pixel_data = raw_world_data.copy()
        raw_pixel_data['y_coord'] *= - 1
        raw_pixel_data[['x_coord', 'y_coord']] /= (
                self.ortho_px_to_meter * self.scale_down_factor)
        raw_pixel_data[['x_coord', 'y_coord']] = raw_pixel_data[
            ['x_coord', 'y_coord']]
        return raw_pixel_data

    def _make_world_coord_pandas(self, raw_pixel_data):
        raw_world_data = raw_pixel_data.copy()
        raw_world_data['y_coord'] *= - 1
        raw_world_data[['x_coord', 'y_coord']] *= (
                self.ortho_px_to_meter * self.scale_down_factor)
        return raw_world_data

    def make_pixel_coord_torch(self, world_batch_coord):
        pixel_batch_coord = world_batch_coord.clone()
        pixel_batch_coord[:, :, 1] *= -1
        pixel_batch_coord /= (self.ortho_px_to_meter * self.scale_down_factor)
        return pixel_batch_coord

    def make_world_coord_torch(self, pixel_batch_coord):
        world_batch_coord = pixel_batch_coord.clone()
        world_batch_coord[:, :, 1] *= -1
        world_batch_coord *= (self.ortho_px_to_meter * self.scale_down_factor)
        return world_batch_coord
