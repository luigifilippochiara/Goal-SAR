import os

import torch

from src.data_src.data_utils import world2pixel_pandas, pixel2world_pandas, \
    world2pixel_torch_multiagent, pixel2world_torch_multiagent
from src.data_src.scene_src.scene_base import Scene_base


class Scene_eth5(Scene_base):
    def __init__(self, scene_name, verbose=False):
        super().__init__()
        self.name = scene_name
        self.dataset_name = "eth5"
        self.dataset_folder = os.path.join(
            self.path_to_root, "data", self.dataset_name)
        self.semantic_map_gt_name = f"{self.name}_mask.png"
        self.has_semantic_map_pred = False

        self.frames_per_second = 2.5
        self.delta_frame = 6 if self.name == 'eth' else 10
        self.unit_of_measure = 'meter'

        scene_to_raw_data = {
            "eth": "biwi_eth.txt",
            "hotel": "biwi_hotel.txt",
            "univ": "students001.txt",
            "zara1": "crowds_zara01.txt",
            "zara2": "crowds_zara02.txt",
        }

        self.raw_scene_data_path = os.path.join(
            self.dataset_folder, self.name, "test",
            scene_to_raw_data[self.name])

        self.load_scene_all(verbose)

    def _invert_x_y_coord_pandas(self, raw_data):
        x_old = raw_data["x_coord"].values.copy()
        y_old = raw_data["y_coord"].values.copy()
        raw_data["x_coord"], raw_data["y_coord"] = y_old, x_old
        return raw_data

    def _invert_x_y_coord_torch(self, batch_coord):
        permutation_index = torch.LongTensor([1, 0])
        batch_coord[..., permutation_index] = batch_coord.clone()
        return batch_coord

    def _make_pixel_coord_pandas(self, raw_world_data):
        raw_world_data_copy = raw_world_data.copy()
        raw_pixel_data = world2pixel_pandas(raw_world_data_copy, self.H_inv)
        # swap Xs and Ys in eth and hotel scenes
        if self.name in ['eth', 'hotel']:
            raw_pixel_data = self._invert_x_y_coord_pandas(raw_pixel_data)
        raw_pixel_data[['x_coord', 'y_coord']] = raw_pixel_data[
            ['x_coord', 'y_coord']]
        return raw_pixel_data

    def _make_world_coord_pandas(self, raw_pixel_data):
        raw_pixel_data_copy = raw_pixel_data.copy()
        # swap Xs and Ys in eth and hotel scenes
        if self.name in ['eth', 'hotel']:
            raw_pixel_data = self._invert_x_y_coord_pandas(raw_pixel_data_copy)
        raw_world_data = pixel2world_pandas(raw_pixel_data, self.H)
        return raw_world_data

    def make_pixel_coord_torch(self, world_batch_coord):
        world_batch_clone = world_batch_coord.clone()
        pixel_coord = world2pixel_torch_multiagent(world_batch_clone, self.H_inv)
        # swap x and y coordinates for eth and hotel scenes
        if self.name in ['eth', 'hotel']:
            pixel_coord = self._invert_x_y_coord_torch(pixel_coord)
        return pixel_coord

    def make_world_coord_torch(self, pixel_batch_coord):
        pixel_batch_clone = pixel_batch_coord.clone()
        # swap x and y coordinates for eth and hotel scenes
        if self.name in ['eth', 'hotel']:
            pixel_batch_clone = self._invert_x_y_coord_torch(pixel_batch_clone)
        world_coord = pixel2world_torch_multiagent(pixel_batch_clone, self.H)
        return world_coord
