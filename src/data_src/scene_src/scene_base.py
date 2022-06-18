import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image
import cv2

from src.data_src import data_utils


class Scene_base(object):
    """
    The Scene class contains generic scene information, mainly
    paths, homography matrix, maps (RGB, semantic) etc...

    It should not contain trajectory speficic data, but only
    scene general information (i.e. FPS, delta_time, delta_frame,
    unit_of_measure etc...)
    """

    def __init__(self):
        self.name = "base_scene"
        self.dataset_folder = ""
        self.path_to_root = data_utils.get_data_location()

        self.frames_per_second = 0
        self.delta_frame = 0
        self.unit_of_measure = None  # meter or pixel

        # extra data
        self.has_H = True
        self.H_name = "H.txt"
        self.has_RGB_image = True
        self.RGB_image_name = "RGB.jpg"
        self.has_semantic_map_gt = True
        self.has_semantic_map_pred = True
        self.semantic_map_gt_name = "scene_mask.png"
        self.semantic_map_pred_name = "pred_mask.png"
        self.has_ortho_px_to_meter = False
        self.ortho_px_to_meter_name = "ortho_px_to_meter.txt"

        # other booleans
        self.has_z_coord = False

        # need to load scene data
        self.raw_scene_data_path = ""
        self.column_names = ["frame_id", "agent_id", "x_coord", "y_coord"]
        self.column_dtype = {"frame_id": int,
                             "agent_id": int,
                             "x_coord": float,
                             "y_coord": float}

        # semantic classes
        self.semantic_classes = OrderedDict([
            ('unlabeled', 'gray'),
            ('pavement', 'blue'),
            ('road', 'red'),
            ('structure', 'orange'),
            ('terrain', 'cyan'),
            ('tree', 'green'),
        ])

    def get_class_name(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"<{self.name} scene object>"

    def get_name(self):
        return self.name

    def _load_H_matix(self):
        if self.has_H:
            self.H_path = os.path.join(self.scene_folder, self.H_name)
            return np.loadtxt(self.H_path)

    def _compute_H_inv(self):
        if self.has_H:
            return np.linalg.inv(self.H)

    def _load_ortho_px_to_meter(self):
        if self.has_ortho_px_to_meter:
            self.ortho_px_to_meter_path = os.path.join(
                self.scene_folder, self.ortho_px_to_meter_name)
            return float(np.loadtxt(self.ortho_px_to_meter_path))

    def _load_RGB_image(self):
        if self.has_RGB_image:
            self.RGB_image_path = os.path.join(
                self.scene_folder, self.RGB_image_name)
            with Image.open(self.RGB_image_path) as f:
                image = np.asarray(f)
            return image

    def _load_semantic_map(self, path):
        sem_map = cv2.imread(path, flags=0)
        # from (X,Y) valued in [0,C] to (X,Y,C) valued in [0,1]
        num_classes = len(self.semantic_classes)
        sem_map = [(sem_map == v) for v in range(num_classes)]
        sem_map = np.stack(sem_map, axis=-1).astype(int)
        return sem_map

    def _load_semantic(self, semantic='gt'):
        if semantic == 'gt' and self.has_semantic_map_gt:
            semantic_map_path = os.path.join(
                self.scene_folder, self.semantic_map_gt_name)
            sem_map = self._load_semantic_map(semantic_map_path)
            return sem_map
        elif semantic == 'pred' and self.has_semantic_map_pred:
            semantic_map_path = os.path.join(
                self.scene_folder, self.semantic_map_pred_name)
            sem_map = self._load_semantic_map(semantic_map_path)
            return sem_map

    def _load_extra_data(self):
        self.scene_folder = os.path.join(self.dataset_folder, self.name)
        self.H = self._load_H_matix()
        self.H_inv = self._compute_H_inv()
        self.ortho_px_to_meter = self._load_ortho_px_to_meter()
        self.RGB_image = self._load_RGB_image()
        self.semantic_map_gt = self._load_semantic(semantic='gt')
        self.semantic_map_pred = self._load_semantic(semantic='pred')

    def _make_pixel_coord_pandas(self, raw_world_data):
        raise NotImplementedError

    def _make_world_coord_pandas(self, raw_pixel_data):
        raise NotImplementedError

    def make_pixel_coord_torch(self, world_batch_coord):
        raise NotImplementedError

    def make_world_coord_torch(self, pixel_batch_coord):
        raise NotImplementedError

    def _load_raw_data_table(self, path):
        # load .txt raw data table
        raw_data = pd.read_csv(path,
                               sep=None,
                               engine='python',
                               header=None,
                               names=self.column_names,
                               dtype=self.column_dtype)
        return raw_data

    def load_raw_data(self):
        if self.unit_of_measure == 'meter':
            print(f"Loading data from {self.raw_scene_data_path} ...")
            self.raw_world_data = self._load_raw_data_table(
                self.raw_scene_data_path)
            print("Scene data loaded")
            self.raw_pixel_data = self._make_pixel_coord_pandas(
                self.raw_world_data)
        elif self.unit_of_measure == 'pixel':
            print(f"Loading data from {self.raw_scene_data_path} ...")
            self.raw_pixel_data = self._load_raw_data_table(
                self.raw_scene_data_path)
            print("Scene data loaded")
            self.raw_world_data = self._make_world_coord_pandas(
                self.raw_pixel_data)
        else:
            raise ValueError("Unit of measure")

    def load_scene_all(self, verbose=False):
        self.delta_time = 1 / self.frames_per_second
        self._load_extra_data()
