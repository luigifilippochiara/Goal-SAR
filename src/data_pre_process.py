import os
import pickle
import random

import numpy as np
import torch

from src.data_src.dataset_src.dataset_create import create_dataset
from src.data_src.experiment_src.experiment_create import create_experiment
from src.models.model_utils.cnn_big_images_utils import create_tensor_image, \
    create_CNN_inputs_loop
from src.utils import maybe_makedir


def is_legitimate_traj(traj_df, step):
    agent_id = traj_df.agent_id.values
    # check if I only have 1 agent (always the same)
    if not (agent_id[0] == agent_id).all():
        print("not same agent")
        return False
    frame_ids = traj_df.frame_id.values
    equi_spaced = np.arange(frame_ids[0], frame_ids[-1] + 1, step, dtype=int)
    # check that frame rate is evenly-spaced
    if not (frame_ids == equi_spaced).all():
        print("not equi_spaced")
        return False
    # if checks are passed
    return True


class Trajectory_Data_Pre_Process(object):
    def __init__(self, args):
        self.args = args

        # Trajectories and data_batches folder
        self.data_batches_path = os.path.join(
            self.args.save_dir, 'data_batches')
        maybe_makedir(self.data_batches_path)

        # Creating batches folders and files
        self.batches_folders = {}
        self.batches_confirmation_files = {}
        for set_name in ['train', 'valid', 'test']:
            # batches folders
            self.batches_folders[set_name] = os.path.join(
                self.data_batches_path, f"{set_name}_batches")
            maybe_makedir(self.batches_folders[set_name])
            # batches confirmation file paths
            self.batches_confirmation_files[set_name] = os.path.join(
                self.data_batches_path, f"finished_{set_name}_batches.txt")

        # exit pre-processing early
        if os.path.exists(self.batches_confirmation_files["test"]):
            print('Data batches already created!\n')
            return

        print("Loading dataset and experiment ...")
        self.dataset = create_dataset(self.args.dataset)
        self.experiment = create_experiment(self.args.dataset)(
            self.args.test_set, self.args)
        print("Done.\n")

        print("Preparing data batches ...")
        self.num_batches = {}
        for set_name in ['train', 'valid', 'test']:
            if not os.path.exists(self.batches_confirmation_files[set_name]):
                self.num_batches[set_name] = 0
                print(f"\nPreparing {set_name} batches ...")
                self.create_data_batches(set_name)

        print('Data batches created!\n')

    def create_data_batches(self, set_name):
        """
        Create data batches for the DataLoader object
        """
        for scene_data in self.experiment.data[set_name]:
            # break if fast_debug
            if self.args.fast_debug and self.num_batches[set_name] >= \
                    self.args.fast_debug_num:
                break
            self.make_batches(scene_data, set_name)
            print(f"Saved a total of {self.num_batches[set_name]} {set_name} "
                  f"batches ...")

        with open(self.batches_confirmation_files[set_name], "w") as f:
            f.write(f"Number of {set_name} batches: "
                    f"{self.num_batches[set_name]}")

    def make_batches(self, scene_data, set_name):
        """
        Query the trajectories fragments and make data batches.
        Notes: Divide the fragment if there are too many people; accumulate some
        fragments if there are few people.
        """
        scene_name = scene_data["scene_name"]
        scene = self.dataset.scenes[scene_name]
        delta_frame = scene.delta_frame
        downsample_frame_rate = scene_data["downsample_frame_rate"]

        df = scene_data['raw_pixel_data']

        if set_name == 'train':
            shuffle = self.args.shuffle_train_batches
        elif set_name == 'test':
            shuffle = self.args.shuffle_test_batches
        else:
            shuffle = self.args.shuffle_test_batches
        assert scene_data["set_name"] == set_name

        fragment_list = []  # container for a batch of data (list of fragments)

        for agent_i in set(df.agent_id):
            hist = df[df.agent_id == agent_i]
            # downsample frame rate happens here, at the single agent level
            hist = hist.iloc[::downsample_frame_rate]

            for start_t in range(0, len(hist), self.args.skip_ts_window):
                candidate_traj = hist.iloc[start_t:start_t + self.args.seq_length]
                if len(candidate_traj) == self.args.seq_length:
                    if is_legitimate_traj(candidate_traj,
                                          step=downsample_frame_rate * delta_frame):
                        fragment_list.append(candidate_traj)

        if shuffle:
            random.shuffle(fragment_list)

        batch_acculumator = []
        batch_ids = {
            "scene_name": scene_name,
            "starting_frames": [],
            "agent_ids": [],
            "data_file_path": scene_data["file_path"]}

        for fragment_df in fragment_list:
            # break if fast_debug
            if self.args.fast_debug and self.num_batches[set_name] >= \
                    self.args.fast_debug_num:
                break

            batch_ids["starting_frames"].append(fragment_df.frame_id.iloc[0])
            batch_ids["agent_ids"].append(fragment_df.agent_id.iloc[0])

            batch_acculumator.append(fragment_df[["x_coord", "y_coord"]].values)

            # save batch if big enough
            if len(batch_acculumator) == self.args.batch_size:
                # create and save batch
                self.massup_batch_and_save(batch_acculumator,
                                           batch_ids, set_name)

                # reset batch_acculumator and ids for new batch
                batch_acculumator = []
                batch_ids = {
                    "scene_name": scene_name,
                    "starting_frames": [],
                    "agent_ids": [],
                    "data_file_path": scene_data["file_path"]}

        # save last (incomplete) batch if there is some fragment left
        if batch_acculumator:
            # create and save batch
            self.massup_batch_and_save(batch_acculumator,
                                       batch_ids, set_name)

    def massup_batch_and_save(self, batch_acculumator, batch_ids, set_name):
        """
        Mass up data fragments to form a batch and then save it to disk.
        From list of dataframe fragments to saved batch.
        """
        abs_pixel_coord = np.stack(batch_acculumator).transpose(1, 0, 2)
        seq_list = np.ones((abs_pixel_coord.shape[0],
                            abs_pixel_coord.shape[1]))

        data_dict = {
            "abs_pixel_coord": abs_pixel_coord,
            "seq_list": seq_list,
        }

        # add cnn maps and inputs
        data_dict = self.add_pre_computed_cnn_maps(data_dict, batch_ids)

        # increase batch number count
        self.num_batches[set_name] += 1
        batch_name = os.path.join(
            self.batches_folders[set_name],
            f"{set_name}_batch" + "_" + str(
                self.num_batches[set_name]).zfill(4) + ".pkl")
        # save batch
        with open(batch_name, "wb") as f:
            pickle.dump((data_dict, batch_ids), f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def add_pre_computed_cnn_maps(self, data_dict, batch_ids):
        """
        Pre-compute CNN maps used by the goal modules and add them to data_dict
        """
        abs_pixel_coord = data_dict["abs_pixel_coord"]
        scene_name = batch_ids["scene_name"]
        scene = self.dataset.scenes[scene_name]

        # numpy semantic map from 0 to 1
        img = scene.semantic_map_pred

        tensor_image = create_tensor_image(
            big_numpy_image=img,
            down_factor=self.args.down_factor)

        input_traj_maps = create_CNN_inputs_loop(
            batch_abs_pixel_coords=torch.tensor(abs_pixel_coord).float() /
                                   self.args.down_factor,
            tensor_image=tensor_image)

        data_dict["tensor_image"] = tensor_image
        data_dict["input_traj_maps"] = input_traj_maps

        return data_dict
