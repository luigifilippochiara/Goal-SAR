import numpy as np
import pandas as pd

from src.data_src.data_utils import split_at_fragment_lambda


class Experiment_base(object):
    """
    The experiment class models an experiment done on a dataset, using
    a particular test_set and trajectory data.

    Here you should define the train/val/test splits strategy, the data
    used for the train/val/test sets and the desired protocol.
    """
    def __init__(self, args):
        self.args = args
        self.dataset = ""
        self.test_set = ""
        self.protocol = None  # "leave_one/many_out", "70_10_20"
        self.train_valid_strategy = None  # "already_split", "to_be_splitted"
        self.data = {}  # where we put train/valid/test partitions

    def _split_df_on_data_points_perc(self, df, perc_train,
                                      perc_valid, perc_test):
        assert perc_train + perc_valid + perc_test == 1

        num_data_points = len(df)
        idx_start_valid = int(num_data_points * perc_train)
        idx_start_test = int(num_data_points * (perc_train + perc_valid))

        train_df = df.iloc[:idx_start_valid]
        valid_df = df.iloc[idx_start_valid:idx_start_test]
        test_df = df.iloc[idx_start_valid:]

        return train_df, valid_df, test_df

    def _split_df_on_frames_perc(self, df, perc_train, perc_valid, perc_test):
        assert perc_train + perc_valid + perc_test == 1

        frames = sorted(list(set(df["frame_id"])))
        tot_frames = len(frames)
        first_valid_frame_idx = int(tot_frames * perc_train)
        first_test_frame_idx = int(tot_frames * (perc_train + perc_valid))
        first_valid_frame = frames[first_valid_frame_idx]
        first_test_frame = frames[first_test_frame_idx]

        train_df = df[df["frame_id"] < first_valid_frame]
        valid_df = df[(first_valid_frame <= df["frame_id"]) & (
                    df["frame_id"] < first_test_frame)]
        test_df = df[first_test_frame <= df["frame_id"]]

        return train_df, valid_df, test_df

    def _print_data_dict_info(self, data_dict):
        print(f"scene_name: {data_dict['scene_name']}")
        print(f"set_name: {data_dict['set_name']}")
        print(f"file_path: {data_dict['file_path']}")
        print()

    def _print_raw_data_info(self, raw_world_data):
        frames = raw_world_data["frame_id"]
        agents_id = raw_world_data["agent_id"]
        X = raw_world_data["x_coord"]
        Y = raw_world_data["y_coord"]
        print("World raw data shape:", raw_world_data.shape)
        print("Number of unique frames:", len(np.unique(frames)))
        print("First frame:", frames.min())
        print("Last frame:", frames.max())
        print("Number of unique agents:", len(set(agents_id)))
        print("First agent id:", agents_id.apply(pd.to_numeric,
                                                 errors='coerce').min())
        print("Last agent id:", agents_id.apply(pd.to_numeric,
                                                errors='coerce').max())
        print(f"Smallest X coord: {X.min():.2f}")
        print(f"Largest X coord: {X.max():.2f}")
        print(f"Mean X coord: {X.mean():.2f}")
        print(f"Width: {X.max() - X.min():.2f}")
        print(f"Smallest Y coord: {Y.min():.2f}")
        print(f"Largest Y coord: {Y.max():.2f}")
        print(f"Mean Y coord: {Y.mean():.2f}")
        print(f"Height: {Y.max() - Y.min():.2f}")

    def verify_experiment_all(self):
        for set_name in ['train', 'valid', 'test']:
            set_name_data = self.data[set_name]
            for data_dict in set_name_data:
                self._print_data_dict_info(data_dict)
                self._print_raw_data_info(data_dict["raw_pixel_data"])
                print("\n" + "* " * 10 + "\n")

    def _split_fragmented(self, df):
        """
        Split trajectories when fragmented (defined as frame_{t+1} - frame_{t} > 1)
        Formally, this is done by changing the agent_id at the fragmented frame
        and above.
        """
        gb = df.groupby('agent_id', as_index=False)
        # calculate frame_{t+1} - frame_{t} and fill NaN which
        # occurs for the first frame of each track
        df['frame_diff'] = gb['frame_id'].diff().fillna(value=1.0).to_numpy()
        # df containing all the first frames of fragmentation
        fragmented = df[df['frame_diff'] != 1.0]
        # helpers for gb.apply
        gb_frag = fragmented.groupby("agent_id")
        frag_idx = fragmented["agent_id"].unique()
        # temporary new_agent_id
        df['new_agent_id'] = df['agent_id']
        # change agent_id for fragmented trajs
        df = gb.apply(split_at_fragment_lambda, frag_idx, gb_frag)
        df['agent_id'] = df['new_agent_id']
        # drop additional columns
        df = df.drop(columns=['new_agent_id', 'frame_diff'])
        return df

    def _load_train_val_test(self):
        raise NotImplementedError

    def _load_all(self):
        self._load_train_val_test()
