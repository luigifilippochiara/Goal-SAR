from src.data_src import data_utils


class Dataset_base(object):
    """
    The Dataset class models the actual dataset and contains the scene objects.
    self.scenes is a dictionary of Scene objects.
    """

    def __init__(self):
        self.name = ""
        self.dataset_folder = ""
        self.path_to_root = data_utils.get_data_location()
        self.scenes = {}
        self.agent_types = ["pedestrian"]

    def get_class_name(self):
        return self.__class__.__name__

    def get_name(self):
        return self.name

    @property
    def n_scenes(self):
        return len(self.scenes)

    @property
    def scene_names(self):
        return [scene.name for scene in self.scenes.values()]

    def __repr__(self):
        return f"<{self.name} dataset object>"

    def compute_dataset_stats(self, verbose):
        """
        This method only works is dataset is initialize with verbose=True
        """
        num_data_points = 0
        num_agents = 0
        num_frames = 0
        for scene in self.scenes.values():
            num_data_points += scene.num_data_points
            num_agents += scene.num_agents
            num_frames += scene.num_frames
        self.num_data_points = num_data_points
        self.num_agents = num_agents
        self.num_frames = num_frames
        if verbose:
            print(f"Number of scenes: {len(self.scenes)}")
            print(f"Number of data points: {num_data_points}")
            print(f"Number of unique agents: {num_agents}")
            print(f"Number of unique frames: {num_frames}")
