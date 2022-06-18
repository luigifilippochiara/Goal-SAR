import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from src.data_src.dataset_src.dataset_create import create_dataset
from src.models.base_model import Base_Model
from src.models.model_utils.U_net_CNN import UNet
from src.losses import MSE_loss, Goal_BCE_loss
from src.metrics import ADE_best_of, KDE_negative_log_likelihood, \
    FDE_best_of_goal, FDE_best_of_goal_world
from src.models.model_utils.sampling_2D_map import sampling, \
    TTST_test_time_sampling_trick


class Goal_SAR(Base_Model):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.dataset = create_dataset(self.args.dataset)

        ##################
        # MODEL PARAMETERS
        ##################

        # set parameters for network architecture
        self.input_size = 2  # size of the input 2: (x,y)
        self.embedding_size = 32  # embedding dimension
        self.nhead = 8  # number of heads in multi-head attentions TF
        self.d_hidden = 2048  # hidden dimension in the TF encoder layer
        self.n_layers_temporal = 1  # number of TransformerEncoderLayers
        self.dropout_prob = 0  # the dropout probability value
        self.add_noise_traj = self.args.add_noise_traj
        self.noise_size = 16  # size of random noise vector
        self.output_size = 2  # output size

        # GOAL MODULE PARAMETERS
        self.num_image_channels = 6

        # U-net encoder channels
        self.enc_chs = (self.num_image_channels + self.args.obs_length,
                        32, 32, 64, 64, 64)
        # U-net decoder channels
        self.dec_chs = (64, 64, 64, 32, 32)

        self.extra_features = 4  # extra information to concat goals: time,
        # last positions, predicted final positions, distance to predicted goals

        ##################
        # MODEL LAYERS
        ##################

        self.goal_module = UNet(
            enc_chs=(self.num_image_channels + self.args.obs_length,
                     32, 32, 64, 64, 64),
            dec_chs=(64, 64, 64, 32, 32),
            out_chs=self.args.pred_length)

        # linear layer to map input to embedding
        self.input_embedding_layer_temporal = nn.Linear(
            self.input_size, self.embedding_size)

        # ReLU and dropout init
        self.relu = nn.ReLU()
        self.dropout_input_temporal = nn.Dropout(self.dropout_prob)

        # temporal encoder layer for temporal sequence
        self.temporal_encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_size + self.extra_features * self.nhead,
            nhead=self.nhead,
            dim_feedforward=self.d_hidden)

        # temporal encoder for temporal sequence
        self.temporal_encoder = TransformerEncoder(
            self.temporal_encoder_layer,
            num_layers=self.n_layers_temporal)

        # fusion layer
        self.fusion_layer = nn.Linear(
            self.embedding_size + self.nhead * self.extra_features + \
            self.extra_features * 2 - 1, self.embedding_size)

        # FC decoder
        if self.add_noise_traj:
            self.output_layer = nn.Linear(
                self.embedding_size + self.noise_size, self.output_size)
        else:
            self.output_layer = nn.Linear(
                self.embedding_size, self.output_size)

    def prepare_inputs(self, batch_data, batch_id):
        """
        Prepare inputs to be fed to a generic model.
        """
        # we need to remove first dimension which is added by torch.DataLoader
        # float is needed to convert to 32bit float
        selected_inputs = {k: v.squeeze(0).float().to(self.device) if \
            torch.is_tensor(v) else v for k, v in batch_data.items()}
        # extract seq_list
        seq_list = selected_inputs["seq_list"]
        # decide which is ground truth
        ground_truth = selected_inputs["abs_pixel_coord"]

        scene_name = batch_id["scene_name"][0]
        scene = self.dataset.scenes[scene_name]
        selected_inputs["scene"] = scene

        return selected_inputs, ground_truth, seq_list

    def init_losses(self):
        losses = {
            "traj_MSE_loss": 0,
            "goal_BCE_loss": 0,
        }
        return losses

    def set_losses_coeffs(self):
        losses_coeffs = {
            "traj_MSE_loss": 1,
            "goal_BCE_loss": 1e6,
        }
        return losses_coeffs

    def init_train_metrics(self):
        train_metrics = {
            "ADE": [],
            "FDE": [],
        }
        return train_metrics

    def init_test_metrics(self):
        test_metrics = {
            "ADE": [],
            "FDE": [],
            "ADE_world": [],
            "FDE_world": [],
            "NLL": [],
        }
        return test_metrics

    def init_best_metrics(self):
        best_metrics = {
            "ADE": 1e9,
            "FDE": 1e9,
            "ADE_world": 1e9,
            "FDE_world": 1e9,
            "goal_BCE_loss": 1e9,
        }
        return best_metrics

    def best_valid_metric(self):
        return "FDE"

    def compute_model_losses(self,
                             outputs,
                             ground_truth,
                             loss_mask,
                             inputs,
                             aux_outputs):
        """
        Compute loss for a generic model.
        """
        out_maps_GT_goal = inputs["input_traj_maps"][:, self.args.obs_length:]
        goal_logit_map = aux_outputs["goal_logit_map"]
        goal_BCE_loss = Goal_BCE_loss(
            goal_logit_map, out_maps_GT_goal, loss_mask)

        losses = {
            "traj_MSE_loss": MSE_loss(outputs, ground_truth, loss_mask),
            "goal_BCE_loss": goal_BCE_loss,
        }

        return losses

    def compute_model_metrics(self,
                              metric_name,
                              phase,
                              predictions,
                              ground_truth,
                              metric_mask,
                              all_aux_outputs,
                              inputs,
                              obs_length=8):
        """
        Compute model metrics for a generic model.
        Return a list of floats (the given metric values computed on the batch)
        """
        if phase == 'test':
            compute_nll = self.args.compute_test_nll
            num_samples = self.args.num_test_samples
        elif phase == 'valid':
            compute_nll = self.args.compute_valid_nll
            num_samples = self.args.num_valid_samples
        else:
            compute_nll = False
            num_samples = 1

        # scale back to original dimension
        predictions = predictions.detach() * self.args.down_factor
        ground_truth = ground_truth.detach() * self.args.down_factor

        # convert to world coordinates
        scene = inputs["scene"]
        pred_world = []
        for i in range(predictions.shape[0]):
            pred_world.append(scene.make_world_coord_torch(predictions[i]))
        pred_world = torch.stack(pred_world)

        GT_world = scene.make_world_coord_torch(ground_truth)

        if metric_name == 'ADE':
            return ADE_best_of(
                predictions, ground_truth, metric_mask, obs_length)
        elif metric_name == 'FDE':
            return FDE_best_of_goal(all_aux_outputs, ground_truth,
                                    metric_mask, self.args)
        if metric_name == 'ADE_world':
            return ADE_best_of(
                pred_world, GT_world, metric_mask, obs_length)
        elif metric_name == 'FDE_world':
            return FDE_best_of_goal_world(all_aux_outputs, scene,
                                          GT_world, metric_mask, self.args)
        elif metric_name == 'NLL':
            if compute_nll and num_samples > 1:
                return KDE_negative_log_likelihood(
                    predictions, ground_truth, metric_mask, obs_length)
            else:
                return [0, 0, 0]
        else:
            raise ValueError("This metric has not been implemented yet!")

    def forward(self, inputs, num_samples=1, if_test=False):

        batch_coords = inputs["abs_pixel_coord"]
        # Number of agent in current batch_abs_world
        seq_length, num_agents, _ = batch_coords.shape

        ##################
        # PREDICT GOAL
        ##################
        # extract precomputed map for goal goal_idx
        tensor_image = inputs["tensor_image"].unsqueeze(0).\
            repeat(num_agents, 1, 1, 1)
        obs_traj_maps = inputs["input_traj_maps"][:, 0:self.args.obs_length]
        input_goal_module = torch.cat((tensor_image, obs_traj_maps), dim=1)
        # compute goal maps
        goal_logit_map_start = self.goal_module(input_goal_module)
        goal_prob_map = torch.sigmoid(
            goal_logit_map_start[:, -1:] / self.args.sampler_temperature)

        if self.args.use_ttst and num_samples > 3:
            goal_point_start = TTST_test_time_sampling_trick(
                goal_prob_map,
                num_goals=num_samples,
                device=self.device)
            goal_point_start = goal_point_start.squeeze(2).permute(1, 0, 2)
        else:
            goal_point_start = sampling(goal_prob_map, num_samples=num_samples)
            goal_point_start = goal_point_start.squeeze(1)

        # START SAMPLES LOOP
        all_outputs = []
        all_aux_outputs = []
        for sample_idx in range(num_samples):
            # Output tensor of shape (seq_length,N,2)
            outputs = torch.zeros(seq_length, num_agents,
                                  self.output_size).to(self.device)
            # add observation as first output
            outputs[0:self.args.obs_length] = batch_coords[0:self.args.obs_length]

            # create noise vector to promote different trajectories
            noise = torch.randn((1, self.noise_size)).to(self.device)

            if if_test:
                goal_point = goal_point_start[:, sample_idx]
            else:
                # teacher forcing
                goal_point = batch_coords[-1]

            # list of auxiliary outputs
            aux_outputs = {
                "goal_logit_map": goal_logit_map_start,
                "goal_point": goal_point}

            ##################
            # loop over seq_length-1 frames, starting from frame 8
            ##################
            for frame_idx in range(self.args.obs_length, self.args.seq_length):
                # If testing phase and frame >= obs_length (prediction)
                if if_test and frame_idx >= self.args.obs_length:
                    # Get current agents positions: from 0 to obs_length take
                    # GT, then previous predicted positions
                    current_agents = torch.cat((
                        batch_coords[:self.args.obs_length],
                        outputs[self.args.obs_length:frame_idx]))
                else:  # Train phase or frame < obs_length (observation)
                    # Current agents positions
                    current_agents = batch_coords[:frame_idx]

                ##################
                # RECURRENT MODULE
                ##################

                # Input Embedding
                temporal_input_embedded = self.dropout_input_temporal(self.relu(
                    self.input_embedding_layer_temporal(current_agents)))

                # compute current positions and current time step
                # and distance to goal
                last_positions = current_agents[-1]
                current_time_step = torch.full(size=(last_positions.shape[0], 1),
                                               fill_value=frame_idx).to(self.device)
                distance_to_goal = goal_point - last_positions
                # prepare everything for concatenation
                # Transformers need everything to be multiple of nhead
                last_positions_to_cat = last_positions.repeat(
                    frame_idx, 1, self.nhead//2)
                current_time_step_to_cat = current_time_step.repeat(
                    frame_idx, 1, self.nhead)
                final_positions_pred_to_cat = goal_point.repeat(
                    frame_idx, 1, self.nhead//2)
                distance_to_goal_to_cat = distance_to_goal.repeat(
                    frame_idx, 1, self.nhead//2)

                # concat additional info BEFORE temporal transformer
                temporal_input_cat = torch.cat(
                    (temporal_input_embedded,
                     final_positions_pred_to_cat,
                     last_positions_to_cat,
                     distance_to_goal_to_cat,
                     current_time_step_to_cat,
                     ), dim=2)

                # temporal transformer encoding
                temporal_output = self.temporal_encoder(
                    temporal_input_cat)
                # Take last temporal encoding
                temporal_output_last = temporal_output[-1]

                # concat additional info AFTER temporal transformer
                fusion_feat = torch.cat((
                    temporal_output_last,
                    last_positions,
                    goal_point,
                    distance_to_goal,
                    current_time_step,
                ), dim=1)

                # fusion FC layer
                fusion_feat = self.fusion_layer(fusion_feat)

                if self.add_noise_traj:
                    # Concatenate noise to fusion output
                    noise_to_cat = noise.repeat(fusion_feat.shape[0], 1)
                    fusion_feat = torch.cat((fusion_feat, noise_to_cat), dim=1)

                # Output FC decoder
                outputs_current = self.output_layer(fusion_feat)
                # append to outputs
                outputs[frame_idx] = outputs_current

            all_outputs.append(outputs)
            all_aux_outputs.append(aux_outputs)

        # stack predictions
        all_outputs = torch.stack(all_outputs)
        # from list of dict to dict of list (and then tensors)
        all_aux_outputs = {k: torch.stack([d[k] for d in all_aux_outputs])
                           for k in all_aux_outputs[0].keys()}
        return all_outputs, all_aux_outputs
