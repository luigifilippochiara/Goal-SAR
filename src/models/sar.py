import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from src.data_src.dataset_src.dataset_create import create_dataset
from src.models.base_model import Base_Model


class SAR(Base_Model):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.dataset = create_dataset(self.args.dataset)

        # set parameters for network architecture
        self.input_size = 2  # size of the input 2: (x,y)
        self.embedding_size = 32  # embedding dimension
        self.dropout_input_prob = 0  # the initial dropout probability value
        self.nhead = 8  # number of heads in multi-head attentions TF
        self.n_layers_temporal = 1  # number of TransformerEncoderLayers
        self.d_hidden = 2048  # hidden dimension in the TF encoder layer
        self.dropout_TF_prob = 0.1  # dropout in transformer encoder layer
        self.noise_size = 16  # size of random noise vector
        self.output_size = 2  # output size

        # linear layer to map input to embedding
        self.input_embedding_layer_temporal = nn.Linear(
            self.input_size, self.embedding_size)

        # ReLU and dropout init
        self.relu = nn.ReLU()
        self.dropout_input = nn.Dropout(self.dropout_input_prob)

        # temporal encoder layer
        self.temporal_encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.nhead,
            dim_feedforward=self.d_hidden,
            dropout=self.dropout_TF_prob)

        # temporal encoder
        self.temporal_encoder = TransformerEncoder(
            encoder_layer=self.temporal_encoder_layer,
            num_layers=self.n_layers_temporal)

        # FC decoder
        if self.args.add_noise_traj:
            self.output_layer = nn.Linear(
                self.embedding_size + self.noise_size, self.output_size)
        else:
            self.output_layer = nn.Linear(
                self.embedding_size, self.output_size)

    def forward(self, inputs, num_samples=1, if_test=False):

        batch_coords = inputs["abs_pixel_coord"]
        num_agents = batch_coords.shape[1]  # Number of agent in current batch
        if self.args.shift_last_obs:
            shift = batch_coords[self.args.obs_length - 1]
        else:
            shift = torch.zeros_like(batch_coords[0])
        batch_coords = (batch_coords - shift) / self.args.traj_normalization

        all_outputs = []
        all_aux_outputs = []
        for sample_idx in range(num_samples):
            # Output tensor of shape (seq_length,N,2)
            outputs = torch.zeros(batch_coords.shape[0], num_agents,
                                  self.output_size).to(self.device)
            outputs[0] = batch_coords[0]  # add starting prediction to outputs
            aux_outputs = {}
            # create noise vector to promote different trajectories
            noise_sample = torch.randn((1, self.noise_size)).to(self.device)

            # loop over seq_length-1 frames, starting from frame 1
            for frame_idx in range(1, self.args.seq_length):
                # If testing phase and frame >= obs_length (prediction)
                if if_test and frame_idx >= self.args.obs_length:
                    # Get current agents positions: from 0 to obs_length
                    # Take ground truth, then previous predicted positions
                    current_agents = torch.cat((
                        batch_coords[:self.args.obs_length],
                        outputs[self.args.obs_length:frame_idx]))
                else:  # Train phase or frame < obs_length (observation)
                    # Current agents positions
                    current_agents = batch_coords[:frame_idx]

                # Input Embedding
                temporal_input_embedded = self.dropout_input(self.relu(
                    self.input_embedding_layer_temporal(current_agents)))

                # Output temporal transformer --> shape: (frame_idx+1, N, 32)
                temporal_output = self.temporal_encoder(
                    temporal_input_embedded)
                # Take last temporal encoding
                last_temporal_output = temporal_output[-1]

                # Concatenate noise to fusion output
                if self.args.add_noise_traj:
                    noise_to_cat = noise_sample.repeat(
                        last_temporal_output.shape[0], 1)
                    last_temporal_output = torch.cat(
                        (last_temporal_output, noise_to_cat), dim=1)

                # Output FC layer
                output_current = self.output_layer(last_temporal_output)
                # append to outputs
                outputs[frame_idx] = output_current

            # shift normalize back
            outputs = outputs * self.args.traj_normalization + shift
            all_outputs.append(outputs)
            all_aux_outputs.append(aux_outputs)

        # stack predictions
        all_outputs = torch.stack(all_outputs)
        # from list of dict to dict of list (and then tensors)
        all_aux_outputs = {k: torch.stack([d[k] for d in all_aux_outputs])
                           for k in all_aux_outputs[0].keys()}
        return all_outputs, all_aux_outputs
