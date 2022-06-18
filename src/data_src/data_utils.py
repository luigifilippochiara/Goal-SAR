from os.path import abspath, dirname

import torch
import numpy as np
import pandas as pd


def get_data_location():
    # Path(__file__).absolute().parent.parent.parent.parent
    return dirname(dirname(dirname(dirname(abspath(__file__)))))


def split_at_fragment_lambda(x, frag_idx, gb_frag):
    """
    Used only for split_fragmented() in sdd and ind experiments
    """
    agent_id = x["agent_id"].iloc()[0]
    counter = 1
    if agent_id in frag_idx:
        split_idx = gb_frag.groups[agent_id]
        for split_id in split_idx:
            x.loc[split_id:, 'new_agent_id'] = '{}_{}'.format(agent_id, counter)
            counter += 1
    return x


def world2pixel_numpy(world_coord, H_inv):
    """
    Convert a seq_lengthx2 array of (x,y) world coordinates into
    a seq_lengthx2 array of (x,y) pixel coordinates.
    H_inv: 3x3 inverse homography matrix
    """
    assert isinstance(world_coord, np.ndarray)
    assert world_coord.ndim == 2
    assert world_coord.shape[-1] == 2

    # augmented coordinates
    world_coord = np.concatenate(
        (world_coord, np.ones((len(world_coord), 1))), axis=1)
    # world --> pixel
    pixel_coord = np.dot(H_inv, world_coord.T).T
    # normalise pixel coordinates
    pixel_coord = pixel_coord[:, 0:2] / pixel_coord[:, 2][:, np.newaxis]
    # float --> int
    pixel_coord = pixel_coord
    return pixel_coord


def pixel2world_numpy(pixel_coord, H):
    """
    Convert a seq_lengthx2 array of (x,y) pixel coordinates into
    a seq_lengthx2 array of (x,y) world corrdinates.
    H: 3x3 homography matrix
    """
    assert isinstance(pixel_coord, np.ndarray)
    assert pixel_coord.ndim == 2
    assert pixel_coord.shape[-1] == 2

    # augmented coordinates
    pixel_coord = np.concatenate(
        (pixel_coord, np.ones((len(pixel_coord), 1))), axis=1)
    # pixel --> world
    world_coord = np.dot(H, pixel_coord.T).T
    # normalise world coordinates
    world_coord = world_coord[:, 0:2] / world_coord[:, 2][:, np.newaxis]
    return world_coord


def world2pixel_pandas(world_df, H_inv):
    """
    Convert a seq_lengthx4 dataframe of (frame_id, agent_id, x, y) world
    coordinates into a seq_lengthx4 dataframe of (frame_id, agent_id, x,
    y) pixel coordinates.
    H_inv: 3x3 inverse homography matrix
    """
    assert isinstance(world_df, pd.DataFrame)

    pixel_df = world_df.copy()
    world_coord = world_df[["x_coord", "y_coord"]].values
    pixel_coord = world2pixel_numpy(world_coord, H_inv)
    pixel_df[["x_coord", "y_coord"]] = pixel_coord
    return pixel_df


def pixel2world_pandas(pixel_df, H):
    """
    Convert a seq_lengthx4 dataframe of (frame_id, agent_id, x, y) pixel
    coordinates into a seq_lengthx4 dataframe of (frame_id, agent_id, x,
    y) world coordinates.
    H: 3x3 homography matrix
    """
    assert isinstance(pixel_df, pd.DataFrame)

    world_df = pixel_df.copy()
    pixel_coord = pixel_df[["x_coord", "y_coord"]].values
    world_coord = pixel2world_numpy(pixel_coord, H)
    world_df[["x_coord", "y_coord"]] = world_coord
    return world_df


def world2pixel_torch(world_coord, H_inv):
    """
    Convert a Nx2 Pytorch tensor of (x,y) world coordinates into
    a Nx2 Pytorch tensor of (x,y) pixel coordinates.
    H_inv: 3x3 inverse homography matrix
    """
    assert isinstance(world_coord, torch.Tensor)
    assert world_coord.dim() == 2
    assert world_coord.shape[-1] == 2

    # H_inv from np.array to tensor
    H_inv = torch.Tensor(H_inv).float()

    # augmented coordinates
    world_coord = torch.cat(
        (world_coord, torch.ones(len(world_coord), 1)), dim=1)
    # world --> pixel
    pixel_coord = torch.matmul(H_inv, world_coord.T).T
    # normalise pixel coordinates
    pixel_coord = pixel_coord[:, :2] / pixel_coord[:, 2].unsqueeze(1)
    return pixel_coord


def pixel2world_torch(pixel_coord, H):
    """
    Convert a Nx2 Pytorch tensor of (x,y) pixel coordinates into
    a Nx2 Pytorch tensor of (x,y) world coordinates.
    H: 3x3 homography matrix
    """
    assert isinstance(pixel_coord, torch.Tensor)
    assert pixel_coord.dim() == 2
    assert pixel_coord.shape[-1] == 2

    # H_inv from np.array to tensor
    H = torch.Tensor(H).float()

    # augmented coordinates
    pixel_coord = torch.cat(
        (pixel_coord, torch.ones(len(pixel_coord), 1)), dim=1)
    # pixel --> world
    world_coord = torch.matmul(H, pixel_coord.T).T
    # normalise world coordinates
    world_coord = world_coord[:, :2] / world_coord[:, 2].unsqueeze(1)
    return world_coord


def world2pixel_torch_multiagent(world_coord, H_inv):
    """
    Convert a seq_length*N_agents*2 Pytorch tensor of (x,y) world
    coordinates into a seq_length*N_agents*2 Pytorch tensor of (x,y) pixel
    coordinates.
    H_inv: 3x3 inverse homography matrix
    """
    assert isinstance(world_coord, torch.Tensor)
    assert world_coord.dim() == 3
    assert world_coord.shape[-1] == 2

    seq_length, N_agents, N_coord = world_coord.shape

    # H_inv from np.array to float tensor on specific device
    H_inv = torch.Tensor(H_inv).float().to(world_coord.device)

    # augmented coordinates
    world_coord = torch.cat(
        (world_coord, torch.ones(seq_length, N_agents, 1).to(
            world_coord.device)), dim=-1)
    # seq_length*N_agents*coord --> N_agents*coord*seq_length
    world_coord = world_coord.permute(1, 2, 0)
    # world --> pixel
    pixel_coord = torch.matmul(H_inv, world_coord)
    # N_agents*coord*seq_length --> seq_length*N_agents*coord
    pixel_coord = pixel_coord.permute(2, 0, 1)
    # normalise pixel coordinates
    pixel_coord = pixel_coord[:, :, 0:2] / pixel_coord[:, :, 2].unsqueeze(
        dim=-1)
    return pixel_coord


def pixel2world_torch_multiagent(pixel_coord, H):
    """
    Convert a seq_length*N_agents*2 Pytorch tensor of (x,y) pixel
    coordinates into a seq_length*N_agents*2 Pytorch tensor of (x,y) world
    coordinates.
    H: 3x3 homography matrix
    """
    assert isinstance(pixel_coord, torch.Tensor)
    assert pixel_coord.dim() == 3
    assert pixel_coord.shape[-1] == 2

    seq_length, N_agents, N_coord = pixel_coord.shape

    # H from np.array to float tensor on specific device
    H = torch.Tensor(H).float().to(pixel_coord.device)

    # augmented coordinates
    pixel_coord = torch.cat(
        (pixel_coord, torch.ones(seq_length, N_agents, 1).to(
            pixel_coord.device)), dim=-1)
    # seq_length*N_agents*coord --> N_agents*coord*seq_length
    pixel_coord = pixel_coord.permute(1, 2, 0)
    # pixel --> world
    world_coord = torch.matmul(H, pixel_coord.float())
    # N_agents*coord*seq_length --> seq_length*N_agents*coord
    world_coord = world_coord.permute(2, 0, 1)
    # normalise world coordinates
    world_coord = world_coord[:, :, 0:2] / world_coord[:, :, 2].unsqueeze(
        dim=-1)
    return world_coord
