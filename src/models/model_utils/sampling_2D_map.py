from typing import Optional

import torch
import numpy as np

from src.models.model_utils.kmeans import kmeans


def normalize_prob_map(x):
    """Normalize a probability map of shape (B, T, H, W) so
    that sum over H and W equal ones"""
    assert len(x.shape) == 4
    sums = x.sum(-1, keepdim=True).sum(-2, keepdim=True)
    x = torch.divide(x, sums)
    return x


def un_normalize_prob_map(x):
    """Un-Normalize a probability map of shape (B, T, H, W) so
    that each pixel has value between 0 and 1"""
    assert len(x.shape) == 4
    (B, T, H, W) = x.shape
    maxs, _ = x.reshape(B, T, -1).max(-1)
    x = torch.divide(x, maxs.unsqueeze(-1).unsqueeze(-1))
    return x


def create_meshgrid(
        x: torch.Tensor,
        normalized_coordinates: Optional[bool]) -> torch.Tensor:
    assert len(x.shape) == 4, x.shape
    _, _, height, width = x.shape
    _device, _dtype = x.device, x.dtype
    if normalized_coordinates:
        xs = torch.linspace(-1.0, 1.0, width, device=_device, dtype=_dtype)
        ys = torch.linspace(-1.0, 1.0, height, device=_device, dtype=_dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=_device, dtype=_dtype)
        ys = torch.linspace(0, height - 1, height, device=_device, dtype=_dtype)
    return torch.meshgrid(ys, xs)  # pos_y, pos_x


def mean_over_map(x):
    """
    Mean over map: Input is a batched image of shape (B, T, H, W)
    where sigmoid/softmax is already performed (not logits).
    Output shape is (B, T, 2).
    """
    (B, T, H, W) = x.shape
    x = normalize_prob_map(x)
    pos_y, pos_x = create_meshgrid(x, normalized_coordinates=False)
    x = x.view(B, T, -1)

    estimated_x = torch.sum(pos_x.reshape(-1) * x, dim=-1, keepdim=True)
    estimated_y = torch.sum(pos_y.reshape(-1) * x, dim=-1, keepdim=True)
    mean_coords = torch.cat([estimated_x, estimated_y], dim=-1)
    return mean_coords


def argmax_over_map(x):
    """
    From probability maps of shape (B, T, H, W), extract the
    coordinates of the maximum values (i.e. argmax).
    Hint: you need to use numpy.amax
    Output shape is (B, T, 2)
    """

    def indexFunc(array, item):
        for idx, val in np.ndenumerate(array):
            if val == item:
                return idx

    B, T, _, _ = x.shape
    device = x.device
    x = x.detach().cpu().numpy()
    maxVals = np.amax(x, axis=(2, 3))
    max_indices = np.zeros((B, T, 2), dtype=np.int64)
    for index in np.ndindex(x.shape[0], x.shape[1]):
        max_indices[index] = np.asarray(
            indexFunc(x[index], maxVals[index]), dtype=np.int64)[::-1]
    max_indices = torch.from_numpy(max_indices)
    return max_indices.to(device)


def sampling(probability_map,
             num_samples=10000,
             rel_threshold=0.05,
             replacement=True):
    """Given probability maps of shape (B, T, H, W) sample
    num_samples points for each B and T"""
    # new view that has shape=[batch*timestep, H*W]
    prob_map = probability_map.view(probability_map.size(0) * probability_map.size(1), -1)
    if rel_threshold is not None:
        # exclude points with very low probability
        thresh_values = prob_map.max(dim=1)[0].unsqueeze(1).expand(-1, prob_map.size(1))
        mask = prob_map < thresh_values * rel_threshold
        prob_map = prob_map * (~mask).int()
        prob_map = prob_map / prob_map.sum()

    # samples.shape=[batch*timestep, num_samples]
    samples = torch.multinomial(prob_map,
                                num_samples=num_samples,
                                replacement=replacement)

    # unravel sampled idx into coordinates of shape [batch, time, sample, 2]
    samples = samples.view(probability_map.size(0), probability_map.size(1), -1)
    idx = samples.unsqueeze(3)
    preds = idx.repeat(1, 1, 1, 2).float()
    preds[:, :, :, 0] = (preds[:, :, :, 0]) % probability_map.size(3)
    preds[:, :, :, 1] = torch.floor((preds[:, :, :, 1]) / probability_map.size(3))
    return preds


def TTST_test_time_sampling_trick(x, num_goals, device):
    """
    From a probability map of shape (B, 1, H, W), sample num_goals
    goals so that they cover most of the space (thanks to k-means).
    Output shape is (num_goals, B, 1, 2).
    """
    assert x.shape[1] == 1
    # first sample is argmax sample
    num_clusters = num_goals - 1
    goal_samples_argmax = argmax_over_map(x)

    # sample a large amount of goals to be clustered
    goal_samples = sampling(x[:, 0:1], num_samples=10000)
    # from (B, 1, num_samples, 2) to (num_samples, B, 1, 2)
    goal_samples = goal_samples.permute(2, 0, 1, 3)

    # Iterate through all person/batch_num, as this k-Means implementation
    # doesn't support batched clustering
    goal_samples_list = []
    for person in range(goal_samples.shape[1]):
        goal_sample = goal_samples[:, person, 0]

        # Actual k-means clustering, Outputs:
        # cluster_ids_x -  Information to which cluster_idx each point belongs
        # to cluster_centers - list of centroids, which are our new goal samples
        cluster_ids_x, cluster_centers = kmeans(X=goal_sample,
                                                num_clusters=num_clusters,
                                                distance='euclidean',
                                                device=device, tqdm_flag=False,
                                                tol=0.001, iter_limit=1000)
        goal_samples_list.append(cluster_centers)

    goal_samples = torch.stack(goal_samples_list).permute(1, 0, 2).unsqueeze(2)
    goal_samples = torch.cat([goal_samples_argmax.unsqueeze(0), goal_samples],
                             dim=0)
    return goal_samples
