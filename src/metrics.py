import torch
import numpy as np
from scipy.stats import gaussian_kde


def compute_metric_mask(seq_list):
    """
    Get a mask to denote whether to account predictions during metrics
    computation. It is supposed to calculate metrics only for pedestrians
    fully present during observation and prediction time-steps.

    Parameters
    ----------
    seq_list : PyTorch tensor
        Size = (seq_len,N_pedestrians).
        Boolean mask that is =1 if pedestrian i is present at time-step t.

    Returns
    -------
    metric_mask : PyTorch tensor
        Shape: (N_pedestrians,)
        metric_mask[i] = 1 if pedestrian i if fully present during
        observation and prediction time-steps.
    """
    metric_mask = seq_list.cumprod(dim=0)
    # fully present on the whole seq_length interval
    metric_mask = metric_mask[-1] > 0
    return metric_mask


def check_metrics_inputs(predictions,
                         ground_truth,
                         metric_mask):
    num_sample, seq_length, N_agents, num_coords = predictions.shape
    # assert data shape
    assert len(predictions.shape) == 4, \
        f"Expected 4D (MxTxNxC) array for predictions, got {predictions.shape}"
    assert ground_truth.shape == (seq_length, N_agents, num_coords), \
        f"Expected 3D (TxNxC) array for ground_truth, got {ground_truth.shape}"
    assert metric_mask.shape == (N_agents,), \
        f"Expected 1D (N) array for metric_mask, got {metric_mask.shape}"

    # assert all data is valid
    assert torch.isfinite(predictions).all(), \
        "Invalid value found in predictions"
    assert torch.isfinite(ground_truth).all(), \
        "Invalid value found in ground_truth"
    assert torch.isfinite(metric_mask).all(), \
        "Invalid value found in metric_mask"


def ADE_best_of(predictions: torch.Tensor,
                ground_truth: torch.Tensor,
                metric_mask: torch.Tensor,
                obs_length: int = 8) -> list:
    """
    Compute ADE metric - Best-of-K selected.
    The best ADE from the samples is selected, based on the best ADE.
    Torch implementation. Returns a list of floats, one for each full
    example in the batch.

    Parameters
    ----------
    predictions : torch.Tensor
        The output trajectories of the prediction model.
        Shape: num_sample * seq_len * N_pedestrians * (x,y)
    ground_truth : torch.Tensor
        The target trajectories.
        Shape: seq_len * N_pedestrians * (x,y)
    metric_mask : torch.Tensor
        Mask to denote if pedestrians are fully present.
        Shape: (N_pedestrians,)
    obs_length : int
        Number of observation time-steps

    Returns
    ----------
    ADE_error : list of float
        Average displacement error
    """
    check_metrics_inputs(predictions, ground_truth, metric_mask)

    # l2-norm for each time-step
    error = torch.norm(predictions - ground_truth, p=2, dim=3)
    # only calculate for fully present pedestrians
    error_full = error[:, obs_length:, metric_mask]

    # mean over time-steps to find best ADE
    error_full_mean = torch.mean(error_full, dim=1)
    # min ADE over samples
    error_full_mean_min, _ = torch.min(error_full_mean, dim=0)  # ADE

    return error_full_mean_min.tolist()


def FDE_best_of(predictions: torch.Tensor,
                ground_truth: torch.Tensor,
                metric_mask: torch.Tensor,
                obs_length: int = 8) -> tuple:
    """
    Compute FDE metric - Best-of-K selected.
    The best FDE from the samples is selected, based on the best ADE.
    Torch implementation. Returns a list of floats, one for each full
    example in the batch.

    Parameters
    ----------
    predictions : torch.Tensor
        The output trajectories of the prediction model.
        Shape: num_sample * seq_len * N_pedestrians * (x,y)
    ground_truth : torch.Tensor
        The target trajectories.
        Shape: seq_len * N_pedestrians * (x,y)
    metric_mask : torch.Tensor
        Mask to denote if pedestrians are fully present.
        Shape: (N_pedestrians,)
    obs_length : int
        Number of observation time-steps

    Returns
    ----------
    FDE : list of float
        Final displacement error
    """
    check_metrics_inputs(predictions, ground_truth, metric_mask)

    # l2-norm for each time-step
    error = torch.norm(predictions - ground_truth, p=2, dim=3)
    # only calculate for fully present pedestrians
    error_full = error[:, -1, metric_mask]

    # best error over samples
    final_error, _ = error_full.min(dim=0)

    return final_error.tolist()


def KDE_negative_log_likelihood(predictions: torch.Tensor,
                                ground_truth: torch.Tensor,
                                metric_mask: torch.Tensor,
                                obs_length: int = 8,
                                log_pdf_lower_bound: int = -20) -> list:
    """
    Kernel Density Estimation-based Negative Log-Likelihood metric (KDE-NLL).
    Computes KDE-NLL for a batch of agents. Returns a list of floats, one for
    each full agents in the batch.
    Only works in a stochastic settings (i.e. multi-future predictions).
    Numpy-Scipy implementation.

    Parameters
    ----------
    predictions : PyTorch tensor
        The output trajectories of the prediction model.
        Shape: num_sample * seq_len * N_pedestrians * (x,y)
    ground_truth : PyTorch tensor
        The ground_truth trajectories.
        Shape: seq_len * N_pedestrians * (x,y)
    metric_mask : torch.Tensor
        Mask to denote if pedestrians are fully present.
        Shape: (N_pedestrians,)
    obs_length : int
        Number of observation time-steps
    log_pdf_lower_bound : int
        Lower bound for log pdf

    Returns
    ----------
    total_nll : list of float
        List negative log likelihoods (floats).
        len(total_nll): number of pedestrians in the batch with a
        complete ground truth trajectory.
    """
    check_metrics_inputs(predictions, ground_truth, metric_mask)

    # only calculate for fully present pedestrians
    predictions_full = predictions[:, obs_length:, metric_mask].\
        permute(2, 0, 1, 3).cpu().numpy()
    ground_truth_full = ground_truth[obs_length:, metric_mask].\
        permute(1, 0, 2).cpu().numpy()

    pred_length = predictions_full.shape[-2]
    total_nll = []

    # for each full agent
    for agent_i in range(predictions_full.shape[0]):
        ground_truth = ground_truth_full[agent_i]  # [pred_length, 2]
        predictions = np.swapaxes(predictions_full[agent_i], 0, 1)
        # [pred_length, num_samples, 2]

        ll = 0.0  # log-likelihood initialization
        same_pred = 0  # number of equal predictions
        # for each timestep
        for timestep in range(pred_length):
            # current ground-truth (x,y)
            curr_gt = ground_truth[timestep]
            # if all identical prediction at particular time-step, skip
            if np.all(predictions[timestep, 1:] == predictions[timestep, :-1]):
                same_pred += 1
                continue
            try:
                # Gaussian KDE
                scipy_kde = gaussian_kde(predictions[timestep].T)
                # We need [0] because it's a (1,)-shaped numpy array.
                log_pdf = np.clip(scipy_kde.logpdf(curr_gt.T),
                                  a_min=log_pdf_lower_bound, a_max=None)[0]
                if np.isnan(log_pdf) or np.isinf(log_pdf) or log_pdf > 100:
                    # difficulties in computing Gaussian_KDE
                    same_pred += 1
                    continue
                ll += log_pdf
            except:  # difficulties in computing Gaussian_KDE
                same_pred += 1

        if same_pred == pred_length:
            raise Exception('LL error: all predictions are identical.')

        nll = - ll / (pred_length - same_pred)
        total_nll.append(nll)
    return total_nll


def FDE_best_of_goal(all_aux_outputs: list,
                     ground_truth: torch.Tensor,
                     metric_mask: torch.Tensor,
                     args,
                     ) -> list:
    """
    Compute the best of 20 FDE metric between the final position GT and
    the predicted goal.
    Works with a goal architecture model only.
    Returns a list of float, FDE errors for each full pedestrians in
    the batch.
    """
    # take only last temporal step (final destination)
    ground_truth = ground_truth[-1]
    end_point_pred = all_aux_outputs["goal_point"]
    end_point_pred = end_point_pred.to(ground_truth.device) * args.down_factor

    # difference
    FDE_error = ((end_point_pred - ground_truth)**2).sum(-1) ** 0.5

    # take minimum over samples
    # take only agents with full trajectories
    best_error_full, _ = FDE_error[:, metric_mask].min(dim=0)

    return best_error_full.flatten().cpu().tolist()


def FDE_best_of_goal_world(all_aux_outputs: list,
                           scene,
                           ground_truth: torch.Tensor,
                           metric_mask: torch.Tensor,
                           args,
                           ) -> list:
    """
    Compute the best of 20 FDE metric between the final position GT and
    the predicted goal.
    Returns a list of float, FDE errors for each full pedestrians in
    the batch.
    """
    # take only last temporal step (final destination)
    ground_truth = ground_truth[-1]
    end_point_pred = all_aux_outputs["goal_point"]
    end_point_pred = end_point_pred.to(ground_truth.device) * args.down_factor
    # from pixel to world coordinates
    end_point_pred = scene.make_world_coord_torch(end_point_pred)

    # difference
    FDE_error = ((end_point_pred - ground_truth)**2).sum(-1) ** 0.5

    # take minimum over samples
    # take only agents with full trajectories
    best_error_full, _ = FDE_error[:, metric_mask].min(dim=0)

    return best_error_full.flatten().cpu().tolist()
