import torch
import torch.nn as nn


def MSE_loss(outputs, ground_truth, loss_mask):
    """
    Compute an averaged Mean-Squared-Error, only on the positions
    in which loss_mask is True.
    outputs.shape = num_samples, seq_length, num_agent, n_coords(2)
    """
    squared_error = (outputs - ground_truth)**2
    # sum over coordinates --> Shape becomes: num_samples * seq_len * n_agents
    squared_error = torch.sum(squared_error, dim=-1)

    # compute error only where mask is True
    loss = squared_error * loss_mask

    # Take a weighted loss, but only on places where loss_mask=True.
    # Divide by loss_mask.sum() instead of seq_len*N_pedestrians (or mean).
    # This is an average loss per time-step per pedestrian.
    loss = loss.sum(dim=-1).sum(dim=-1) / loss_mask.sum()

    # minimum loss over samples (only 1 sample during training)
    loss, _ = loss.min(dim=0)

    return loss


def Goal_BCE_loss(logit_map, goal_map_GT, loss_mask):
    """
    Compute the Binary Cross-Entropy loss for the probability distribution
    of the goal. Prediction and GT are two pixel maps.
    """
    losses_samples = []
    for logit_map_sample_i in logit_map:
        loss = BCE_loss_sample(logit_map_sample_i, goal_map_GT, loss_mask)
        losses_samples.append(loss)
    losses_samples = torch.stack(losses_samples)

    # minimum loss over samples (only 1 sample during training)
    loss, _ = losses_samples.min(dim=0)

    return loss


def BCE_loss_sample(logit_map, goal_map_GT, loss_mask):
    """
    Compute the Binary Cross-Entropy loss for the probability distribution
    of the goal maps. logit_map is a logit map, goal_map_GT a probability map.
    """
    batch_size, T, H, W = logit_map.shape
    # reshape across space and time
    output_reshaped = logit_map.view(batch_size, -1)
    target_reshaped = goal_map_GT.view(batch_size, -1)

    # takes as input computed logit and GT probabilities
    BCE_criterion = nn.BCEWithLogitsLoss(reduction='none')

    # compute the Goal CE loss for each agent and sample
    loss = BCE_criterion(output_reshaped, target_reshaped)

    # Mean over maps (T, W, H)
    loss = loss.mean(dim=-1)

    # Take a weighted loss, but only on places where loss_mask=True
    # Divide by full_agents.sum() instead of seq_len*N_pedestrians (or mean)
    full_agents = loss_mask[-1]
    loss = (loss * full_agents).sum(dim=0) / full_agents.sum()

    return loss
