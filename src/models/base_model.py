import torch

from src.losses import MSE_loss
from src.metrics import ADE_best_of, FDE_best_of, KDE_negative_log_likelihood


class Base_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_trainable = True

    def prepare_inputs(self, batch_data, batch_id):
        """
        Prepare inputs to be fed to a generic model.
        """
        # we need to remove first dimension which is added by torch.DataLoader
        # float is needed to convert to 32bit float

        entries_to_remove = ['input_traj_maps']
        for key in entries_to_remove:
            if key in batch_data:
                del batch_data[key]

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
        losses = {"traj_MSE_loss": 0}
        return losses

    def set_losses_coeffs(self):
        losses_coeffs = {"traj_MSE_loss": 1}
        return losses_coeffs

    def init_sample_losses(self):
        losses = self.init_losses()
        sample_losses = {k: [] for k in losses.keys()}
        return sample_losses

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
        }
        return best_metrics

    def best_valid_metric(self):
        return "ADE"

    def compute_loss_mask(self, seq_list, obs_length: int = 8):
        """
        Get a mask to denote whether to account predictions during loss
        computation. It is supposed to calculate losses for a person at
        time t only if his data exists from time 0 to time t.

        Parameters
        ----------
        seq_list : PyTorch tensor
            input is seq_list[1:]. Size = (seq_len,N_pedestrians). Boolean mask
            that is =1 if pedestrian i is present at time-step t.
        obs_length : int
            number of observation time-steps

        Returns
        -------
        loss_mask : PyTorch tensor
            Shape: (seq_len,N_pedestrians)
            loss_mask[t,i] = 1 if pedestrian i if present from beginning till time t
        """
        loss_mask = seq_list.cumprod(dim=0)
        # we should not compute losses for step 0, as ground_truth and
        # predictions are always equal there
        loss_mask[0] = 0
        return loss_mask

    def compute_model_losses(self,
                             outputs,
                             ground_truth,
                             loss_mask,
                             inputs,
                             aux_outputs):
        """
        Compute loss for a generic model.
        """
        model_loss = MSE_loss(outputs, ground_truth, loss_mask)
        losses = {"traj_MSE_loss": model_loss}
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
            return FDE_best_of(
                predictions, ground_truth, metric_mask, obs_length)
        if metric_name == 'ADE_world':
            return ADE_best_of(
                pred_world, GT_world, metric_mask, obs_length)
        elif metric_name == 'FDE_world':
            return FDE_best_of(
                pred_world, GT_world, metric_mask, obs_length)
        elif metric_name == 'NLL':
            if compute_nll and num_samples > 1:
                return KDE_negative_log_likelihood(
                    predictions, ground_truth, metric_mask, obs_length)
            else:
                return [0, 0, 0]
        else:
            raise ValueError("This metric has not been implemented yet!")

    def forward(self, inputs, num_samples=1, if_test=False):
        """Signature of a forward pass in a generic model"""
        raise NotImplementedError
