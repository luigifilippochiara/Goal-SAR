import os
import time

import matplotlib.pyplot as plt

import torch
import numpy as np
from tqdm import tqdm
import wandb

from src.data_loader import get_dataloader
from src.metrics import compute_metric_mask
from src.utils import add_dict_prefix, formatted_time, print_model_summary, \
    find_trainable_layers, maybe_makedir


class trainer(object):
    def __init__(self, args):
        self.args = args
        # initialize data loaders
        self.data_loaders = dict()
        for set_name in ['train', 'valid', 'test']:
            self.data_loaders[set_name] = get_dataloader(args, set_name=set_name)
        # initialize device
        self.device = self._set_device()
        # initialize network
        self.net = self._initialize_network(args.model_name).to(self.device)

        # Prepare log curve file and initialize best validation metrics
        self.log_curve_file = os.path.join(self.args.model_dir, 'log_curve.txt')

        # Best metrics
        self.best_metrics = self.net.init_best_metrics()
        self.best_metrics_epochs = {k: -1 for k in self.best_metrics.keys()}

    def _initialize_network(self, model_name: str):
        """
        Import and initialize the requested model
        """
        if model_name == 'SAR':
            from src.models.sar import SAR
            net = SAR(self.args, self.device)
        elif model_name == 'Goal_SAR':
            from src.models.goal_sar import Goal_SAR
            net = Goal_SAR(self.args, self.device)
        else:
            raise NotImplementedError(
                f"Model architecture {model_name} does not exist yet.")

        # save net architecture to txt
        with open(os.path.join(self.args.model_dir, 'net.txt'), 'w') as f:
            f.write(str(net))

        return net

    def _set_optimizer(self, optimizer_name: str, parameters):
        """
        Set selected optimizer
        """
        if optimizer_name == 'Adam':
            return torch.optim.Adam(
                parameters, lr=self.args.learning_rate)
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(
                parameters, lr=self.args.learning_rate)
        else:
            raise NameError(f'Optimizer {optimizer_name} not implemented.')

    def _set_scheduler(self, scheduler_name: str):
        """
        Set selected scheduler
        """
        # ReduceOnPlateau
        if scheduler_name == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=50,
                min_lr=1e-6,
                verbose=True)
        # Exponential
        elif scheduler_name == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.99)
        # CosineAnnealing
        elif scheduler_name == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=30)
        else:  # Not set
            return None

    def _save_checkpoint(self, epoch, best_epoch=False):
        """
        Save model and optimizer states
        """
        saved_models_path = os.path.join(self.args.model_dir, 'saved_models')
        if not os.path.exists(saved_models_path):
            os.makedirs(saved_models_path)
        # Save current checkpoint
        if not best_epoch:
            saved_model_name = os.path.join(
                saved_models_path,
                self.args.model_name + '_epoch_' +
                str(epoch).zfill(3) + '.pt')
        else:  # best model name
            saved_model_name = os.path.join(
                saved_models_path,
                self.args.model_name + '_best_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
        }, saved_model_name)

    def _load_checkpoint(self, load_checkpoint):
        """
        Load a pre-trained model. Can then be used to test or resume training.
        """
        if load_checkpoint is not None:
            # Create load model path
            if load_checkpoint == 'best':
                saved_model_name = os.path.join(
                    self.args.model_dir, 'saved_models',
                    self.args.model_name + '_best_model.pt')
            else:  # Load specific checkpoint
                assert int(load_checkpoint) > 0, \
                    "Check args.load_model. Must be an integer > 0"
                saved_model_name = os.path.join(
                    self.args.model_dir, 'saved_models',
                    self.args.model_name + '_epoch_' +
                    str(load_checkpoint).zfill(3) + '.pt')
            print("\nSaved model path:", saved_model_name)
            # Load model
            if os.path.isfile(saved_model_name):
                print('Loading checkpoint ...')
                checkpoint = torch.load(saved_model_name,
                                        map_location=self.device)
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(
                    checkpoint['model_state_dict'])
                print('Loaded checkpoint at epoch', model_epoch, '\n')
                return model_epoch
            else:
                raise ValueError("No such pre-trained model:", saved_model_name)
        else:
            raise ValueError('You need to specify an epoch (int) if you want '
                             'to load a model or "best" to load the best '
                             'model! Check args.load_checkpoint')

    def _load_or_restart(self):
        """
        Load a pre-trained model to resume training or restart from scratch.
        Can start from scratch or resume training, depending on input
        self.args.load_checkpoint parameter.
        """
        # load pre-trained model to resume training
        if self.args.load_checkpoint is not None:
            loaded_epoch = self._load_checkpoint(self.args.load_checkpoint)
            # start from the following epoch
            start_epoch = int(loaded_epoch) + 1
        else:
            start_epoch = 1
            # log_file header only the first time
            with open(self.log_curve_file, 'w') as f:
                f.write("epoch,learning_rate,"
                        "train_ADE,train_FDE,"
                        "valid_ADE,valid_FDE," +
                        ",".join(sorted(self.net.init_losses().keys())) +
                        "\n")
        return start_epoch

    def test(self, load_checkpoint):
        """
        Load a trained model and test it on the test set.
        """
        print('*** Test phase started ***')
        # some models do not need to be trained nor loaded
        if self.net.is_trainable:
            best_epoch = self._load_checkpoint(load_checkpoint)
        else:
            best_epoch = load_checkpoint
        print('Testing ...')
        total_results = []
        for run_idx in range(self.args.num_test_runs):
            print(f"\nTest run #{run_idx} ...")
            test_losses, test_metrics = self._evaluate_epoch(best_epoch,
                                                             mode='test')
            run_results = dict(**test_losses, **test_metrics)
            # print losses and metrics for run i
            print(f'Test_set: {self.args.test_set},',
                  f'test_run_idx: {run_idx},',
                  f'epoch: {load_checkpoint},',
                  ', '.join([f"{k}={v:.5f}" for k, v in run_results.items()]))
            total_results.append(run_results)
        average_results = {k: np.mean([i[k] for i in total_results])
                           for k in run_results}
        # print average losses and metrics for
        print("\n" + "#"*25)
        print("#"*5 + " FINAL RESULTS " + "#"*5)
        print("#" * 25)
        print(f'Test_set: {self.args.test_set},',
              f'epoch: {load_checkpoint},',
              ', '.join([f"{k}={v:.5f}" for k, v in average_results.items()]))

    def train(self):
        """
        Train the model. Wrapper for train_loop.
        """
        # find where to start
        start_epoch = self._load_or_restart()

        # print model info
        print_model_summary(self.net, self.args.model_name)

        # parameters to update
        params = find_trainable_layers(self.net)

        # Set optimizer
        self.optimizer = self._set_optimizer(self.args.optimizer, params)
        # Set scheduler
        self.scheduler = self._set_scheduler(self.args.scheduler)

        # start training
        self._train_loop(start_epoch=start_epoch,
                         end_epoch=self.args.num_epochs)

    def train_test(self):
        """
        Perform training and then test on the best validation epoch.
        """
        self.train()
        print()
        self.test(load_checkpoint='best')

    def _train_loop(self, start_epoch, end_epoch):
        """
        Train the model. Loop over the epochs, train and update network
        parameters, save model, check results on validation set,
        print results and save log data.
        """
        # saved metrics before validation begins
        valid_metrics = {"valid_ADE": 0, "valid_FDE": 0}

        # initial learning rate
        if self.scheduler is not None:
            learning_rate = self.optimizer.param_groups[0]['lr']
        else:
            learning_rate = self.args.learning_rate

        phase_name = 'Train'
        best_metric_name = self.net.best_valid_metric()

        print(f'*** {phase_name} phase started ***')
        print(f"Starting epoch: {start_epoch}, final epoch: {end_epoch}")

        if self.args.use_wandb:
            wandb.init(settings=wandb.Settings(start_method="thread"),
                       project="GoalSAR", config=self.args, entity="vimp_traj",
                       group=f"{self.args.dataset}",
                       job_type=f"{self.args.test_set}",
                       tags=None, name=None)

        for epoch in range(start_epoch, end_epoch + 1):
            start_time = time.time()  # time epoch
            train_losses, train_metrics = self._train_epoch(epoch)

            if epoch % self.args.save_every == 0:
                self._save_checkpoint(epoch)  # save model
                print(f"Saved checkpoint at epoch {epoch}")

            # validation
            if epoch >= self.args.start_validation and \
                    epoch % self.args.validate_every == 0:
                valid_losses, valid_metrics = self._evaluate_epoch(epoch,
                                                                   mode='valid')

                # comment some of this print, if it is too long
                print(f'----Epoch {epoch},',
                      f'time/epoch={formatted_time(time.time() - start_time)},',
                      f'learning_rate={learning_rate:.5f},',
                      ', '.join([f"{loss_name}={loss_value:.5f}" for
                                 loss_name, loss_value in
                                 train_losses.items()]) + ',',
                      ', '.join([f"{loss_name}={loss_value:.5f}" for
                                 loss_name, loss_value in
                                 valid_losses.items()]) + ',',
                      ', '.join([f"{metric_name}={metric_value:.3f}" for
                                 metric_name, metric_value in
                                 train_metrics.items()]) + ',',
                      ', '.join([f"{metric_name}={metric_value:.3f}" for
                                 metric_name, metric_value in
                                 valid_metrics.items()]))

                # update best model and best metrics
                for k, v in self.best_metrics.items():
                    try:
                        current_metric_loss = valid_metrics["valid_" + k]
                    except KeyError:
                        current_metric_loss = valid_losses["valid_" + k]

                    if current_metric_loss < v:
                        self.best_metrics[k] = current_metric_loss
                        self.best_metrics_epochs[k] = epoch
                        # save best model on best metric
                        if k == best_metric_name:
                            self._save_checkpoint(epoch, best_epoch=True)
                            print(f"Saved best model at epoch {epoch}")

                print(', '.join([f"best_{metric_name}={metric_value:.3f}" for
                                 metric_name, metric_value in
                                 self.best_metrics.items()]) + ',',
                      ', '.join([f"best_epoch_{metric_name}="
                                 f"{metric_epoch}" for
                                 metric_name, metric_epoch in
                                 self.best_metrics_epochs.items()]))
            else:
                print(f'----Epoch {epoch},',
                      f'time/epoch={formatted_time(time.time() - start_time)},',
                      f'learning_rate={learning_rate:.5f},',
                      ', '.join([f"{loss_name}={loss_value:.5f}" for
                                 loss_name, loss_value in
                                 train_losses.items()]) + ',',
                      ', '.join([f"{metric_name}={metric_value:.3f}" for
                                 metric_name, metric_value in
                                 train_metrics.items()]))

            # Update learning rate value
            if self.scheduler is not None:
                if self.args.scheduler == 'ReduceLROnPlateau':
                    if epoch >= self.args.start_validation and \
                            epoch % self.args.validate_every == 0:
                        lr_sched_metric = valid_metrics["valid_ADE"]
                        self.scheduler.step(lr_sched_metric)
                elif self.args.scheduler in ['ExponentialLR',
                                             'CosineAnnealingLR']:
                    self.scheduler.step()
                learning_rate = self.optimizer.param_groups[0]['lr']

            # save metrics to log_curve.txt
            with open(self.log_curve_file, 'a') as f:
                f.write(','.join(str(m) for m in [
                    epoch, learning_rate,
                    train_metrics["train_ADE"],
                    train_metrics["train_FDE"],
                    valid_metrics["valid_ADE"],
                    valid_metrics["valid_FDE"]] +
                    [train_losses[loss_name] for loss_name in sorted(
                        train_losses)]) + '\n')

            # save metrics to WandB
            if self.args.use_wandb and epoch >= self.args.start_validation:
                wandb.log({'learning_rate': learning_rate}, step=epoch)
                wandb.log(train_losses, step=epoch)
                wandb.log(train_metrics, step=epoch)
                # validation
                if epoch >= self.args.start_validation and \
                        epoch % self.args.validate_every == 0:
                    wandb.log(valid_losses, step=epoch)
                    wandb.log(valid_metrics, step=epoch)
                    # best metrics
                    for k, v in self.best_metrics.items():
                        wandb.run.summary["best_" + k] = v
                    for k, v in self.best_metrics_epochs.items():
                        wandb.run.summary["best_epoch_" + k] = v

    def _train_epoch(self, epoch):
        """
        Train one epoch of the model on the whole training set.
        """
        self.net.train()  # train mode

        # INIT LOSSES and METRICS
        losses_epoch = self.net.init_losses()
        losses_coeffs = self.net.set_losses_coeffs()
        metrics_epoch = self.net.init_train_metrics()

        # Progress bar
        train_bar = tqdm(self.data_loaders['train'], ascii=True, ncols=100,
                         desc=f'Epoch {epoch}. Train batches')
        num_train_batches = len(self.data_loaders['train'])

        if_plot = True

        for batch_data, batch_id in train_bar:

            inputs, ground_truth, seq_list = \
                self.net.prepare_inputs(batch_data, batch_id)
            del batch_data

            self.optimizer.zero_grad()  # sets grads to zero

            # Forward network
            all_outputs, all_aux_outputs = self.net.forward(inputs, if_test=False)

            # compute loss_mask
            loss_mask = self.net.compute_loss_mask(
                seq_list, self.args.obs_length).to(self.device)

            # compute metric_mask
            metric_mask = compute_metric_mask(seq_list)

            # compute model losses
            losses = self.net.compute_model_losses(
                all_outputs, ground_truth, loss_mask, inputs, all_aux_outputs)

            # overall loss
            loss = torch.zeros(1).to(self.device)
            for loss_name, loss_value in losses.items():
                loss += losses_coeffs[loss_name]*loss_value
                losses_epoch[loss_name] += loss_value.item()

            # Update network weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()

            if self.args.use_wandb and if_plot:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(inputs['tensor_image'][0:3].permute(1, 2, 0).
                          cpu().detach().numpy())
                ax.plot(all_outputs[0][:, 0, 0].detach().cpu().numpy(),
                        all_outputs[0][:, 0, 1].detach().cpu().numpy(),
                        "-o")
                ax.plot(ground_truth[:, 0, 0].detach().cpu().numpy(),
                        ground_truth[:, 0, 1].detach().cpu().numpy(),
                        color="white", label='GT')
                plt.legend(loc='best')
                maybe_makedir("images")
                plt.savefig('images/train_traj.png')
                wandb.log({"train/traj_image": wandb.Image(fig)}, step=epoch)
                plt.close()

                if "goal_logit_map_goal_0" in all_aux_outputs.keys():
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    prob_maps_GT = inputs["input_traj_maps"][:, self.args.obs_length:].detach().cpu().numpy()
                    prob_map = torch.sigmoid(all_aux_outputs[
                        f"goal_logit_map_goal_0"]).detach().cpu().numpy()
                    ax[0].imshow(prob_maps_GT[0, -1])
                    ax[1].imshow(prob_map[0, 0, -1])
                    plt.savefig('images/train_goal.png')
                    wandb.log({"train/goal_image": wandb.Image(fig)},
                              step=epoch)
                    plt.close()

            # update train metrics. Not used during backprop, only print and log
            with torch.no_grad():
                for metric_name in metrics_epoch.keys():
                    metrics_epoch[metric_name].extend(
                        self.net.compute_model_metrics(
                            metric_name=metric_name,
                            phase='train',
                            predictions=all_outputs,
                            ground_truth=ground_truth,
                            metric_mask=metric_mask,
                            all_aux_outputs=all_aux_outputs,
                            inputs=inputs,
                            obs_length=self.args.obs_length,
                        ))

            del inputs, all_outputs, all_aux_outputs, batch_id
            del ground_truth, seq_list, loss_mask, metric_mask, losses, loss
            if_plot = False

        # update losses
        for loss_name in losses_epoch.keys():
            # mean losses over batches
            losses_epoch[loss_name] = losses_epoch[loss_name] / num_train_batches
        losses_epoch = add_dict_prefix(losses_epoch, prefix='train')

        # update metrics
        train_metrics = {}
        for metric_name in metrics_epoch.keys():
            array_metric = np.array(metrics_epoch[metric_name])
            train_metrics[metric_name] = array_metric.mean()
            if "goal" in metric_name:
                train_metrics[metric_name + '_std'] = array_metric.std()
        train_metrics = add_dict_prefix(train_metrics, prefix='train')

        return losses_epoch, train_metrics

    @torch.no_grad()
    def _evaluate_epoch(self, epoch, mode='valid'):
        """
        Loop over the validation or test set once. Compute metrics and save
        output trajectories.
        """
        self.net.eval()  # evaluation mode

        # INIT LOSSES and METRICS
        losses_epoch = self.net.init_losses()
        metrics_epoch = self.net.init_test_metrics()

        # Progress bar
        evaluate_bar = tqdm(self.data_loaders[mode], ascii=True, ncols=100,
                            desc=f'Epoch {epoch}. {mode.title()} batches')
        num_evaluate_batches = len(self.data_loaders[mode])

        # num_samples
        if mode == 'valid':
            num_samples = self.args.num_valid_samples
        elif mode == 'test':
            num_samples = self.args.num_test_samples
        else:
            raise ValueError("mode must be in ['valid', 'test']")

        if_plot = True

        # loop over batches
        for batch_data, batch_id in evaluate_bar:

            inputs, ground_truth, seq_list = \
                self.net.prepare_inputs(batch_data, batch_id)
            del batch_data

            # compute loss_mask
            loss_mask = self.net.compute_loss_mask(
                seq_list, self.args.obs_length).to(self.device)

            # compute metric_mask
            metric_mask = compute_metric_mask(seq_list)

            all_output, all_aux_outputs = self.net.forward(
                inputs, num_samples=num_samples, if_test=True)

            # compute losses
            losses = self.net.compute_model_losses(
                all_output, ground_truth, loss_mask,
                inputs, all_aux_outputs)

            # overall loss
            for loss_name, loss_value in losses.items():
                losses_epoch[loss_name] += loss_value.item()

            if self.args.use_wandb and if_plot and mode == 'valid':
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(inputs['tensor_image'][0:3].permute(1, 2, 0).\
                          detach().cpu().numpy())
                # plot with 20 different colors
                cm = plt.get_cmap('tab20')
                NUM_COLORS = 20
                ax.set_prop_cycle(
                    color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
                for outputs in all_output:
                    ax.plot(outputs[:, 0, 0].detach().cpu().numpy(),
                            outputs[:, 0, 1].detach().cpu().numpy(),
                            "-o")
                ax.plot(ground_truth[:, 0, 0].detach().cpu().numpy(),
                        ground_truth[:, 0, 1].detach().cpu().numpy(),
                        color="white", label='GT')
                plt.legend(loc='best')
                plt.savefig('images/test_traj.png')
                wandb.log({"test/traj_image": wandb.Image(fig)}, step=epoch)
                plt.close()

                if "goal_logit_map_goal_0" in all_aux_outputs.keys():
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    prob_maps_GT = inputs["input_traj_maps"][:, self.args.obs_length:].detach().cpu().numpy()
                    prob_map = torch.sigmoid(all_aux_outputs[
                        f"goal_logit_map_goal_0"]).detach().cpu().numpy()
                    ax[0].imshow(prob_maps_GT[0, -1])
                    ax[1].imshow(prob_map[0, 0, -1])
                    plt.savefig('images/test_goal.png')
                    wandb.log({"test/goal_image": wandb.Image(fig)},
                              step=epoch)
                    plt.close()

            # update metrics
            for metric_name in metrics_epoch.keys():
                metrics_epoch[metric_name].extend(
                    self.net.compute_model_metrics(
                        metric_name=metric_name,
                        phase=mode,
                        predictions=all_output,
                        ground_truth=ground_truth,
                        metric_mask=metric_mask,
                        all_aux_outputs=all_aux_outputs,
                        inputs=inputs,
                        obs_length=self.args.obs_length,
                    ))

            del inputs, batch_id, all_output, all_aux_outputs
            del ground_truth, seq_list, loss_mask, metric_mask, losses
            if_plot = False

        # update losses
        for loss_name in losses_epoch.keys():
            # mean losses over batches
            losses_epoch[loss_name] = losses_epoch[loss_name]/num_evaluate_batches
        losses_epoch = add_dict_prefix(losses_epoch, prefix=mode)

        # update metrics
        evaluate_metrics = {}
        for metric_name in metrics_epoch.keys():
            array_metric = np.array(metrics_epoch[metric_name])
            evaluate_metrics[metric_name] = array_metric.mean()
            if "goal" in metric_name:
                evaluate_metrics[metric_name + '_std'] = array_metric.std()
        evaluate_metrics = add_dict_prefix(evaluate_metrics, prefix=mode)

        return losses_epoch, evaluate_metrics

    def _set_device(self):
        """
        Set the device for the experiment. GPU if available, else CPU.
        """
        # torch.cuda.is_available() already checked in parsed args
        device = torch.device(self.args.device)
        print('\nUsing device:', device)

        # Additional info when using cuda
        if device.type == 'cuda':
            print('Number of available GPUs:', torch.cuda.device_count())
            print('GPU name:', torch.cuda.get_device_name(0))
            print('Cuda version:', torch.version.cuda)
        print()
        return device
