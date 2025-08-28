'''A wrapper class for optimizer '''
import numpy as np
import torch

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, init_lr_scale, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = init_lr_scale*np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class CosineAnnealingOptim:
    '''A wrapper class for optimizer + cosine annealing scheduler'''

    def __init__(self, optimizer, total_steps, min_lr=1e-6):
        """
        :param optimizer: wrapped optimizer (e.g. AdamW)
        :param total_steps: total number of training steps (epochs * steps_per_epoch)
        :param min_lr: minimum learning rate at the end of cosine cycle
        """
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

    def step_and_update_lr(self):
        """Step with the inner optimizer and update the learning rate using scheduler"""
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        """Zero out the gradients"""
        self.optimizer.zero_grad()

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']