"""Custom schedulers for Yoke.

A module of custom schedulers to use to train Yoke models.

"""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import _LRScheduler


def calc_lr(step, dim_embed, warmup_steps):
    """Warm-up with root decay."""

    return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))


class WarmUpRootDecayScheduler(_LRScheduler):
    """Scheduler with warm-up and root decay.
    
    Scheduler mentioned in *https://kikaben.com/transformers-training-details*

    This goes through a linear warm-up phase for the first `warmup_steps` in
    which the learning rate goes from 0 to `warmup_steps**-0.5`. Then the
    learning rate decays as the inverse square root of the number of steps.

    """
    def __init__(
            self, 
            optimizer: Optimizer,
            dim_embed: int,
            warmup_steps: int,
            last_epoch: int=-1,
            verbose: bool=False
    ) -> None:
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups
    

class CosineWithWarmupLR(LambdaLR):
    """Cosine decay after warmup.

    From:
    https://github.com/krasserm/perceiver-io/blob/main/perceiver/scripts/lrs.py

    """
    def __init__(
        self,
        optimizer: Optimizer,
        training_steps: int = 0,
        warmup_steps: int = 0,
        num_cycles: float = 0.5,
        min_fraction: float = 0.0,
        last_epoch: int = -1,
    ):
        # Can be updated after instantiation
        self.training_steps = training_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps)
            progress = progress / float(max(1, self.training_steps - warmup_steps))

            tmp_lr = (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
            tmp_lr = 0.5 * (1.0 - min_fraction) * tmp_lr
            lr_return = max(0.0, tmp_lr)
            lr_return = min_fraction + lr_return
            
            return lr_return


        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class ConstantWithWarmupLR(LambdaLR):
    """Constant LR after warmup

    From:
    https://github.com/krasserm/perceiver-io/blob/main/perceiver/scripts/lrs.py

    """
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_steps: int = 0,
            last_epoch: int = -1
    ):
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1.0, warmup_steps))
            
            return 1.0

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


