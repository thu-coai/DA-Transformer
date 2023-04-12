# Modified from https://github.com/bytedance/lightseq/blob/812d9d798e491ab9139c1f36113693308c4c0637/lightseq/training/cli/fs_modules/ls_adam.py
# Licensed under the Apache License, Version 2, by the authors of LightSeq

import logging
import math
from dataclasses import dataclass, field
from typing import List, Any

import torch
import torch.distributed as dist
import torch.optim
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from omegaconf import II
try:
    from lightseq.training.ops.pytorch.adam import LSAdam
except ModuleNotFoundError as err:
    LSAdam = None

logger = logging.getLogger(__name__)


@dataclass
class LSFSAdamConfig(FairseqDataclass):
    adam_betas: Any = field(
        default=(0.9, 0.999), metadata={"help": "betas for Adam optimizer"}
    )
    adam_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for Adam optimizer"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    use_old_adam: bool = field(
        default=False, metadata={"help": "Use fairseq.optim.adam.Adam"}
    )
    # TODO common vars below in parent
    tpu: bool = II("common.tpu")
    lr: List[float] = II("optimization.lr")


@register_optimizer("ls_adam", dataclass=LSFSAdamConfig)
class LSFSAdam(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, cfg, params):
        super().__init__(cfg)
        if LSAdam is None:
            raise NotImplementedError("Please install lightseq before using ls_adam")
        fused_adam_cls = LSAdam
        use_fused_adam = (
            not getattr(cfg, "use_old_adam", False)
            and fused_adam_cls is not None
            and torch.cuda.is_available()
        )
        logger.info("using LightSeq Adam")
        assert use_fused_adam
        self._optimizer = fused_adam_cls(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0],
            "betas": eval(self.cfg.adam_betas),
            "eps": self.cfg.adam_eps,
            "weight_decay": self.cfg.weight_decay,
        }

    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)

