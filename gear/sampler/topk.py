import logging

import torch
import torch.distributed as dist
import torch.distributions as distributions
from gear.comm import Communicator

from .base import DistributedSampler
from .sample_return import SampleReturn


class TopKSampler(DistributedSampler):
    def __init__(self, context) -> None:
        super().__init__(context)

    def sample(self, weights, batch_size, compact):
        weights = weights.to(self.device)
        paritial_weights_sum = torch.sum(weights)
        values, indices = torch.topk(weights, k=batch_size)
        return SampleReturn(indices, values, paritial_weights_sum)
