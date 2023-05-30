import logging

import torch
from gear.comm import Communicator

from .base import DistributedSampler
from .sample_return import SampleReturn


@torch.jit.script
def resample_weight_max_deviation(weights: torch.FloatTensor):
    weights_sum = weights.sum()
    return (((weights - weights_sum) / weights.size(0)).abs()).max()


"""
    To achieve similar results as global sampling, samplers do sample more 
    indices than needed to enable re-sampling.
"""


class WeightedSampler(DistributedSampler):
    def __init__(self, context) -> None:
        super().__init__(context)

    def sample(self, weights, batch_size):
        weights = weights.to(self.device)
        paritial_weights_sum = torch.sum(weights)
        samples = torch.sort(
            torch.multinomial(weights, num_samples=batch_size, generator=self.generator)
        ).values
        return SampleReturn(samples, weights[samples], paritial_weights_sum)

    def sync(self, sample_return: SampleReturn, comm: Communicator):
        rank = self.context.get_rank()
        world_size = self.context.get_world_size()

        if world_size == 1:
            return SampleReturn.replica(sample_return, 1)

        sample_return = SampleReturn.replica(sample_return, world_size)
        # TODO: tree-reduction
        if rank > 0:
            comm.send(sample_return.indices[rank], dst=0)
            comm.send(sample_return.weights[rank], dst=0)
            comm.send(sample_return.partial_sum[rank], dst=0)

            comm.recv(sample_return.indices, src=0)
            comm.recv(sample_return.weights, src=0)
            comm.recv(sample_return.partial_sum, src=0)
        else:
            # self.context.get_rank() == 0
            for i in range(1, self.context.get_world_size()):
                comm.recv(sample_return.indices[i], src=i)
                comm.recv(sample_return.weights[i], src=i)
                comm.recv(sample_return.partial_sum[i], src=i)

            size_var, size_mean = torch.var_mean(sample_return.partial_sum)
            if (size_var / size_mean) > 0.05:
                logging.warning("WARNING: <WeightedSampler> Imbalance parition sum")

            for i in range(1, self.context.get_world_size()):
                comm.send(sample_return.indices, dst=i)
                comm.send(sample_return.weights, dst=i)
                comm.send(sample_return.partial_sum, dst=i)

        return sample_return
