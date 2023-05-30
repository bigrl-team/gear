import torch
from gear.comm import Communicator

from .base import DistributedSampler
from .sample_return import SampleReturn


class UniformSampler(DistributedSampler):
    def __init__(self, context, ratio=1, seed=None) -> None:
        super().__init__(context, max(ratio, 1), seed)

    def distributed_sample(self, size, batch_size):
        oversample_batch_size = int(batch_size * self.ratio)
        ret = torch.zeros(oversample_batch_size, device=self.device, dtype=torch.int32)
        weights = torch.ones(
            oversample_batch_size, device=self.device, dtype=torch.float32
        )
        paritial_sum = torch.tensor(
            oversample_batch_size, device=self.device, dtype=torch.float32
        )
        torch.randint(
            low=0,
            high=size,
            size=ret.size(),
            generator=self.generator,
            dtype=torch.int32,
            device=self.device,
            out=ret,
        )
        return SampleReturn(ret, weights, paritial_sum, batch_size, self.ratio)

    def merge_gather(self, sample_return: SampleReturn, comm: Communicator):
        rank = self.context.get_rank()
        world_size = self.context.get_world_size()

        if world_size == 1:
            return SampleReturn.replica(sample_return, 1)

        sample_return = SampleReturn.replica(sample_return, world_size)
        # TODO: tree-reduction
        if rank > 0:
            comm.send(sample_return.indices[rank], dst=0)
            # comm.send(sample_return.weights[rank], dst=0)
            comm.send(sample_return.partial_sum[rank], dst=0)

            comm.recv(sample_return.indices, src=0)
            # comm.recv(sample_return.weights, src=0)
            comm.recv(sample_return.partial_sum, src=0)
        else:
            # self.context.get_rank() == 0
            for i in range(1, self.context.get_world_size()):
                comm.recv(sample_return.indices[i], src=i)
                # comm.recv(sample_return.weights[i], src=i)
                comm.recv(sample_return.partial_sum[i], src=i)

            sum_mean = torch.mean(sample_return.partial_sum)
            sum_max = torch.max()
            if (size_var / size_mean) > 0.05:
                self._performance_degrade_warning()

            for i in range(1, self.context.get_world_size()):
                comm.send(sample_return.indices, dst=i)
                # comm.send(sample_return.weights, dst=i)
                comm.send(sample_return.partial_sum, dst=i)

        return sample_return

    def gather(
        self,
    ):
        pass

    def scatter(
        self,
    ):
        pass

    def _performance_degrade_warning(self):
        logging.warning(
            "WARNING: <UniformSampler> Imbalance parition sizes, \
        pesudo-uniform may cause performance degrade, consider set larger ratio \
        to enable aggresive resampling."
        )
