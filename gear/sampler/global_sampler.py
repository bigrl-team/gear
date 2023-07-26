from typing import List

import torch
import torch.distributed as dist

if hasattr(torch.distributed.distributed_c10d, "get_global_rank"):
    from torch.distributed.distributed_c10d import get_global_rank as _get_global_rank
else:
    from torch.distributed.distributed_c10d import _get_global_rank

import numpy as np
import torch._C
import torch.jit

from .utils import torch_dtype_to_int, torch_int_to_dtype


def get_global_rank(group, group_rank):
    return _get_global_rank(group, group_rank)


def _send(tensor, dst, group, tag, device):
    # param dst is group rank, convert to global rank
    dst = get_global_rank(group, dst)
    ndim_dtype_tensor = torch.LongTensor(
        data=[len(tensor.size()), torch_dtype_to_int[tensor.dtype]]
    ).to(device)
    shape_tensor = torch.LongTensor(data=[tensor.size()]).to(device)
    dist.send(ndim_dtype_tensor, dst=dst, group=group)
    dist.send(shape_tensor, dst=dst, group=group)
    # print(f"===>RANK{dist.get_rank()} send tensor shape {tensor.shape} shape tensor {shape_tensor.tolist()} to dst {dst}")
    dist.send(tensor, dst=dst, group=group)
    return


def _recv(src, group, tag, device):
    # param src is group rank, convert to global rank
    src = get_global_rank(group, src)
    ndim_dtype_tensor = torch.LongTensor(data=[0, 0]).to(device)
    dist.recv(ndim_dtype_tensor, src=src, group=group)
    # print(
    #     f"RANK=={dist.get_rank()} get ndtim dtype tensor {ndim_dtype_tensor.cpu()} from source {src}"
    # )
    shape_tensor = torch.LongTensor(data=[1] * ndim_dtype_tensor[0]).to(device)
    dist.recv(shape_tensor, src=src, group=group)
    # print(
    #     f"RANK=={dist.get_rank()} get shape tensor {shape_tensor.cpu()} from source {src}"
    # )
    tensor = torch.zeros(
        shape_tensor.tolist(), dtype=torch_int_to_dtype[ndim_dtype_tensor[1].item()]
    ).to(device)
    dist.recv(tensor, src=src, group=group)
    # print(
    #     f"===>RANK{dist.get_rank()}recv tensor shape {tensor.shape} shape tensor {shape_tensor.tolist()} ndim tensor {ndim_dtype_tensor.tolist()} from src {src}"
    # )
    return tensor


def ensure_tensor(v, device="cuda"):
    if isinstance(v, torch.Tensor):
        return v.to(device)
    elif isinstance(v, np.ndarray):
        v = torch.from_numpy(v).to(device)
    else:
        v = torch.Tensor(v).to(device)
    return v


@torch.jit.script
def _uniform_sample(
    indices_collection: List[torch.Tensor], num_samples: int, device: torch.device
):
    concat_indices = torch.cat(indices_collection, dim=0)
    return torch.sort(
        concat_indices[
            torch.randint(
                low=0,
                high=concat_indices.size(0),
                size=(num_samples,),
                dtype=torch.long,
                device=device,
            )
        ]
    ).values


class UniformSampler:
    def __init__(self, mpu, seed=42) -> None:
        self._mpu = mpu
        self._generator = torch.Generator(mpu.device)
        self._generator.manual_seed(seed)

    def sync_sample(self, indices, weights, batch_size):
        dp_rank = self._mpu.get_data_parallel_rank()
        device = self._mpu.device

        self._cache = torch.zeros(batch_size, device=device, dtype=torch.long)
        if dp_rank == 0:
            print("sdasdasdas")
            sampled_indices = self._dp_rank_zero_sync(indices, weights, batch_size)
        else:
            print(f"dasdasdasdas{dp_rank}")
            sampled_indices = self._dp_rank_nonzero_sync(indices, weights, batch_size)

        return sampled_indices

    def _dp_rank_zero_sync(self, indices, weights, batch_size):
        dp_rank = self._mpu.get_data_parallel_rank()
        dp_group = self._mpu.get_data_parallel_group()
        dp_world = self._mpu.get_data_parallel_world_size()
        device = self._mpu.device

        indices_collections = [None] * dp_world

        indices_collections[0] = ensure_tensor(indices, device)
        # weights_collections[0] = ensure_tensor(weights, device)
        for i in range(1, dp_world):
            # print(f"RANK0 recving {i}-th tensor ....")
            indices_collections[i] = _recv(src=i, group=dp_group, tag=i, device=device)
        # print([d.shape for d in indices_collections])
        concat_indices = torch.cat(indices_collections, dim=-1)

        torch.randint(
            low=0,
            high=concat_indices.size(-1),
            size=self._cache.size(),
            generator=self._generator,
            dtype=torch.long,
            device=device,
            out=self._cache,
        )

        sampled_indices = torch.index_select(
            input=concat_indices, dim=-1, index=torch.sort(self._cache).values
        )
        print(f"sampled_indices shape is {sampled_indices.shape}")

        for i in range(1, dp_world):
            _send(sampled_indices, dst=i, group=dp_group, tag=i, device=device)
        return sampled_indices

    def _dp_rank_nonzero_sync(self, indices, weights, batch_size):
        dp_rank = self._mpu.get_data_parallel_rank()
        dp_group = self._mpu.get_data_parallel_group()
        device = self._mpu.device

        # print(f"RANK={dp_rank} sending tensor {indices.shape}....")
        _send(
            ensure_tensor(indices, device),
            dst=0,
            group=dp_group,
            tag=dp_rank,
            device=device,
        )

        sampled_indices = _recv(src=0, group=dp_group, tag=dp_rank, device=device)
        return sampled_indices


class WeightedSampler(UniformSampler):
    def _dp_rank_zero_sync(self, mpu, device, indices, weights, batch_size):
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
        dp_world = mpu.get_data_parallel_world_size()

        indices_collections = [None] * dp_world

        indices_collections[0] = ensure_tensor(indices, device)
        # weights_collections[0] = ensure_tensor(weights, device)
        for i in range(1, dp_world):
            indices_collections[i] = _recv(src=i, group=dp_group, tag=i, device=device)
        # print([d.shape for d in indices_collections])
        concat_indices = torch.cat(indices_collections, dim=0)

        torch.multinomial(input=weights, num_samples=batch_size, out=self._cache)

        sampled_indices = torch.sort(concat_indices[self._cache]).values

        for i in range(1, dp_world):
            _send(sampled_indices, dst=i, group=dp_group, tag=i, device=device)
        return sampled_indices


class TopKSampler(UniformSampler):
    def _dp_rank_zero_sync(self, mpu, device, indices, weights, batch_size):
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
        dp_world = mpu.get_data_parallel_world_size()

        indices_collections = [None] * dp_world
        # weights_collections = [None] * self._world_size

        indices_collections[0] = ensure_tensor(indices, device)
        # weights_collections[0] = ensure_tensor(weights, device)
        for i in range(1, dp_world):
            indices_collections[i] = _recv(src=i, group=dp_group, tag=i, device=device)
        # print([d.shape for d in indices_collections])
        concat_indices = torch.cat(indices_collections, dim=0)

        sampled_indices = torch.sort(
            concat_indices[
                torch.topk(input=weights, k=batch_size, dim=0, largest=True).indices
            ]
        ).values

        for i in range(1, dp_world):
            _send(sampled_indices, dst=i, group=dp_group, tag=i, device=device)
        return sampled_indices
