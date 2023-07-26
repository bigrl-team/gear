import importlib
import logging
import pathlib
import pickle as pkl
import random
from typing import Any, Dict, List, Literal, Sequence, Union

import libgear.core as glibc
import torch
from gear.dataset import SharedDataset
from gear.dtypes import CVT_DTYPES_GEAR_TO_TORCH, CVT_DTYPES_TORCH_TO_GEAR
from gear.mpu import ModelParallelismUnit as Mpu
from gear.sampler import TopKSampler, UniformSampler, WeightedSampler
from gear.utils import get_local_node_hashed_int32

from .basic_loader import BasicLoader


def ensure_dataset(dataset: Union[SharedDataset, str, pathlib.Path], *args, **kwargs):
    """
    Helper function for dataset preparation.

    params:
        dataset: Union[SharedDataset, str, pathlib.Path], either string-like path and pathlib.Path that points the serialized state file or a built SharedDataset.

    return:
        built SharedDataset
    """
    if isinstance(dataset, SharedDataset):
        return dataset
    elif isinstance(dataset, str) or isinstance(dataset, pathlib.Path):
        return SharedDataset().load(dataset, *args, **kwargs)
    else:
        raise NotImplementedError(
            f"Conversion from {dataset} to SharedDataset is unknown"
        )


class DistributedOfflineLoader(BasicLoader):
    @staticmethod
    def create(
        data_path: Union[pathlib.Path, SharedDataset],
        batch_size: int,
        mpu: Mpu,
        sampling_method: Literal["Uniform", "Weighted", "Topk"] = "Uniform",
        patterns: Union[
            None, Sequence[Union[glibc.SubscribePattern, Dict[str, Any], None]]
        ] = None,
        key: Union[int, None] = None,
        attach: bool = True,
    ):
        if key is None:
            key = get_local_node_hashed_int32()
        dataset = ensure_dataset(data_path, key)
        return DistributedOfflineLoader(
            dataset, batch_size, mpu, sampling_method, patterns
        )

    def __init__(
        self,
        dataset: Union[SharedDataset, str, pathlib.Path],
        batch_size: int,
        mpu,
        sampling_method: Literal["Uniform", "Weighted", "Topk"] = "Uniform",
        patterns: List[glibc.SubscribePattern] = None,
        logger: logging.Logger = None,
    ) -> None:
        super().__init__(batch_size)

        self._dataset = ensure_dataset(dataset)
        self._handler = self._dataset._handler
        self._table = self._dataset._table
        self._iset = self._dataset._iset
        self._sub_cols = list(
            [self._table.get_table_spec().index(p["name"]) for p in patterns]
        )
        self._patterns = list(
            [
                glibc.SubscribePattern(
                    p["offset"],
                    p["length"],
                    glibc.SubscribePattern.PadOption.tail
                    if p["pad"] == "tail"
                    else glibc.SubscribePattern.PadOption.head,
                )
                for p in patterns
            ]
        )
        self._handler.set(self._sub_cols, self._patterns)
        self._mpu = mpu
        self._dp_rank = self._mpu.get_data_parallel_rank()
        self._dp_world = self._mpu.get_data_parallel_world_size()
        self._dp_capacity = self._iset.local_capacity
        self._batch_size = batch_size
        self._slice_size = max(int(batch_size / self._dp_world), 1)

        self._device_weights = self._iset.weights.to(mpu.device)
        self._weights_dirty = False

        self._logger = (
            logger
            if logger
            else logging.Logger(name="OfflineLoader", level=logging.DEBUG)
        )

        SamplerClass = getattr(
            importlib.import_module("gear.sampler"), f"{sampling_method}Sampler"
        )
        self._sampler: Union[
            UniformSampler, WeightedSampler, TopKSampler
        ] = SamplerClass(self._mpu, seed=42)
        self._indices = (
            torch.arange(
                self._iset.local_capacity, dtype=torch.long, device=self._mpu.device
            )
            + self._iset.index_offset
        )
        self._timesteps = self._iset.timesteps.to(self._mpu.device)

    @property
    def table_spec(self) -> Union[glibc.TableSpec, None]:
        if self._table:
            return self._table.get_table_spec()
        else:
            return None

    def partition_indices(self, global_indices):
        batch_size = global_indices.size(-1)
        slice_size = self._slice_size

        indices_partitions = list(
            [[None] * self._dp_world for _ in range(self._dp_world)]
        )
        for i in range(self._dp_world):
            assigned_indices = global_indices[
                ..., slice_size * i : slice_size * (i + 1)
            ]
            for j in range(self._dp_world):
                start = j * self._dp_capacity
                end = start + self._dp_capacity
                indices_partitions[i][j] = assigned_indices[
                    ..., (start <= assigned_indices[0]) & (assigned_indices[0] < end)
                ]
                print(
                    f"collecting {indices_partitions[i][j]} from {j} to {i}(batchsize {batch_size})"
                )

        return indices_partitions

    def generate_rank_comm_order(self):
        """
        Any matrix with properties:
            (i, k) = j => (j, k) = i,
            m[][k] = {0, 1, ..., n}
            m[k][] = {0, 1, ..., n}
        """
        ranks = [None] * (self._dp_world)
        ranks[0] = (0 - self._dp_rank + self._dp_world) % self._dp_world
        for i in range(1, self._dp_world):
            ranks[i] = (ranks[i - 1] + 1) % self._dp_world
        return ranks

    def fused_iterate(self):
        sampled_return = self._sampler.sync_sample(
            torch.stack([self._indices, self._timesteps], dim=0),
            self._iset.weights,
            self._batch_size,
        )
        print(
            "============>>>>>>",
            sampled_return,
            sampled_return.shape,
            torch.stack([self._indices, self._timesteps], dim=0).shape,
            self._indices.shape,
            self._timesteps.shape,
        )
        parts = self.partition_indices(sampled_return)
        print(f"Partitioned index are {parts}")
        data_batch = [None] * len(self._sub_cols)

        orders = self.generate_rank_comm_order()

        timesteps = torch.zeros(self._slice_size, dtype=torch.long)
        idx_prev = 0
        bounds = [0] * (len(orders) + 1)
        for visit_order, rank in enumerate(orders):
            selected_part = parts[self._dp_rank][rank]
            if selected_part.shape[-1] == 0:
                continue

            idx_curr = selected_part.shape[-1]
            timesteps[idx_prev : idx_prev + idx_curr] = selected_part[1].cpu()
            idx_prev += idx_curr
            bounds[visit_order + 1] = idx_prev

        for i, col_id in enumerate(self._sub_cols):
            cspec: glibc.ColumnSpec = self._table.get_column_spec(col_id)
            # pytorch allocator
            data_batch[i] = torch.zeros(
                size=(
                    self._slice_size,
                    self._patterns[i].length,
                )
                + tuple(cspec.shape),
                dtype=CVT_DTYPES_GEAR_TO_TORCH[cspec.dtype],
                device=self._mpu.device,
            )

            lrank = self._dp_rank
            for visit_order, rank in enumerate(orders):
                print(
                    f"Before Partitioned Indices are {parts}, data_batch shape {data_batch[i].shape} {(self._batch_size,self._patterns[i].length,)} {tuple(cspec.shape)}"
                )
                if lrank >= rank:
                    self._try_send(parts, lrank, rank, col_id)
                    torch.cuda.synchronize()
                    print(
                        f"After try-send Partitioned Indices are {parts}, data_batch shape {data_batch[i].shape} {id(parts[0][0])}"
                    )
                    self._try_recv(
                        parts, lrank, rank, col_id, data_batch[i][bounds[visit_order] :]
                    )
                    torch.cuda.synchronize()
                    print(
                        f"After try-recv Partitioned Indices are {parts}, data_batch shape {data_batch[i].shape} {id(parts[0][0])}"
                    )
                else:
                    self._try_recv(
                        parts, lrank, rank, col_id, data_batch[i][bounds[visit_order] :]
                    )
                    torch.cuda.synchronize()
                    self._try_send(parts, lrank, rank, col_id)
                    torch.cuda.synchronize()
                print(
                    f"AFter Partitioned Indices are {parts}, data_batch shape {data_batch[i].shape} {(self._batch_size,self._patterns[i].length,)} {tuple(cspec.shape)}"
                )

            # torch.cuda.synchronize()
        return timesteps, data_batch

    def __next__(self):
        return self.fused_iterate()

    def _try_recv(self, parts, lrank, rrank, cid, dst):
        print(f"Rank {lrank} recv from {rrank} {parts[lrank][rrank]}")
        selected_part = parts[lrank][rrank]
        if selected_part.shape[-1] == 0:
            return
        
        if self._mpu.is_on_the_same_node(rrank):
            tss = selected_part[1]
            indices = (selected_part[0] % self._dp_capacity)
            print(f"In-recva partitioned indices are {parts}")
            print(f"subs local mem index {indices} timesteps {tss} dst {dst.shape}")
            self._handler.subcopy(indices, tss, tss, cid, dst)
            print(f"In-recvb partitioned indices are {parts}")
        else:
            tss = selected_part[1].cpu()
            indices = (selected_part[0] % self._dp_capacity).cpu()
            print(f"subs remote mem index {indices} timesteps {tss}")
            self._handler.subrecv(
                rrank,
                dst,
                glibc.Int64Span.from_tensor(indices),
                glibc.Int64Span.from_tensor(tss),
                glibc.Int64Span.from_tensor(tss),
                cid,
            )

    def _try_send(
        self,
        parts,
        lrank,
        rrank,
        cid,
    ):
        selected_part = parts[rrank][lrank]
        if selected_part.shape[-1] == 0:
            return
        tss = selected_part[1].cpu()
        indices = (selected_part[0] % self._dp_capacity).cpu()
        if self._mpu.is_on_the_same_node(rrank):
            return
        else:
            self._handler.subsend(
                rrank,
                glibc.Int64Span.from_tensor(indices),
                glibc.Int64Span.from_tensor(tss),
                glibc.Int64Span.from_tensor(tss),
                cid,
            )
