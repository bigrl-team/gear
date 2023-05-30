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


def ensure_dataset(dataset: Union[SharedDataset, str, pathlib.Path]):
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
        return SharedDataset().load(dataset)
    else:
        raise NotImplementedError(
            f"Conversion from {dataset} to SharedDataset is unknown"
        )


class OfflineLoader(BasicLoader):
    @staticmethod
    def create(
        data_path: pathlib.Path,
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
        dataset = SharedDataset().load(data_path, key)
        return OfflineLoader(dataset, batch_size, mpu, sampling_method, patterns)

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

    @property
    def table_spec(self) -> Union[glibc.TableSpec, None]:
        if self._table:
            return self._table.get_table_spec()
        else:
            return None

    def partition_indices(self, global_indices):
        batch_size = global_indices.size(0)
        slice_size = max(int(batch_size / self._dp_world), 1)

        indices_partitions = [[None] * self._dp_world] * self._dp_world
        for i in range(self._dp_world):
            assigned_indices = global_indices[slice_size * i : slice_size * (i + 1)]
            for j in range(self._dp_world):
                start = j * self._dp_capacity
                end = start + self._dp_capacity
                indices_partitions[i][j] = assigned_indices[
                    (start <= assigned_indices) & (assigned_indices < end)
                ]
                # print(f"collecting {indices_partitions[i][j]} from {j} to {i}")

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
        sampled_indices = self._sampler.sync_sample(
            self._indices,
            self._iset.weights,
            self._batch_size,
        )
        # print("============>>>>>>", sampled_indices)
        parts = self.partition_indices(sampled_indices)

        data_batch = [None] * len(self._sub_cols)

        orders = self.generate_rank_comm_order()

        timesteps = torch.zeros(self._batch_size, dtype=torch.long)
        idx_prev = 0
        for rank in orders:
            if len(parts[self._dp_rank][rank].shape) == 0:
                continue
            idx_curr = len(parts[self._dp_rank][rank])
            timesteps[idx_prev : idx_prev + idx_curr] = self._iset.timesteps[
                parts[self._dp_rank][rank].cpu()
            ]
            idx_prev += idx_curr
        for i, col_id in enumerate(self._sub_cols):
            cspec: glibc.ColumnSpec = self._table.get_column_spec(col_id)
            # pytorch allocator
            data_batch[i] = torch.zeros(
                size=(
                    self._batch_size,
                    self._patterns[i].length,
                )
                + tuple(cspec.shape),
                dtype=CVT_DTYPES_GEAR_TO_TORCH[cspec.dtype],
                device=self._mpu.device,
            )

            idx_prev = 0
            for rank in orders:
                if len(parts[self._dp_rank][rank].shape) == 0:
                    continue
                idx_curr = len(parts[self._dp_rank][rank])
                tss = (timesteps[idx_prev:idx_curr] % self._dp_capacity).to(
                    self._mpu.device
                )

                self._handler.subcopy(
                    parts[self._dp_rank][rank], tss, tss, col_id, data_batch[i]
                )
            # torch.cuda.synchronize()
        return timesteps, data_batch

    def low_level_controlled_iterate(self):
        sampled_indices = self._sampler.sync_sample(
            self._indices,
            self._iset.weights,
            self._batch_size * self._dp_world,
        ).cpu()
        # print("============>>>>>>", sampled_indices)
        parts = self.partition_indices(sampled_indices)

        ncol = len(self._sub_cols)
        data_batch = [None] * ncol
        src_offsets = torch.zeros(
            int(sampled_indices.numel() / self._dp_world), dtype=torch.long
        )
        dst_offsets = torch.zeros(
            int(sampled_indices.numel() / self._dp_world), dtype=torch.long
        )
        copied_lens = torch.zeros(
            int(sampled_indices.numel() / self._dp_world), dtype=torch.long
        )

        orders = self.generate_rank_comm_order()
        timesteps = torch.zeros(self._batch_size, dtype=torch.long)
        idx_prev = 0
        for rank in orders:
            if len(parts[self._dp_rank][rank].shape) == 0:
                continue
            idx_curr = len(parts[self._dp_rank][rank])
            timesteps[idx_prev : idx_prev + idx_curr] = self._iset.timesteps[
                parts[self._dp_rank][rank].cpu()
            ]
            idx_prev += idx_curr

        for i, col_id in enumerate(self._sub_cols):
            cspec: glibc.ColumnSpec = self._table.get_column_spec(col_id)
            # pytorch allocator
            data_batch[i] = torch.zeros(
                size=(
                    self._batch_size,
                    self._patterns[i].length,
                )
                + tuple(cspec.shape),
                dtype=CVT_DTYPES_GEAR_TO_TORCH[cspec.dtype],
                device=self._mpu.device,
            )

            src_offsets.fill_(0)
            dst_offsets.fill_(0)
            copied_lens.fill_(0)

            if len(parts[self._dp_rank]) == 0:
                continue
            tss = self._iset.timesteps[parts[self._dp_rank] % self._dp_capacity]

            max_len = self._handler.sub(
                glibc.Int64Span.from_tensor(parts[self._dp_rank]),  # indices
                glibc.Int64Span.from_tensor(tss),  # timesteps
                glibc.Int64Span.from_tensor(tss),  # length(offline case == timesteps)
                col_id,  # column id
                glibc.Int64Span.from_tensor(src_offsets),
                glibc.Int64Span.from_tensor(dst_offsets),
                glibc.Int64Span.from_tensor(copied_lens),
            )
            glibc.vcopy(
                self._table.get_address(),
                data_batch[i],
                src_offsets.cuda(),
                dst_offsets.cuda(),
                copied_lens.cuda(),
                max_len,
            )
            torch.cuda.synchronize()
        return tss, data_batch

    def __next__(self):
        return self.fused_iterate()
        # return self.low_level_controlled_iterate()

    def exposed_fused_iterate(self, indices):
        data_batch = [None] * len(self._sub_cols)

        for i, col_id in enumerate(self._sub_cols):
            cspec: glibc.ColumnSpec = self._table.get_column_spec(col_id)
            # pytorch allocator
            data_batch[i] = torch.zeros(
                size=(
                    len(indices),
                    self._patterns[i].length,
                )
                + tuple(cspec.shape),
                dtype=CVT_DTYPES_GEAR_TO_TORCH[cspec.dtype],
                device=self._mpu.device,
            )
            local_indices = indices % self._dp_capacity
            tss = self._iset.timesteps[local_indices].to(self._mpu.device)

            self._handler.subcopy(indices.cuda(), tss, tss, col_id, data_batch[i])
            # torch.cuda.synchronize()
        return data_batch

    def exposed_low_level_controlled_iterate(self, indices):
        ncol = len(self._sub_cols)
        data_batch = [None] * ncol
        src_offsets = torch.zeros(
            int(indices.numel() / self._dp_world), dtype=torch.long
        )
        dst_offsets = torch.zeros(
            int(indices.numel() / self._dp_world), dtype=torch.long
        )
        copied_lens = torch.zeros(
            int(indices.numel() / self._dp_world), dtype=torch.long
        )

        for i, col_id in enumerate(self._sub_cols):
            cspec: glibc.ColumnSpec = self._table.get_column_spec(col_id)
            # pytorch allocator
            data_batch[i] = torch.zeros(
                size=(
                    len(indices),
                    self._patterns[i].length,
                )
                + tuple(cspec.shape),
                dtype=CVT_DTYPES_GEAR_TO_TORCH[cspec.dtype],
                device=self._mpu.device,
            )

            src_offsets.fill_(0)
            dst_offsets.fill_(0)
            copied_lens.fill_(0)

            tss = self._iset.timesteps[indices % self._dp_capacity]

            max_len = self._handler.sub(
                glibc.Int64Span.from_tensor(indices),  # indices
                glibc.Int64Span.from_tensor(tss),  # timesteps
                glibc.Int64Span.from_tensor(tss),  # length(offline case == timesteps)
                col_id,  # column id
                glibc.Int64Span.from_tensor(src_offsets),
                glibc.Int64Span.from_tensor(dst_offsets),
                glibc.Int64Span.from_tensor(copied_lens),
            )
            torch.cuda.synchronize()
            glibc.vcopy(
                self._table.get_address(),
                data_batch[i],
                src_offsets.cuda(),
                dst_offsets.cuda(),
                copied_lens.cuda(),
                max_len,
            )
            torch.cuda.synchronize()
        return tss, data_batch
