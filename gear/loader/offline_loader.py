import importlib
import logging
import pathlib
import pickle as pkl
import random
from typing import Any, Dict, List, Literal, Sequence, Union

import torch
import libgear as glib
import libgear.storage as glibs

import gear
from gear.dataset import SharedDataset
from gear.dtypes import CVT_DTYPES_GEAR_TO_TORCH, CVT_DTYPES_TORCH_TO_GEAR
from gear.mpu import ModelParallelismUnit as Mpu
from gear.sampler import TopKSampler, UniformSampler, WeightedSampler
from gear.utils import get_local_node_hashed_int32

from .basic_loader import BasicLoader


def ensure_dataset(dataset: Union[SharedDataset, str, pathlib.Path]) -> SharedDataset:
    """
    Helper function for dataset preparation.

    :type dataset: dataset: Union[SharedDataset, str, pathlib.Path]
    :param dataset:
        Either string-like path and pathlib.Path that points the serialized state file or a built SharedDataset.

    :rtype gear.dataset.SharedDataset
    :return:
        build SharedDataset
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
    """
    Loading data from a temporal built or serialized SharedDataset


    Usage:

    .. code-block:: python

        timestep, data_batch = next(loader)
    """

    @staticmethod
    def create(
        data_path: pathlib.Path,
        batch_size: int,
        mpu: Mpu,
        sampling_method: Literal["Uniform", "Weighted", "Topk"] = "Uniform",
        patterns: Union[
            None, Sequence[Union[glibs.SubscribePattern, Dict[str, Any], None]]
        ] = None,
        key: Union[int, None] = None,
        attach: bool = True,
    ):
        """
        OfflineLoader factory function

        """
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
        patterns: List[glibs.SubscribePattern] = None,
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
                glibs.SubscribePattern(
                    p["offset"],
                    p["length"],
                    glibs.SubscribePattern.PadOption.tail
                    if p["pad"] == "tail"
                    else glibs.SubscribePattern.PadOption.head,
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
    def table_spec(self) -> Union[glibs.TableSpec, None]:
        """Get the table spec from the subscribed table.

        .. seealso::

            :py:class:`libgear.storage.TableSpec`."""
        if self._table:
            return self._table.get_table_spec()
        else:
            return None

    def get_tensor_view(
        self, index: int, cids: Union[Sequence[int], None] = None
    ) -> tuple[torch.Tensor]:
        """
        .. seealso::

            :py:func:`gear.dataset.SharedDataset.get_tensor_view`.


        :type index: int
        :param index:
            The index of the corresponding trajectory id and the trajectory id should be on the local node.

        :type cids: Sequence[int]
        :param cids:
            The ids of the columns to be subscribed. All columns will be returned if no column specified.

        :rtype: tuple[torch.Tensor]
        """
        if self._handler is None:
            raise gear.errors.HandlerMissing()
        return self._dataset.get_tensor_view(index, cids)

    def _partition_indices(self, global_indices):
        batch_size = global_indices.size(0)
        slice_size = max(int(batch_size / self._dp_world), 1)

        indices_partitions = list(
            [[None] * self._dp_world for _ in range(self._dp_world)]
        )
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

    def _generate_rank_comm_order(self):
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

    def _fused_iterate(self):
        sampled_indices = self._sampler.sync_sample(
            self._indices,
            self._iset.weights,
            self._batch_size * self._dp_world,
        )
        # print("============>>>>>>", sampled_indices)
        parts = self._partition_indices(sampled_indices)

        data_batch = [None] * len(self._sub_cols)

        orders = self._generate_rank_comm_order()

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
            cspec: glibs.ColumnSpec = self._table.get_column_spec(col_id)
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

    def _low_level_controlled_iterate(self):
        sampled_indices = self._sampler.sync_sample(
            self._indices,
            self._iset.weights,
            self._batch_size * self._dp_world,
        ).cpu()
        # print("============>>>>>>", sampled_indices)
        parts = self._partition_indices(sampled_indices)

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

        orders = self._generate_rank_comm_order()
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
            cspec: glibs.ColumnSpec = self._table.get_column_spec(col_id)
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
                glib.Int64Span.from_tensor(parts[self._dp_rank]),  # indices
                glib.Int64Span.from_tensor(tss),  # timesteps
                glib.Int64Span.from_tensor(tss),  # length(offline case == timesteps)
                col_id,  # column id
                glib.Int64Span.from_tensor(src_offsets),
                glib.Int64Span.from_tensor(dst_offsets),
                glib.Int64Span.from_tensor(copied_lens),
            )
            glib.cuda.vcopy(
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
        return self._fused_iterate()
        # return self._low_level_controlled_iterate()
