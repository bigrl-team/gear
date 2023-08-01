import os
import pathlib
import pickle
import random
from collections import namedtuple
from typing import Sequence, Union


import torch
import libgear as glib
import libgear.storage as glibs
import libgear.index as glibi
from gear.config import KEY_T_RANGE
from gear.dtypes import DataType
from gear.specs import ColumnSpec, TableSpec
from gear.utils import get_local_node_hashed_int32, ensure_path
from torch.utils.data.dataset import Dataset

SharedDatasetMeta = namedtuple("SharedDatasetMeta", ["storage", "index"])
IndexMeta = namedtuple(
    "IndexMeta", ["global_capacity", "local_capacity", "index_offset", "shared", "key"]
)


class SharedDataset(Dataset):
    """
    SharedDataset consists a set of trajectories and their statuses, and provides straightforward API interfaces for storing, fetching and updating trajectory data. Its storage space  is allocated on the host's shared-memory, facilitating fast Inter-Process-Communication(IPC) data sharing without the need of memory serialization/deserialization. Furthermore, PyTorch dataloaders inherently assume a global view of underlying datasets, which means all data have to been visible to the dataloader. ``SharedDataset`` fulfill the requirement of the global accessibility within a single node, thereby eliminating the memory redundancy that arised by loading the entire copy of datasets in each process.
    """

    @staticmethod
    def create(
        spec: TableSpec = Union[TableSpec, None],
        create: bool = True,
        shard_rank: int = 0,
        shard_world: int = 1,
        key: Union[int, None] = None,
    ):
        """
        TODO

        :type spec: Union[TableSpec, None]
        :param spec:
            TableSpec

        :type create: bool
        :param create:


        :type shard_rank: int
        :param shard_rank:

        """
        if key is None:
            key = get_local_node_hashed_int32()
        rng = random.Random(key)
        table_key = rng.randint(0, KEY_T_RANGE)
        iset_key = rng.randint(0, KEY_T_RANGE)

        if create:
            if spec is None:
                raise RuntimeError(
                    "SharedDataset creator process should provide storage specs!"
                )
            table = glibs.TrajectoryTable(spec, table_key, True)
            iset = glibi.Indexset(
                int(spec.capacity * spec.worldsize),
                int(spec.capacity / shard_world),
                int(
                    spec.capacity * spec.rank + spec.capacity * shard_rank / shard_world
                ),
                shared=True,
                key=iset_key,
                create=True,
            )
        else:
            table = glibs.TrajectoryTable(spec, table_key, False)
            iset = glibi.Indexset(
                spec.capacity * spec.worldsize,
                spec.capacity / shard_world,
                index_offset=(
                    spec.capacity * spec.rank + spec.capacity * shard_rank / shard_world
                ),
                shared=True,
                key=iset_key,
                create=False,
            )

        trc = -1
        while trc == -1:
            trc = table.connect()
        return SharedDataset(
            table,
            iset,
        )

    def __init__(
        self,
        table: Union[glibs.TrajectoryTable, None] = None,
        index_set: Union[glibi.Indexset, None] = None,
    ) -> None:
        super().__init__()

        self._table = table
        self._iset = index_set
        if table and index_set:
            self._entry_cls = namedtuple(
                "SharedDatasetEntry",
                ["weight", "timestep"]
                + list(
                    [f"column{i}" for i in range(table.get_table_spec().num_columns)]
                ),
            )
            self._handler = glibs.get_cpu_handler(
                table,
                index_set.global_capacity,
                glib.Range(0, 0),
                glib.Range(
                    index_set.index_offset,
                    index_set.index_offset + index_set.local_capacity,
                ),
            )
        else:
            self._handler = None

    @property
    def weights(self) -> torch.FloatTensor:
        """Return the weights of the underlying indices in a tensor view(in host memory)"""
        return self._iset.weights

    @property
    def timesteps(self) -> torch.LongTensor:
        """Return the timesteps of the underlying indics in a tensor view(in host memory)"""
        return self._iset.timesteps

    @property
    def num_col(self) -> int:
        """Return number of columns in the table."""
        return self._table.ncolumns

    def __getitem__(self, index):
        ret = (
            self.weights[index],
            self.timesteps[index],
        )
        ret += self.get_tensor_view(index)
        return self._entry_cls(*ret)

    def __len__(self):
        return self._iset.capacity

    def __setitem__(self, index, value):
        weight, timestep, columns = value
        self._iset.weights[index] = weight
        self._iset.timesteps[index] = timestep
        for cid in range(self.num_col):
            self._handler.view(index, cid).copy(columns[cid].view(-1))

    def set_trajectory(
        self, index: int, weight: float, timestep: int, *columns: Sequence[torch.Tensor]
    ):
        """
        Set the values/attributes that represent the trajectory.

        GEAR provides two apis to set the trajectory entry with equal functionality:
        
        .. code-block:: python

            dataset[trajectory_index] = weight, timestep, columns
            dataset.set_trajectory(index, weight, timestep, columns)


        :type index: int
        :param index:
            The trajectory index.

        :type weight: float
        :param weight:
            Priority/weight of the trajectory.

        :type timestep: int
        :param timestep:
            The length of the trajectory.

        :type columns: Sequence[torch.tensor]
        :param columns:
            Column value tensors
        """
        self._iset.weights[index] = weight
        self._iset.timesteps[index] = timestep
        for cid in range(self.num_col):
            self._handler.view(index, cid).copy(columns[cid].view(-1))

    def get_column_spec(self, cid: int) -> ColumnSpec:
        """Return the corresponding ColumnSpec of the given column id"""
        return self._table.get_column_spec(cid)

    def get_column_dtype(self, cid: int) -> DataType:
        """Get the datatype of the column id specified"""
        return self._table.get_column_spec(cid).dtype

    def get_column_shape(self, cid: int) -> DataType:
        """Get column data shape of the given column id"""
        return (self._table.get_table_spec().trajectory_length,) + tuple(
            self._table.get_column_spec(cid).shape
        )

    def get_raw_view(
        self, index: int, cids: Sequence[int] = None
    ) -> tuple[glib.Uint8Span]:
        """
        Get raw memory subscription of the underlying columns of a trajectory.

        :type index: int
        :param index:
            The index of the corresponding trajectory id and the trajectory id should be on the local node.

        :type cids: Sequence[int]
        :param cids:
            The ids of the columns to be subscribed.
        
        .. seealso::

            :py:meth:`libgear.storage.CpuTrajectoryStorageHandler.raw`.
        """

        if cids is None:
            cids = range(self.num_col)

        return tuple([self._handler.view(index, cid) for cid in cids])

    def get_tensor_view(
        self, index: int, cids: Sequence[int] = None
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """
        Get  memory subscriptions in tensor view of the underlying columns of a trajectory.

        :type index: int
        :param index:
            The index of the corresponding trajectory id and the trajectory id should be on the local node.

        :type cids: Sequence[int]
        :param cids:
            The ids of the columns to be subscribed. All columns will be returned if no column specified.
        
        .. seealso::

            :py:meth:`libgear.storage.CpuTrajectoryStorageHandler.view`.
        """
        if cids is None:
            cids = range(self.num_col)
        raw_views = self.get_raw_view(index, cids)
        return tuple(
            [
                raw_views[idx]
                .cast_tensor(self.get_column_dtype(cid))
                .view(self.get_column_shape(cid))
                for idx, cid in enumerate(cids)
            ]
        )

    def checkpoint(self, path: Union[pathlib.Path, str]):
        """
        Dump the checkpointed state to the disk.

        :type path: Union[pathlib.Path, str]
        :param path: Dump path.
        """
        ensure_path(path)

        with open(path, "wb") as of:
            pickler = pickle.Pickler(of)
            pickler.dump(
                SharedDatasetMeta(
                    self._table.get_table_spec(),
                    IndexMeta(
                        self._iset.global_capacity,
                        self._iset.local_capacity,
                        self._iset.index_offset,
                        self._iset.is_sharing,
                        self._iset.shm_key,
                    ),
                )
            )
            print("meta dumped")
            pickler.dump(self._table)
            print("table dumped")
            pickler.dump(self._iset.get_state())

    def load(
        self,
        path: Union[pathlib.Path, str],
        key: Union[int, None] = None,
        create: bool = True,
        proc_rank: int = 0,
    ):
        """
        Invoked by an empty SharedDataset to load the dumped state.

        :type path:  Union[pathlib.Path, str]
        :param path:
            Path pointing to the state to load.

        :type key: Union[int, None]
        :param key:
            Set the shared key, which should be shared among the processes which attaching to the same trajectory table.

        :type create: bool
        :param create:
            Whether create or attach to the trajectory table. Practically only the first process within a data-sharing group/node create the table, with other processes attaching to the table with ``create=False``

        """
        self._release_resources()

        if key is None:
            key = get_local_node_hashed_int32()

        rng = random.Random(key)
        table_key = rng.randint(0, KEY_T_RANGE)
        iset_key = rng.randint(0, KEY_T_RANGE)

        path = pathlib.Path(path)

        with open(path, "rb") as f:
            unpickler = pickle.Unpickler(f)
            meta: SharedDatasetMeta = unpickler.load()
            if create:
                self._table = unpickler.load()
                self._iset = glibi.Indexset.load_state(
                    unpickler.load(), meta.index.shared, iset_key, True
                )
            else:
                self._table = glibs.TrajectoryTable(meta.storage, table_key, False)
                self._iset = glibi.Indexset(
                    meta.index.global_capacity,
                    meta.index.local_capacity,
                    meta.index.index_offset,
                    meta.index.shared,
                    iset_key,
                    False,
                )
            self._entry_cls = namedtuple(
                "SharedDatasetEntry",
                ["weights", "timesteps"]
                + list(
                    [
                        f"column{i}"
                        for i in range(self._table.get_table_spec().num_columns)
                    ]
                ),
            )

        self._ensure_attached()
        self._handler = glibs.get_cpu_handler(
            self._table,
            self._iset.global_capacity,
            glib.Range(0, 0),
            glib.Range(
                self._iset.index_offset,
                self._iset.index_offset + self._iset.local_capacity,
            ),
        )
        return self

    def _release_resources(self):
        if self._table:
            del self._table
        if self._iset:
            del self._iset

    def _ensure_attached(self, blocking: bool = True):
        resources_list = [self._table]
        for res in resources_list:
            rc = -1
            while rc == -1:
                rc = res.connect()
                if not blocking:
                    break
