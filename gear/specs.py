from typing import Any, Dict, Sequence, Union

import torch
import libgear.storage as glibs
from gear.dtypes import CVT_DTYPES_TORCH_TO_GEAR, DataType


class ColumnSpec:
    """
    A factory class that helps the creation procedure of :py:class:`libgear.storage.ColumnSpec`.
    """

    @staticmethod
    def create(
        shape: Sequence[int],
        dtype: torch.dtype = torch.float32,
        name: str = "",
        *args,
        **kwargs
    ) -> glibs.ColumnSpec:
        """
        Factory function of ColumnSpec.

        :type shape: Sequence[int]
        :param shape:
            Describe the tensor shape of a single frame within the field.

            **Note**: If a desired return for subscribing the column is of shape ``[batch_size, sequence_length, *frame_shape]``, then the ``shape`` argument should only contains contents of ``*frame_shape``

        :type dtype: torch.dtype
        :param dtype:
            Data type of the column tensor, providing hints for memory parsing and organization.

        :type name: str
        :param name:
            Column name in string format.

            .. seealso::
              :py:func:`libgear.storage.TableSpec.index`.

        :return:
            Constructed ColumnSpec instance.
        :rtype: :py:class:`libgear.storage.ColumnSpec`.
        """
        if isinstance(dtype, DataType):
            dtype = dtype
        else:
            dtype = CVT_DTYPES_TORCH_TO_GEAR[dtype]
        return glibs.ColumnSpec(shape, dtype, name)


class TableSpec:
    """
    A factory class that helps the creation procedure of :py:class:`libgear.storage.TableSpec`.
    """

    @staticmethod
    def create(
        rank: int = 0,
        worldsize: int = 1,
        trajectory_length: int = 100,
        capacity: int = 32,
        column_specs: Sequence[Union[Dict[str, Any], glibs.ColumnSpec]] = None,
        *args,
        **kwargs
    ) -> glibs.TableSpec:
        """
        Factory function of TableSpec.

        :type rank: int
        :param rank:
            The rank of the individual table. The ``rank`` param together with  the ``capacity`` param determine the index range ``[rank * capacity, (rank + 1) * capacity)``, whose underlying trajectory memory blocks are located on the same physical node that can be directly accessed via shared-memory.

        :type worldsize: int
        :param worldsize: int
            Total number of tables. The overall capacity of the distributed trajectory storage is equal to ``worldsize * capacity``.

        :type trajectory_length: int
        :param trajectory_length: int
            The maximum sequence length/timesteps of a trajectory. Data of a single column/field is stored in a continuous memory block of size ``trajectory_length * column_shape``, facilitating time-ranged request in experience replay.

        :type capacity: int
        :param capacity:
            The maximum number of trajectories that can be stored in local table.

        :type column_specs: Sequence[Union[Dict[str, Any], :py:class:`libgeare.storage.ColumnSpec`]]
        :param column_specs:
            Grouped param dicts and :py:class:`libgeare.storage.ColumnSpec`. that describe all of the columns of a table.

        :return:
            Constructed TableSpec instance.
        :rtype: :py:class:`libgear.storage.TableSpec`.
        """
        num_columns = 0
        cspecs = []
        if column_specs is not None:
            num_columns = len(column_specs)
            for column_spec in column_specs:
                if isinstance(column_spec, glibs.ColumnSpec):
                    cspecs.append(column_spec)
                else:
                    cspecs.append(ColumnSpec.create(**column_spec))
        return glibs.TableSpec(
            rank, worldsize, trajectory_length, capacity, num_columns, cspecs
        )
