import pytest
import pickle as pkl
import torch

import gear
from gear.specs import ColumnSpec, TableSpec
from gear.dtypes import CVT_DTYPES_GEAR_TO_TORCH as CVT_GTT


@pytest.mark.parametrize(
    "shape, dtype, name",
    [
        ((3,), torch.bool, "bool"),
        ((3, 4), torch.uint8, "uint8"),
        ((3, 4, 5), torch.int8, "int8"),
        ((3, 4, 5, 6), torch.int32, "int32"),
        ((3, 4, 5, 6, 7), torch.int64, "int64"),
        ((3, 4, 5, 6, 7, 8), torch.float32, "float32"),
        ((3, 4, 5, 6, 7, 8, 9), torch.float64, "float64"),
    ],
)
class TestColumnSpec:
    def test_column_spec_build(self, shape, dtype, name):
        cspec = ColumnSpec.create(shape, dtype, name)
        assert tuple(cspec.shape) == shape
        assert CVT_GTT[cspec.dtype] == dtype
        assert cspec.name == name

    def test_column_spec_pickle(self, shape, dtype, name):
        cspec = ColumnSpec.create(shape, dtype, name)
        cspec_recoverd = pkl.loads(pkl.dumps(cspec))
        assert cspec.shape == cspec_recoverd.shape
        assert cspec.dtype == cspec_recoverd.dtype
        assert cspec.name == cspec_recoverd.name

