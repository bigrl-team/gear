import pickle as pkl

import gear
import libgear.core as glibc
from gear.dtypes import DataType
from gear.specs import TableSpec


def test_table_pickling(table):
    with open("/tmp/gear_test_table_pickling.pkl", "wb") as of:
        pkl.dump(table, of)


if __name__ == "__main__":
    table_spec_params = {
        "rank": 0,
        "worldsize": 1,
        "trajectory_length": 100,
        "capacity": 1000,
        "column_specs": [
            {"shape": [3, 32], "dtype": DataType.float32},
            {"shape": [1], "dtype": DataType.int32},
        ],
    }

    tspec = TableSpec.create(**table_spec_params)
    table = glibc.TrajectoryTable(tspec, 7, True)
    table.connect()
    handler = glibc.get_cpu_handler(table)
    glibc.Uint8Span.to_tensor(handler.view(0, 0)).fill_(16)
    test_table_pickling(table)
