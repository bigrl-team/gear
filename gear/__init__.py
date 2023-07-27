import torch
import libgear as glib

from . import comm, dataset, dtypes, errors, loader, mpu, sampler, specs
from .dtypes import DataType


def init(device_id: int = None):
    """
    GEAR initialization function, should be called after setting
    "LOCAL_RANK" env var in the system environment, the value of
    "LOCAL_RANK" will be used for subscripting local cuda devices.

    similar to "cuda_device = torch.device(f"cuda:{LOCAL_RANK})"
    """

    if device_id is None:
        import os

        try:
            device_id = int(os.environ["LOCAL_RANK"])
        except KeyError as ke:
            device_id = 0
    glib.cuda.init(device_id)


__all__ = [
    "initialize",
    "comm",
    "loader",
    "dataset",
    "sampler",
    # dtypes
    "dtypes",
    "DataType",
    "errors",
    "mpu",
    "specs",
]


def check_visible_device():
    glib.print_environ()
