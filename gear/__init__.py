from typing import Union

import torch
import libgear as glib

from . import comm, dataset, dtypes, errors, loader, mpu, sampler, specs
from .dtypes import DataType


def init(device_id: Union[int, None] = None) -> None:
    """
    GEAR cuda env init function, if a valid int between [0, node_device_count) representing tthe device_id not set in the function call, the "LOCAL_RANK" environment variable in the system environment will be used for subscripting local cuda devices.

    similar to "cuda_device = torch.device(f"cuda:{LOCAL_RANK})"

    :param device_id(optional): int

    :rtype: None
    :return:
        None
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
    """
    Invokes :py:func:`libgear.print_environ`.

    """
    glib.print_environ()
