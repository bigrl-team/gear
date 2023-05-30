from .index import *
from .memory.memory import *
from .storage.dtypes import *
from .storage.handler import *
from .storage.specs import *
from .storage.table import *

def init(device_id: int):
    """
    Initialize context settings for gear, such as CUDADevice, etc.

    Params:
        device_id: int, handler for local gpu
    """

    ...
