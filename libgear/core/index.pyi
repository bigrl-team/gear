import torch

class RawSimulStatus:
    def __init__(self) -> None: ...
    @property
    def ownership(self) -> int: ...
    @property
    def terminated(self) -> int: ...

class Indexset:
    """ """

    def __init__(
        self,
        global_capacity: int,
        local_capacity: int,
        index_offset: int,
        shared: bool,
        key: int,
        create: bool,
    ) -> None: ...
    @property
    def timesteps(self) -> torch.Tensor: ...
    @property
    def weights(self) -> torch.Tensor: ...

class SharedMemoryIndexServer:
    """
    A index server that
    """

    def __init__(
        self, key: int, num_clients: int, index_offset: int, capacity: int
    ) -> None:
        """
        test
        """
        ...
    def connect(self) -> int:
        """
        test
        """
        ...
    def release(self) -> None: ...
    def acquire(self) -> None: ...
