import libgear
import torch


class Context:
    def __init__(self, rank=0, world_size=1, device="cuda") -> None:
        self._rank = rank
        self._world_size = world_size
        self._device = torch.device(device)

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size

    def get_device(self):
        return self._device


class MemoryRegion:
    def __init__(self, ptr, size, create_rank) -> None:
        self._ptr = ptr
        self._size = size
        self._create_rank = create_rank
        self._destroyed = False

    @property
    def active(self):
        return not self._destroyed

    @property
    def size(self):
        return self._size

    @property
    def raw_ptr(self):
        return self._ptr

    @property
    def create_rank(self):
        return self._create_rank

    def valid_memory_subscription(self, offset, stride) -> bool:
        return offset >= 0 and stride >= 0 and (offset + stride <= self.size)


class SharedMemoryRegion(MemoryRegion):
    def __init__(self, ptr, size, create_rank, key, create) -> None:
        super().__init__(ptr, size, create_rank)
        self._key = key
        self._create = create


class MemoryRegionSubscription:
    def __init__(self, memory_region_or_subscription, offset, stride) -> None:
        if isinstance(memory_region_or_subscription, MemoryRegion):
            assert memory_region_or_subscription.valid_memory_subscription(
                offset, stride
            ), "invalid subscription of MemoryRegion instance(s)"
            self._ptr = memory_region_or_subscription._ptr + offset
            self._size = stride
            self._src = memory_region_or_subscription
        elif isinstance(memory_region_or_subscription, MemoryRegionSubscription):
            assert memory_region_or_subscription.valid_memory_subscription(
                offset, stride
            ), "invalid subscription of MemoryRegionSubscription instance(s)"
            self._ptr = memory_region_or_subscription._ptr + offset
            self._size = stride
            self._src = memory_region_or_subscription._src
        else:
            raise RuntimeError("Non-supported type to subscribe from")

    @property
    def size(self):
        return self._size

    @property
    def active(self):
        return self._src.active

    def valid_memory_subscription(self, offset, stride) -> bool:
        return offset >= 0 and stride >= 0 and (offset + stride <= self.size)


class CUDAContext:
    def __init__(self, rank=0) -> None:
        self._context = libgear.Context()
        self._rank = rank

    def alloc(self, size):
        ptr = self._context.host_malloc(size)
        return MemoryRegion(ptr=ptr, size=size, create_rank=self._rank)

    def free(self, mr: MemoryRegion):
        assert isinstance(mr, MemoryRegion)
        assert (
            mr.create_rank == self._rank
        ), f"cannot free memoryRegion created from other context {mr.create_rank}, current context rank {self._rank}"

        if mr.active:
            self._context.host_free(mr._ptr)

    def register_shm(self, key, size, create):
        ptr = self._context.host_register_shm(key, size, create)
        return SharedMemoryRegion(
            ptr=ptr, size=size, create_rank=self._rank, key=key, create=create
        )

    def free_shm(self, mr: MemoryRegion):
        assert isinstance(mr, SharedMemoryRegion)
        self._context.host_free_shm(mr.key)

    def expose_via_rdma(self, smr: SharedMemoryRegion, server_config):
        self._rdma_server = libgear.RDMAServer()
