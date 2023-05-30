from abc import abstractmethod

import torch


class DistributedSampler:
    def __init__(self, ratio, seed) -> None:
        self._ratio = ratio

        if seed:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)
        else:
            self._generator = None

    @property
    def generator(self):
        return self._generator

    @property
    def ratio(self):
        return self._ratio

    @abstractmethod
    def sample(self):
        raise NotImplementedError()

    @abstractmethod
    def sync(self):
        raise NotImplementedError()
