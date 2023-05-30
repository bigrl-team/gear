from abc import abstractmethod


class BasicLoader:
    def __init__(self, batch_size) -> None:
        self._batch_size = batch_size

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def checkpoint(self):
        pass