from abc import abstractmethod


class Communicator:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def send(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def recv(self, *args, **kwargs):
        raise NotImplementedError()
