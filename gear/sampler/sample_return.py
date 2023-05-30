from typing import NamedTuple

import torch


class SampleReturn(NamedTuple):
    indices: torch.IntTensor
    weights: torch.FloatTensor
    partial_sum: torch.FloatTensor  # zero-dim float32 tensor
    batch_size_per_rank: int
    oversample_ratio: float

    @staticmethod
    def replica(sample_return: "SampleReturn", num: int = 1) -> "SampleReturn":
        assert num > 0 and isinstance(
            num, int
        ), "num replica should be a postive integer"
        assert (
            len(sample_return.partial_sum.size()) == 0
        ), "parital_sum should be a zero-dim float tensor"

        if num == 1:
            return SampleReturn(
                sample_return.indices.unsqueeze(0),
                sample_return.weights.unsqueeze(0),
                sample_return.partial_sum.unsqueeze(0),
                sample_return.batch_size_per_rank,
                sample_return.oversample_ratio,
            )
        else:
            return SampleReturn(
                torch.stack([sample_return.indices] * num, dim=0),
                torch.stack([sample_return.weights] * num, dim=0),
                torch.stack([sample_return.partial_sum.unsqueeze(0)] * num, dim=0),
                sample_return.batch_size_per_rank,
                sample_return.oversample_ratio,
            )
