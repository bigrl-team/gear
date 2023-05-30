import torch
import pickle as pkl
import time

def valid_data_integrity(loader):
    import pickle as pkl

    with open("/tmp/gear/datasets/example.pt", "rb") as f:
        data = pkl.load(f)
    num_trajectory = len(data[list(data.keys())[0]])
    indices = torch.Tensor(list(range(num_trajectory))).long()
    # print(indices)

    torch.cuda.synchronize()
    
    loaded_data = loader.exposed_fused_iterate(indices)
    # tss, loaded_data = loader.exposed_low_level_controlled_iterate(indices)
    for i, k in enumerate(["observations", "actions"]):
        for tid in range(num_trajectory):
            length = loader._iset.timesteps[tid].item()
            loaded_ver = loaded_data[i][tid][:length]
            raw_ver = torch.from_numpy(data[k][tid]).cuda()
            print(
                loaded_ver.shape,
                raw_ver.shape,
                loaded_data[i][tid].shape,
                loader._iset.timesteps[tid],
            )
            assert torch.all(loaded_ver == raw_ver)

    torch.cuda.synchronize()


def benchmark(loader, num_iter):
    for i in range(num_iter):
        if i == num_iter / 2:
            start = time.perf_counter()
        data_batch = next(loader)
    end = time.perf_counter()

    if dist.get_rank() == 0:
        s = np.sum([d.numel() * d.element_size() for d in data_batch]) / (10**9)
        print(list(d.shape for d in data_batch))
        print(
            f"AVG throughput {loader._dp_world * s * num_iter * 0.5 / (end-start)} GB/s"
        )