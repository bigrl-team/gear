import torch

checkpoint_path = "/tmp/checkpoints/example_shared_dataset.pt"

table_spec_params = {
    "rank": 0,
    "worldsize": 1,
    "trajectory_length": 1000,  # Literal["Uniform", "Weighted", "Topk"]
    "capacity": 1000,
    "column_specs": [
        {"shape": [3, 32], "dtype": torch.float32},
        {"shape": [1], "dtype": torch.int32},
    ],
}

offline_loader_params = {
    "data_path": checkpoint_path,
    "mpu": None,
    "batch_size": 2,
    "sampling_method": "Uniform",
    "patterns": [
        {
            "pad": "head",  # Literal["head", "tail"]
            "offset": -100,
            "length": 100,
        },
        {
            "pad": "head",  # Literal["head", "tail"]
            "offset": -100,
            "length": 100,
        },
    ],
}


mock_rng = torch.Generator()
mock_rng.manual_seed(42)
mock_data = None


def build_mock_data():
    global mock_data
    mock_data = {
        "weights": torch.rand(table_spec_params["capacity"], dtype=torch.float32),
        "columns": dict(
            {
                cid: torch.ones(
                    (
                        table_spec_params["capacity"],
                        table_spec_params["trajectory_length"],
                    )
                    + tuple(cspec["shape"]),
                    dtype=cspec["dtype"],
                )
                for cid, cspec in enumerate(table_spec_params["column_specs"])
            }
        ),
        "timesteps": torch.ones(table_spec_params["capacity"], dtype=torch.long) * 100,
    }
