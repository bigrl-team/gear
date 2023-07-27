# Single-Node-Multi-GPU Offline Training Example
This example provides a minimal example for single-node-multi-gpu offline training an RL model with GEAR. The training dataset used in this example is collected from [D4RL](https://github.com/Farama-Foundation/D4RL) open-source dataset, licensed under [Creative Commons Attribution 4.0 License (CC BY)](https://creativecommons.org/licenses/by/4.0/). 


## Dependencies
To run this example, additional dependencies besides standard installation, which can be setup by running:

```shell
$ cd gear/examples/offline/single_node
$ pip install -r single_node_requirements.txt 
```


## Example Usage

### Step 1: Download the Dataset
We provide a code snippet in downloading the dataset, namely ``create.py``. By running
```shell
$ python create.py --hdf5_data_path=<download-path> --data_path=<path-for-the-converted-dataset>
```
The script is designed to download the ``Hopper-expert`` D4RL dataset and then transform it into GEAR's ``SharedDataset`` data structure. The SharedDataset provides an interface for altering and managing key experience data within a shared memory block in the node. This design allows multiple local training processes to access the same memory block, thereby enabling efficient data sharing.

Please be aware that the SharedDataset is intended to function as an intermediary data structure for dataset construction or as a read-only shared storage. It's important to note that the SharedDataset is not thread-safe, additional efforts are required if you intend to modify the content of a SharedDataset concurrently. This can be done by using separate indexing regions or distributed barriers. 

### Step 2: Run Training Scripts
Modify the training script ``run.sh`` by adding the argparse argument ``--data_path`` if you use a different path for the converted SharedDataset, otherwise it will be as simple as the following commands to run the example. We provide MLP and [Multi-Agent-Transformer(MAT)](https://github.com/PKU-MARL/Multi-Agent-Transformer) implementations in the example, which can be selected via command line arguments.
```
$ bash run.sh --model mlp --data_path <path>
(or)
$ bash run.sh --model mat --data_path <path>
```

The shell outputs and local tensorboard logs are expected:
```shell
Iteration: 0 evalution reward 9.236835588585471
Iteration: 10 evalution reward 9.1882160116987
Iteration: 20 evalution reward 14.362209263295448
Iteration: 30 evalution reward 32.71335036630997
Iteration: 40 evalution reward 42.059565945066886
Iteration: 50 evalution reward 58.90286469262947
Iteration: 60 evalution reward 69.48901672354052
Iteration: 70 evalution reward 108.51661991762988
Iteration: 80 evalution reward 140.5521157296422
Iteration: 90 evalution reward 147.95927004941876
[2023-07-24 12:25:17,424] [INFO] [logging.py:96:log_dist] [Rank 0] step=100, skipped=0, lr=[0.001], mom=[(0.9, 0.999)]
[2023-07-24 12:25:17,425] [INFO] [timer.py:215:stop] epoch=0/micro_step=100/global_step=100, RunningAvgSamplesPerSec=5720.511220447356, CurrSamplesPerSec=5370.427656850192, MemAllocated=0.01GB, MaxMemAllocated=0.12GB
```

An tensorboard log file will be generate under the ``logs`` folder.
<p align="center">
<img src=https://github.com/bigrl-team/gear/blob/main/examples/offline/single-node/figs/example_tensorboard_logs.png width=512/>
</p>

