# GEAR Project

<p align="center">
<img src=figs/icon.jpg width=512/>
</p>

Welcome to the official Github Repository for <font color=rgb(21,110,175)>GEAR</font>, a GPU-centric experience replay system supporting training Reinforcement Learning(RL) models at scale. 

Recent works have shown large sequence models's impressive capability in RL challenges, such as multi-agent, multi-model and multi-task senarios. To make the training affordable, RL models are often trained with past experience and parallel framework. Given plenty exploration on the accelerating training with distributed GPU servers and multi-dimensional parallelism, the pivotal challenge that hinders the scaling of these large RL models relies in the experience replay systems. The systems are required to efficiently manage massive volume of experience data(~100TBs), select large trajectory batch(~K trajectories) and distribute data batch to the training servers according to various parallelism schemas. Traditional experience replay systems fail to fulfill the requirements, which inspires the design of GEAR.

![GEAR-Overview](figs/gear-overview.jpg "GEAR-Overview")
<center><b>GEAR-Overview</b></center>

GEAR bridges online/offline RL data with parallely trained models. As illustrated in the overview, GEAR serve as a experience replay system enabling distributed trajectory storage and management, GPU accelerated trajectory selection and collection, featuring:
* Efficient: GEAR proposes comprehensive optimizations towards data accessing & computation patterns of RL training workflows, including data locality in distributed training, CUDA kernels for trajectory selection, zero-copy host memory accessing from GPUs and etc.
* Scalable: GEAR revisit the hardware capability(RAM, GPU, NICs) of modern GPU-based training servers, which are often underutilized in conventional DL tasks. Combining optimizations in architectural design and utilization of hardware resources, GEAR can achieve better scability in distributed RL training.
* Usable: GEAR provides clean and highly usable interface for distributed deployment, enabling its seamless intergration with existing distributed DL systems like DeepSpeed.




## Installation
GEAR is designed as a framework-independent library and for usability we have implemented an Python interface for its integration with [PyTorch](https://github.com/pytorch/pytorch), the popular open-sourced Tensor library, and [DeepSpeed](https://github.com/microsoft/DeepSpeed), a widely-used distributed deep learning optimization library. Therefore, the following dependencies have to be met before installing GEAR.

```plain
torch==1.13.0+cu117
deepspeed
```

In this section, we will show how to make a clean installation for GEAR.

### Prerequisite

#### NVIDIA Driver
Since GEAR's features and implementations have hard dependencies on CUDA, please verify the system(s) which you would like to GEAR installed has its nvidia driver(driver version >= 117) properly setup. For a Ubuntu-based(we recommended ubuntu20.04 LTS) system, nvidia drivers can be installed with the following instructions as listed by NVIDIA:
```shell
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
$ wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
$ sudo dpkg -i cuda-keyring_1.0-1_all.deb
$ sudo apt-get update
```

After updating the cache of APT repos, drivers can be installed with:
```shell
$ sudo apt install nvidia-dkms-530 nvidia-utils-530
```

The above instructions will install the CUDA-12.1 driver and utilities on the system. After reboot, a properly set driver will given an output by executing the shell command:
```shell
$ nvidia-smi
```

More detailed instructions can be found on the [official documentation](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#introduction).

#### CUDA Toolkit

The compilation of GEAR requires the CUDA Compiler, namely *nvcc*, installed on the system, which is included in the *cuda-toolkit* package. With APT repo set as described in the *NVDIA Driver* section, you may now able to search and install CUDA Toolkit with *APT* package manager:

```shell
$ apt search cuda-toolkit
Sorting... Done
Full Text Search... Done
cuda-toolkit/unknown 12.2.0-1 amd64
  CUDA Toolkit meta-package

cuda-toolkit-11-0/unknown 11.0.3-1 amd64
  CUDA Toolkit 11.0 meta-package

cuda-toolkit-11-1/unknown 11.1.1-1 amd64
  CUDA Toolkit 11.1 meta-package

cuda-toolkit-11-2/unknown 11.2.2-1 amd64
  CUDA Toolkit 11.2 meta-package

cuda-toolkit-11-3/unknown 11.3.1-1 amd64
  CUDA Toolkit 11.3 meta-package
....
```

Since CUDA Driver offers forward compatiblity cuda-toolkit, the choice of cuda-toolkit version can vary, cuda-toolkit-11-7 is recommanded(matched version with the PyTorch binary).
```shell
$ apt install cuda-toolkit-11-7
```

Run the following shell commands to check whether cuda-toolkit is successfully installed:
```shell
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:49:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0

$ which nvcc
/usr/local/cuda-11.7/bin/nvcc
```

#### NCCL
GEAR have [*NCCL*](https://github.com/NVIDIA/nccl) as their dependency, which can be installed via:
```shell
$ git clone https://github.com/NVIDIA/nccl.git
$ cd nccl
$ make -j src.build
```

After building the NCCL library, set the *NCCL_HOME* environment variable in your shell before installing GEAR:
```shell
export NCCL_HOME=<path-to-nccl-git-cloned-dir>/build
```

#### ibVerbs
We are actively working on one-sided access feature for remote data, which relies on the InfiniBand and ibVerbs primitives. To install the ibverb dependency, you can install ``libibverbs`` via *APT* package manager:
```shell
apt install libibverbs-dev
```

#### Multi-node Support(Optional)
In the current state, GEAR provide distributed training interface with DeepSpeed, which relies on ``pdsh`` tool for distributed task launching. To install ``pdsh`` with *APT*:
```shell
$ apt install pdsh -y
```

### GEAR Installation Guide

#### Step 1: Create Python Environment

We recommand using [*conda*](https://docs.conda.io/en/latest/miniconda.html) to create a clean python environment. The following steps will guide you through the installtion process on an amd64 linux system:
```shell
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
$ ./miniconda.sh
``` 
Create a conda environment:
```shell
$ conda create -n gear-dev python==3.10 -y
Retrieving notices: ...working... done
Collecting package metadata (current_repodata.json): done
Solving environment: failed with repodata from current_repodata.json, will retry with next repodata source.
Collecting package metadata (repodata.json): / 
...
```
#### Step 2: Install PyTorch with CUDA Support
Please note, installing PyTorch directly via pip using the command ``pip install torch`` could result in a PyTorch installation that lacks CUDA support. Therefore, we recommend installing PyTorch using the --index-url option for complete functionality

```shell
$ pip install torch==1.13 --index-url https://download.pytorch.org/whl/cu117
```

You can check the CUDA support of your current PyTorch installation via:
```shell
$ python -c "import torch; print(torch.cuda.is_available())"
True
```

#### Step 3: Clone GEAR's Repo and Install
```shell
$ git clone https://github.com/bigrl-team/gear.git
$ cd gear
$ pip install -r requirements.txt
$ pip install .
```

Check GEAR's installation with command:
```shell
$ python -c "import gear; gear.check_visible_device()"
CHECK visible_device_count.......................................4.
```


## Quick Start with Docker
Users may find it difficult to get hands on GEAR since there exists a bunch of dependencies. Hence we provide a dockerfile which can be utilized to quickly setup a docker container with GEAR-ready environment. Before starting, make sure that ``nvidia-ctk``, namely [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), is properly installed on your local system.
```shell
$ cd gear
$ docker build -t gear:v0.3 .
$ docker run -itd --gpus all gear:v0.3 /bin/bash 
$ docker exec -it <container-id> /bin/bash
```

Then in the container ``bash`` shell:
```shell
$ cd gear; eval "$(~/miniconda/bin/conda shell.bash hook)"; conda activate gear; pip install -e .; python -c "import gear; gear.check_visible_device()"
```

## Evaluation

<center>
<figure>
<img src=./figs/e2e.jpg width=256 />     <img src=./figs/tp.jpg width=256 />
</figure>
</center>
<center><b>(Left)End-to-end throughput comparison with Reverb.(Right)Trajectory collection throughput with varied batch sizes</b></center>

As shown in end-to-end throughput comparison with Reverb, GEAR achieves 3x performance under single-node settings and 6x performance under multi-node settings. And GEAR also show linear scalability w.r.t larger batch sizes, which is pratical in training large RL models at scale.




## Acknowledgement

### Third-party libraries:
* [**Infinity**](https://github.com/claudebarthels/infinity): Infinity is a simple highly-usable yet powerful  library offering abstractions of ibverbs. Certain releases of GEAR are built upon Infinity for RDMA support.
* [**PyTorch**](https://github.com/pytorch/pytorch): PyTorch is a popular machine learning framework build upon an extensible tensor library. GEAR provide native conversions between its data/memory interface and torch::Tensor to help users integret GEAR with their existing PyTorch models/code.
* [**DeepSpeed**](https://github.com/microsoft/DeepSpeed): DeepSpeed is a widely-used distribtued deep learning library with extensible components, GEAR provides iterfaces to make it easily integreted with existing DeepSpeed applications.


## Cite GEAR

```bibtex
@article{wang2023gear,
  title={GEAR: A GPU-Centric Experience Replay System for Large Reinforcement Learning Models},
  author={Wang, Hanjing and Sit, Man-Kit and He, Congjie and Wen, Ying and Zhang, Weinan and Wang, Jun and Yang, Yaodong and Mai, Luo}
}
```