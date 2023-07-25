# GEAR Project

<p align="center">
<img src=https://github.com/bigrl-team/gear/blob/main/icon.jpg width=512/>
</p>

Welcome to the official Github Repository for <font color=rgb(21,110,175)>GEAR</font>, a GPU-centric experience replay system supporting training large RL models. This repository will serve as a collaborative platform for hosting and managing source code and actively accepting community feedback, bringing together our develop team, users and anyone who would like to generously offer their insights and comments for the projects.

Our vision for GEAR is to create an open-source project that stands out for its innovation, usability and contribution to the RL community. In the current state, however, due to time contraints the released version of GEAR may not include all the features and functionality we have planned for it. We are actively working on code cleaning and development and will be gradually open-sourcing the project.

If you have any questions or suggestions, feel free to reach out to us. Your feedback is invaluable for GEAR. Thank you for your understanding and support!

## Installation
GEAR is designed as a framework-independent library and for usability we have implemented an Python interface for its integration with [PyTorch](https://github.com/pytorch/pytorch), the popular open-sourced Tensor library, and [DeepSpeed](https://github.com/microsoft/DeepSpeed), a widely-used distributed deep learning optimization library. Therefore, the following depencies have to be met before installing GEAR.

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


## Acknowledgement

### Third-party libraries:
* [**Infinity**](https://github.com/claudebarthels/infinity): Infinity is a simple highly-usable yet powerful  library offering abstractions of ibverbs. Certain releases of GEAR are built upon Infinity for RDMA support.
* [**PyTorch**](https://github.com/pytorch/pytorch): PyTorch is a popular machine learning framework build upon an extensible tensor library. GEAR provide native conversions between its data/memory interface and torch::Tensor to help users integret GEAR with their existing PyTorch models/code.
* [**DeepSpeed**](https://github.com/microsoft/DeepSpeed): DeepSpeed is a widely-used distribtued deep learning library with extensible components, GEAR provides iterfaces to make it easily integreted with existing DeepSpeed applications.