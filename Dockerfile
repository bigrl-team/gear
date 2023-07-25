FROM nvcr.io/nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

ARG ARG_TIMEZONE=UTC

ENV ENV_TIMEZONE ${ARG_TIMEZONE}
ENV TERM xterm-256color

WORKDIR /root/

RUN apt update \
  && echo '$ENV_TIMEZONE' > /etc/timezone \
  && ln -sfn /usr/share/zoneinfo/$ENV_TIMEZONE /etc/localtime \
  && apt install -y -q apt-utils locales systemd cron vim wget \
  git build-essential libibverbs-dev openssh-server

RUN /etc/init.d/ssh start

RUN git clone https://github.com/NVIDIA/nccl.git \
  && cd nccl \
  && make -j src.build

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
  && bash ~/miniconda.sh -b -p $HOME/miniconda

RUN echo 'eval "$(~/miniconda/bin/conda shell.bash hook)"' >> /root/.bashrc \
  && echo 'export PATH=$PATH:/usr/local/cuda/bin' >> /root/.bashrc \
  && echo 'export NCCL_HOME=/root/nccl/build:$NCCL_HOME' >> /root/.bashrc

ENV NCCL_HOME /root/nccl/build

RUN git clone git@github.com:bigrl-team/gear.git \
  && cd gear \
  && eval "$(~/miniconda/bin/conda shell.bash hook)" \
  && conda create -n gear python==3.10 -y \
  && conda activate gear \
  && pip install torch==1.13 --index-url https://download.pytorch.org/whl/cu117 \
  && pip install -r requirements.txt 