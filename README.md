This repository contains the source code implementation for the paper "Group-based Interleaved Pipeline Parallelism for Large-scale DNN Training"

## Directory Structure

### `lib`

WPipe's runtime, which implements model parallelism, input pipelining, 
as well as as comunication in PyTorch. This can be fused with data parallelism to give hybrid
model and data parallelism, and input pipelining.

### `cv`

Image classification  task entry point, as well as splits of model

### `nlp`

NLP task entry point, as well splits of model

### `network_conf`

Experiments configurations

### `experiments`

Experiments running scripts

### `tool`

Some helper scripts


## Setup

### Software Dependencies

To run WPipe, you will need a NVIDIA GPU with CUDA 10.1, GPU driver version 418.67, nvidia-docker2,
and Python 3. On a Linux server with NVIDIA GPU(s) and Ubuntu 16.04

All dependencies are in the `pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime` container, which can be downloaded using:

```bash
docker pull pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
```

The PyTorch Docker Container can then be run using:

```bash
nvidia-docker run -it -v /mnt:/mnt --ipc=host --net=host pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime /bin/bash
```

### Initialization

Before runing wpipe program, 

```bash
cd tool && sh init.sh
```

### Data Prepare

#### CV

We run experiments for fine-tune using cifar10, cifar100 and oxford-flower-102,
and the throughput experiments using oxford-flower-102 dataset

To download cifar10 and cifar100 from [this website](http://www.cs.toronto.edu/~kriz/cifar.html)
To download oxfordflowers102 from [this website](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

#### NLP

We run experiments for fine-tune and throughput using a subset of the GLUE dataset(QQP and MNLI).
To download the GLUE dataset use [this script](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py).

### Run experiments

All experiments can be carried out using scripts in the experiments directory.
You can perform an experiment as follows:

```
sh experiments/cv_throughput_single_node.sh
```
