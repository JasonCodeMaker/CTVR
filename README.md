# Continual Learning for Video-Text Retrieval Benchmark

## Content

- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Usage](#usage)
- [Citation](#citation)

## Prerequisites

It is essential to install all the dependencies and libraries needed to run the project. To this end, you need to run this line: 

## Dataset

We provide the metadata for each Video Continual Learning (CL) setup proposed in this benchmark. This metadata contains the data subsets corresponding to the set of tasks of each CL setup.  However, you have to download the video datasets required by the proposed CL setups and extract the frames of videos. 

### Data
#### UCF101
- [UCF101 - 10 Tasks](https://github.com/ojedaf/vCLIMB_Benchmark/raw/main/data/UCF101_data.pkl)
- [UCF101 - 20 Tasks](https://github.com/ojedaf/vCLIMB_Benchmark/raw/main/data/UCF101_data_20tasks.pkl)
#### Kinetics400
- [Kinetics400 - 10 Tasks](https://github.com/ojedaf/vCLIMB_Benchmark/raw/main/data/Kinetics400_data_tasks_10.pkl)
- [Kinetics400 - 20 Tasks](https://github.com/ojedaf/vCLIMB_Benchmark/raw/main/data/Kinetics400_data_tasks_20.pkl)
#### ActivityNet
- [ActivityNet - 10 Tasks](https://github.com/ojedaf/vCLIMB_Benchmark/raw/main/data/ActivityNet_data_10tasks.pkl)
- [ActivityNet - 20 Tasks](https://github.com/ojedaf/vCLIMB_Benchmark/raw/main/data/ActivityNet_data_20tasks.pkl)
##### Configuration for Trimmed Version:
- is_activityNet: True
- train_per_noise: 0
- val_per_noise: 0
- co_threshold: 0
##### Configuration for Untrimmed Version:
- is_activityNet: True
- train_per_noise: 1
- val_per_noise: 1
- co_threshold: 0


## Usage

The configuration file must be created or modified according to the provided examples, the code path, your particular requirements, and the selected setup.


## Citation

If you find this repository useful for your research, please consider citing our paper:

