This repo is the implementation of "Data augmentation techniques on graph structures for table type classification" paper, submitted to S+SSPR 2022 workshop. Once our work will be accepted, we will upload the full script to reproduce our experiments.

#### Table of contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Configuration](#configuration)


## Introduction

[Data Augmentation On Graphs For Table Type Classification](https://github.com/AILab-UniFI/DA-GraphTab) is a project with the purpose of identifying data augmentation techniques on table datas.


## Setup

1. Clone the repository into your terminal
```sh
git clone https://github.com/AILab-UniFI/DA-GraphTab.git
```
2. Install requirements from "requirements.txt".
3. Create ".env" file and set ROOT project folder.
4. Inside "paths.py" file you can change project's paths as you prefer. You can also leave it unchanged.
5. In "graph.yaml" inside "configs" directory, you can change some parameters about graph builder, data loader and training model.


## Configuration

1. Create and build graphs from input pdf files (in "raw" folder) and "metadata.json" file (containing some useful information about these papers) using "loader.py". So you can run this file to create graph dataset.
2. Now you can train and test model using "model_predict.py" and changing main variables as you prefer. Results are available in "ouput" folder.

## Cite this project
If you want to use our dataset in your project[^1], please cite us:
```
@misc{https://doi.org/10.48550/arxiv.2208.11210,
  doi = {10.48550/ARXIV.2208.11210},
  url = {https://arxiv.org/abs/2208.11210},
  author = {del Bimbo, Davide and Gemelli, Andrea and Marinai, Simone},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Data augmentation on graphs for table type classification},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}
```
