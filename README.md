# CS6208: Paper Review
> **Student:** Kong Lingdong (A0260240X)<br>
> **Title:** "User-Specific Recommendation with K-GCN"<br>
> **Time:** AY 2022-2023, Semester II

## About
This repository contains the code and implementation details of the Paper Review assignment for the CS6208 course. In this assignment, we review Knowledge Graph Convolutional Network (K-GCN), with an application on a movie recommendation scenarios.

Collaborative filtering is a traditional technique for solving user-specific recommendation problems, but it has several drawbacks, such as the sparsity of user-item interactions. [(Wang et al., 2018)](https://arxiv.org/abs/1803.03467) and [(Huang et al., 2018)](https://static.aminer.cn/upload/pdf/890/478/667/5b67b46f17c44aac1c8631c3_0.pdf) have used knowledge graphs (KG) to overcome these issues, which are heterogeneous graphs with `nodes` and `edges` representing item `attributes` and `relations`, respectively, and build feature- and connection-rich scenarios to improve precision. Graph neural networks, specifically graph convolutional networks (GCN), have become powerful tools for processing such data. [(Wang et al., 2019)](https://arxiv.org/abs/1904.12575) combined KG and GCN in recommendation systems to achieve good performance on multiple datasets. This assignment reviews this knowledge graph convolutional network (K-GCN) through a user-specific movie recommendation problem.

<p align="center">
  <img src="figure/framework.png" align="center" width="60%">
  <br>
  Fig. Illustrations of (a) A two-layer receptive field of an entity (blue node) in a KG. (b) The framework of K-GCN. Images adopted from (Wang et al., 2019).
</p>


## Installation
This codebase is tested with `torch==1.11.0` with `CUDA 11.3`. In order to successfully reproduce the results reported, we recommend to follow the exact same configuation. However, similar versions that came out lately should be good as well.

- Step 1: Create Enviroment
```
conda create -n my_kgcn python=3.10
```
- Step 2: Activate Enviroment
```
conda activate my_kgcn
```
- Step 3: Install PyTorch
```
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```
- Step 4: Install Necessary Libraries
```
pip install numpy matplotlib sklearn pandas
```

## Data Preparation
We subsample a 10\% subset of [MovieLens-20M](https://grouplens.org/datasets/movielens/20m/) as our dataset, where ratings greater than 3 are considered positive. We split the dataset into a training set and a test set in an 8:2 ratio and use the KG from [(Wang et al., 2019)](https://arxiv.org/abs/1904.12575).

Download the complete [MovieLens-20M](https://grouplens.org/datasets/movielens/20m/) using the following commands:
```
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
mv ml-20m/ratings.csv data/movie/
```

Alternatively, you can download the data from my Google Drive via the following link:<br>
🔗 https://drive.google.com/file/d/1nyqaNs-HboPjFuGowENWgQUsRBM-zJFV/view?usp=sharing.

Then, uncompress this `.zip` file and replace the current `data/movie/` folder.


## Getting Started
The K-GCN framework is implemented by the following components:
- `data_loader.py`: Prepare and load movie recommendation data.
- `model.py`: Reproduce the K-GCN model.
- `aggregator.py`: Reproduce the Sum and Concat aggragators in K-GCN.

The main scripts are provided in the `myKGCN.ipynb` notebook, with reproducible steps. Follow the procedures in this notebook then you can get the exact same outputs as those in the submitted report.

The configuration of the ablation study is attached as follows:

| # | Variant | Aggregator Type | Number of Iterations | Number of Embedding Dimensions |
| :-: | :-----: | :-------------: | :-------------: | :-------------: |
| (a) | `Iter1_Sum_Dim16` | Summation | 1 | 16 |
| (b) | `Iter2_Sum_Dim16` | Summation | 2 | 16 |
| (c) | `Iter2_Sum_Dim32` | Summation | 2 | 32 |
| (d) | `Iter1_Concat_Dim16` | Concatenation | 1 | 16 |
| (e) | `Iter2_Concat_Dim16` | Concatenation | 2 | 16 |
| (f) | `Iter2_Concat_Dim32` | Concatenation | 2 | 32 |

## Result

