# BiHGH

Official Implementation of "Bi-directional Heterogeneous Graph Hashing towards Efficient Outfit Recommendation".

![](framework.jpg)

## Abstract

> Personalized outfit recommendation, which aims to recommend the outfit to a user according to his/her preference, has gained increasing research attention due to its great practical economic value. Majority of existing methods mainly focus on improving the recommendation effectiveness, while overlook the recommendation efficiency. Inspired by this, we devise a novel bi-directional heterogeneous graph hashing scheme, BiHGH for short, towards efficient personalized outfit recommendation. In particular, this scheme consists of three key components: heterogeneous graph node embedding, bi-directional sequential graph convolution, and hash code learning. We first unify four types of entities (i.e., users, outfits, items, and attributes) and their relations with a heterogeneous four-partite graph. We then creatively devise a bi-directional sequential graph convolution paradigm to sequentially and repeatedly transferring knowledge from top-down and down-top directions, whereby we divide the four-partite graph into three subgraphs, each of which include two adjacent levels of entities. Finally, we adopt the commonly used Bayesian Personalized Ranking loss for the user preference learning, and design the bi-level similarity preserving regularization to prevent the information loss during the hash learning. Extensive experiments on three benchmark datasets demonstrate the superiority of BiHGH.

## Data Preparation

We provide all data files used in the `/data` folder, including edge files and node initialization features used to build the graph. At the same time, we provide the data preprocessing code to obtain these data files, see the `/preprocess` folder.

## Environment
   python 3.9.0
   
   pytorch 1.8.1
   
   You can install all the dependencies required for execution with the following command.
   
    pip install -r requirements.txt

## Configuration

Before start, the configuration file `conf/gat_tf_emb_max_v1.yaml` is supposed to be modified first.

### Train

    CUDA_VISIBLE_DEVICES=[gpu_id] python trainer.py
    

After the training is completed, the model file, configuration file and tensorboard log file will be saved to the `/experiments` folder.
### Test

Then, you can read the saved model for testing with the following command.

    CUDA_VISIBLE_DEVICES=[gpu_id] python predict.py

## Note

Any question please contact me by email: zhang.hy.2019@gmail.com

## Citation

If this work is helpful, please cite it:

> @inproceedings{10.1145/3503161.3548020,
> author = {Guan, Weili and Song, Xuemeng and Zhang, Haoyu and Liu, Meng and Yeh, Chung-Hsing and Chang, Xiaojun},
title = {Bi-Directional Heterogeneous Graph Hashing towards Efficient Outfit Recommendation},
year = {2022},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3503161.3548020},
doi = {10.1145/3503161.3548020},
booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
pages = {268–276},
numpages = {9}
}

