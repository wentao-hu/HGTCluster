
# HGTCluster
This project is based on [alibaba/graph-learn](https://github.com/alibaba/graph-learn), a distributed framework designed for the development and application of large-scale graph neural networks. We attempt to adapt [Heterogeneous Graph Transformer](https://arxiv.org/pdf/2003.01332.pdf) into self-supervised setting for two aims:

1. Learn better user embedding, so we can improve the performance of u2u2i recall channel.
2. Learn better user cluster, so that we can boost the weight of notes from the users in the same cluster.

## Method 
#### baseline: TwHIN+KMeans
baseline model first uses [TwHIN](https://arxiv.org/pdf/2202.05387.pdf) to learn user embedding. It uses [TranE](https://proceedings.neurips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf) to embed nodes and edges. After getting user node embedding, it uses KMeans to cluster user nodes. This method suffer from two problems:
1. It doesn't consider information from further neighbors such as 2-hop neighbors.
2. The learning objective of TwHIN is to classify positive triplets and negative triplets. It is not consistent with downstream KMeans clustering.

#### Improved model 1: HGT+KMeans
To solve the 1-st problem, we use Heterogeneous Graph Transformer to learn better user embedding with 2-hop neighbors at first stage. The second stage we still use KMeans for clustering.


### Improved model 2: HGT+SOM


### Dataset
Our fake dataset contain two types of nodes: ['user','note'] and two types of edges ['follow_note','share_note']. Our model can be easily extended to more node types and edge types in production setting.

### Distributed training
On real production data, we run code with parameter-server architecture including parameter-server node, worker node and graphlearn node. graphlearn node is for building graph and sampling neighborhood nodes. We have 20 graphlearn nodes, so our data is partitioned into 20 parts at distributed setting.

After building the above distributed training enviroment, you can run our distributed version code with the following steps:
1. generate fake data for distributed setting: 
```
cd gen_fake_data
python gen_fake_node_dist.py
python gen_fake_edge_dist.py
```

2. train model with distributed trainer:
```
python dist_train_hgt.py
```

### Local training 
For demonstration, you can locally run demo code with the following steps:
1. generate fake data for local setting:
```
cd gen_fake_data
python gen_fake_node_local.py
python gen_fake_edge_local.py
 ```
2. train model with local trainer: 
```
python local_train_hgt.py
```
3. local training results on local fake data
```
2023-08-19 12:45:30.728442 Epoch 0, Iter 10, LocalStep/sec 3.03, Time(s) 0.3299, Loss 0.70043, Accuracy 0.32118
2023-08-19 12:45:30.891671 Epoch 0, Iter 20, LocalStep/sec 61.28, Time(s) 0.0163, Loss 0.72848, Accuracy 0.33388
2023-08-19 12:45:31.055665 Epoch 0, Iter 30, LocalStep/sec 60.99, Time(s) 0.0164, Loss 0.71306, Accuracy 0.32597
2023-08-19 12:45:31.219610 Epoch 0, Iter 40, LocalStep/sec 61.01, Time(s) 0.0164, Loss 0.68536, Accuracy 0.32973
2023-08-19 12:45:31.383051 Epoch 0, Iter 50, LocalStep/sec 61.20, Time(s) 0.0163, Loss 0.67499, Accuracy 0.33195
2023-08-19 12:45:31.547282 Epoch 0, Iter 60, LocalStep/sec 60.91, Time(s) 0.0164, Loss 0.68997, Accuracy 0.33554
......
```


### Requirements
```
python: 2.7
tensorflow: 1.12.1
graphlearn: 1.0.1
```

### TODO
- 画一个HGT+KMeans的示意图
- 画一个HGT+SOM的示意图
- 项目readme要不断更新完善