
# HGTCluster
Based on [alibaba/graph-learn](https://github.com/alibaba/graph-learn), we attempt to integrate [Heterogeneous Graph Transformer](https://arxiv.org/pdf/2003.01332.pdf) with self-supervised setting for two aims:

1. Learn better user embedding, so we can improve the performance of u2u2i recall channel.
2. Learn better user cluster, so that we can boost the weight of notes from the users in the same cluster.

## Method 
### baseline: TwHIN+KMeans
baseline model first uses [TwHIN](https://arxiv.org/pdf/2202.05387.pdf) to learn user embedding. It only considers 1-hop neighbors to form lots of triplets (s,r,t). And after getting user embedding, it uses KMeans to cluster users. This method suffer from two problems:
1. It doesn't consider information from further neighbors such as 2-hop neighbors.
2. The learning objective of TwHIN is to classify positive triplets and negative triplets. It is not consistent with downstream KMeans clustering.

### Improved model 1: HGT+KMeans
To solve the 1-st problem, we use Heterogeneous Graph Transformer to learn better user embedding with 2-hop neighbors at first stage. The second stage we still use KMeans for clustering.


### Improved model 2: HGT+SOM

- 画一个HGT+KMeans的示意图
- 画一个HGT+SOM的示意图
- 项目readme要不断更新完善


## Dataset
Our dataset contain two types of nodes: ['user','note'] and three types of edges {'follow_note','comment_note','share_note'}.

## Distributed training
On real production data, we run code with parameter-server architecture including parameter-server node, worker node and graphlearn node. graphlearn node is for building graph and sampling neighborhood nodes. We have 20 graphlearn nodes, so our data is partitioned into 20 parts at distributed setting.

After building the above distributed training enviroment, you can run our distributed version code with the following steps:
1. generate distributed fake data: 
```
python generate_dist_fake_data.py
```

2. train model with distributed trainer:
```
python dist_hgt_cluster.py
```

## Local training 
For demonstration, you can locally run demo code with the following steps:
1. generate local fake data:
 ```
 cd gen_fake_data
 python generate_local_fake_data.py
 ```
2. train model with local trainer: 
```
python local_hgt_cluster.py
```

## desensitization result
### u2u2i recall


# Requirements
```
python: 2.7
tensorflow: 1.12.1
graphlearn: 1.0.1
```