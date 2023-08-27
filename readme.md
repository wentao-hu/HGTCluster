
# HGTCluster
In this demo project, we show how to locally train original [Heterogeneous Graph Transformer](https://arxiv.org/pdf/2003.01332.pdf)(HGT) under self-supervised setting. We do not show distributed training code in production setting and our improvements based on HGT for interest of company. 

We use [alibaba/graph-learn](https://github.com/alibaba/graph-learn) and distributed tensorflow to train graph neural networks on large-scale graphs with billion scale nodes and edges in production settting.

### Dataset
Our generated fake dataset contain two types of nodes: ['user','note'] and two types of edges ['follow_note','share_note']. It can be easily extended to more node types and edge types.


### Local training 
For demonstration, you can locally run our demo code with the following steps:
1. generate fake data for local setting:
```
cd gen_fake_data/
python gen_fake_node_local.py
python gen_fake_edge_local.py
 ```
2. train model with local trainer: 
```
cd hgt_model/
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
