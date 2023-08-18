
# 起一个好听的项目名字
- 画一个HGT+KMeans的示意图
- 画一个HGT+SOM的示意图

参考 [alibaba/graph-learn](https://github.com/alibaba/graph-learn) 的readme文档进行包装项目, 要图文并茂 ，简洁明了，抓住眼球

- 介绍如何生成fake dataset， dataset的意思是什么
- 介绍下方法
- 项目readme要不断更新完善，加油
- 测试

## Dataset
Our dataset contain two types of nodes: ['user','note'] and three types of edges {'follow_note','comment_note','share_note'}.

# Single-Machine Demo Version
For demonstration, you can run our single-machine demo code with the following steps:
1. generate local fake data:
 ```
 cd gen_fake_data
 python generate_local_fake_data.py
 ```
2. train model with local trainer: 
```python local_hgt_cluster.py```


# Distributed Version
On real production data, we run our code with parameter-server architecture including parameter-server node, worker node and graphlearn node. graphlearn node is for building graph and sampling neighborhood nodes.

After building the above distributed training enviroment, you can run our distributed version code with the following steps:
1. generate distributed fake data: `python generate_dist_fake_data.py`
2. train model with distributed trainer:`python dist_hgt_cluster.py`


## desensitization result
### u2u2i recall


# Requirements
```
python: 2.7
tensorflow: 1.12.1
graphlearn: 1.0.1
```