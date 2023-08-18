from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import json
import os
import sys
import numpy as np
import tensorflow as tf

import graphlearn as gl
import graphlearn.python.nn.tf as tfg
import ego_data_loader as ego_data



class EgoHGTDataLoader:
  def __init__(self, graph, mask=gl.Mask.TRAIN, sampler='random',
               batch_size=128, window=10,                   # parameters for base EgoDataLoader
               pos_relation=None, neg_relation=None, pos_relation_dict=None, neg_relation_dict=None, 
               user_feature_handler=None, author_feature_handler=None, note_feature_handler = None,
               nbrs_num=None):  
    
    # self defined parameters 
    self._pos_relation = pos_relation
    self._neg_relation = neg_relation
    self._pos_relation_dict = pos_relation_dict 
    self._neg_relation_dict = neg_relation_dict 
    self._user_feature_handler = user_feature_handler
    self._author_feature_handler = author_feature_handler
    self._note_feature_handler = note_feature_handler
    self._nbrs_num = nbrs_num

              
    # base initialization 
    self._graph = graph
    self._sampler = sampler
    self._batch_size = batch_size
  
    # we have two GSL entrance here, g.E(pos_relation) and g.E(neg_relation)
    # we must return all sampled note in 1 query, 1 query for 1 dataset, and trainer will use the unique dataset and its query
    self._q_pos, self._pos_relation_path_dict, self._pos_node_type_path_dict, self._pos_relation_count_dict, self._pos_alias_dict = self._query(self._graph, self._pos_relation)
    self._q_neg, self._neg_relation_path_dict, self._neg_node_type_path_dict, self._neg_relation_count_dict, self._neg_alias_dict  = self._query(self._graph, self._neg_relation)
    
    print('= for link pred:pos_relation_path_dict',self._pos_relation_path_dict)
    print('= for link pred:pos_node_type_path_dict',self._pos_node_type_path_dict)
    print('= for link pred:pos_relation_count_dict',self._pos_relation_count_dict)  
    print('= for link pred:pos_alias_dict',self._pos_alias_dict)
   
    self._pos_dataset = tfg.Dataset(self._q_pos,window=window)      
    self._pos_iterator = self._pos_dataset.iterator
    self._pos_data_dict = self._pos_dataset.get_data_dict()

    self._neg_dataset = tfg.Dataset(self._q_neg, window=window)      
    self._neg_iterator = self._neg_dataset.iterator
    self._neg_data_dict = self._neg_dataset.get_data_dict()


  def x_list(self, query, data_dict, alias_dict, node_type_path_dict):
    """ Transforms and organizes the input data to a list of list,
    each element of list is also a list which consits of k-hop multi-relations
    neighbor nodes' feature tensor.
    """ 

    x_list_src_dst = []   # x_list_src_dst = [x_list_src, x_list_dst]
    for flag in ['src','dst']:
      alias_list = alias_dict[flag]
      node_type_path = node_type_path_dict[flag]
      
      x_list = []
      for i in range(len(alias_list)): # len(alias_list)=1+hop_num
          neigh_hop_i_alias_list = alias_list[i]   # i=0 means self
          neigh_hop_i_node_type_list = node_type_path[i]
          nbr_list = []
          for j in range(len(neigh_hop_i_alias_list)):
              alias = neigh_hop_i_alias_list[j]              
              node_type = neigh_hop_i_node_type_list[j][-1]
              # print('==for link prediction, get x_list for hop: ',i,'alias: ', alias, 'node type: ',node_type)
              if node_type == 'user':
                feature_handler = self._user_feature_handler
              elif node_type == 'author':
                feature_handler = self._author_feature_handler
              else:
                feature_handler = self._note_feature_handler
              # data = data_dict[alias]
              # print('data type',type(data))
              nbr_list.append(feature_handler.forward(data = data_dict[alias]))  
          x_list.append(nbr_list)
      x_list_src_dst.append(x_list) 
    return x_list_src_dst
 
   

  def _query(self, graph, relation):
    """ 
      relation: pos_relation or neg_relation, the relation for GSL entrance 
      k-hop neighbor sampling using different relations.
      For train, the query node name are as follows:
      root: ['train']
      1-hop neighbors: ['train_hop_0_r_0', 'train_hop_0_r_1']
      2-hop neighbors: ['train_hop_0_r_0_hop_1_r_0', 'train_hop_0_r_0_hop_1_r_0', 
                        'train_hop_0_r_1_hop_1_r_0', 'train_hop_0_r_1_hop_1_r_0']
      ...
    """
    
    # for relation in [pos_relation, neg_relation]
    q = graph.E(relation).batch(self._batch_size).shuffle(traverse=True).alias(relation)  # GSL entrance, relation is pos_relation or neg_relation
    
    if 'neg' in relation:
      src_node_type = self._neg_relation_dict[relation][0]
      dst_node_type = self._neg_relation_dict[relation][1]
    else:
      src_node_type = self._pos_relation_dict[relation][0]
      dst_node_type = self._pos_relation_dict[relation][1]
    src = q.outV().alias('src_node_'+src_node_type)
    dst = q.inV().alias('dst_node_'+dst_node_type)
    
    
    relation_path_dict = {}
    node_type_path_dict = {}
    relation_count_dict = {}
    alias_dict = {}
    for node in [src,dst]:
      if node==src:
        node_type = src_node_type
      else:
        node_type = dst_node_type
      
      current_hop_list = [node]  # source node for sampling neighbors
      current_relation_list = [[node_type]]
      current_node_type =[[node_type]]

      relation_path= [current_relation_list]
      node_type_path = [current_node_type]
       
      neigh_rel_count = []
      alias_path = [[node.get_alias()]]

      for idx, hop in enumerate(self._nbrs_num):
        next_hop_list = []
        next_hop_relation = []
        next_hop_node_type = []
        # print('idx',idx,'hop',hop)
        next_hop_neigh_rel_count = []
        next_hop_alias = []
        for j in range(len(current_hop_list)):
            hop_q = current_hop_list[j]
            hop_q_relation = current_relation_list[j]
            hop_q_node_type = current_node_type[j]
            # print('j',j,'hop_q alias',hop_q.get_alias(),'hop_q_relation',hop_q_relation,'hop_q_node_type',hop_q_node_type)
            
            hop_q_neigh_rel_count = 0
            for rel in self._pos_relation_dict.keys():
              # if the src and dst of rel are the same node type, we prioritize the outV, regard current node as src
              if self._pos_relation_dict[rel][0]==hop_q_node_type[-1]:  
                alias = hop_q.get_alias() + '-hop' + str(idx) + '_out_' + rel
                next_hop_list.append(hop_q.outV(rel).sample(hop).by(self._sampler).alias(alias))
                next_hop_relation.append(hop_q_relation + [rel])
                next_hop_node_type.append(hop_q_node_type + [self._pos_relation_dict[rel][1]]) 
                hop_q_neigh_rel_count +=1
                next_hop_alias.append(alias)
              
              elif self._pos_relation_dict[rel][1]== hop_q_node_type[-1]:
                alias = hop_q.get_alias() + '-hop' + str(idx) + '_in_' + rel
                next_hop_list.append(hop_q.inV(rel).sample(hop).by(self._sampler).alias(alias))
                next_hop_relation.append(hop_q_relation + [rel])
                next_hop_node_type.append(hop_q_node_type + [self._pos_relation_dict[rel][0]])
                hop_q_neigh_rel_count +=1
                next_hop_alias.append(alias)
              else:
                continue  
            next_hop_neigh_rel_count.append(hop_q_neigh_rel_count)  # for getting relaion count

        
        current_hop_list = next_hop_list
        current_relation_list = next_hop_relation
        current_node_type = next_hop_node_type
        relation_path.append(current_relation_list)
        node_type_path.append(current_node_type)
        neigh_rel_count.append(next_hop_neigh_rel_count)
        alias_path.append(next_hop_alias)
      
      if node==src:
        relation_path_dict['src'] = relation_path
        node_type_path_dict['src'] = node_type_path
        relation_count_dict['src'] = neigh_rel_count
        alias_dict['src'] = alias_path
      else:
        relation_path_dict['dst'] = relation_path
        node_type_path_dict['dst'] = node_type_path
        relation_count_dict['dst'] = neigh_rel_count
        alias_dict['dst'] = alias_path

    return q.values(), relation_path_dict, node_type_path_dict, relation_count_dict, alias_dict

