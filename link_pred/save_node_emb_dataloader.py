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



class SaveNodeDataLoader:
  def __init__(self, graph, mask=gl.Mask.TRAIN, sampler='random',
               batch_size=128, window=10,                   # parameters for base EgoDataLoader
               src_node_type = None, pos_relation_dict=None, label_relation=None,
               user_feature_handler=None, author_feature_handler=None, note_feature_handler = None,
               nbrs_num=None):  
    
    # self defined parameters 
    self._src_node_type = src_node_type
    self._pos_relation_dict = pos_relation_dict 
    self._user_feature_handler = user_feature_handler
    self._author_feature_handler = author_feature_handler
    self._note_feature_handler = note_feature_handler
    self._nbrs_num = nbrs_num
    self._label_relation = label_relation
          
    # base initialization 
    self._graph = graph
    self._sampler = sampler
    self._batch_size = batch_size

  
    # we have 1 GSL entrance for saving node embedding, g.V(src_node_type)
    self._q, self._relation_path, self._node_type_path, self._relation_count, self._alias_list = self._query_by_relation(self._graph, self._label_relation, self._src_node_type)
    
    print('=for save emb: relation_path',self._relation_path)
    print('=for save emb: node_type_path',self._node_type_path)
    print('=for save emb: relation_count',self._relation_count)  
    print('=for save emb: alias_list',self._alias_list)
    
    self._dataset = tfg.Dataset(self._q, window=window)      
    self._iterator = self._dataset.iterator
    self._data_dict = self._dataset.get_data_dict()
    

  def x_list(self, query, data_dict, alias_list, node_type_path):
    """ Transforms and organizes the input data to a list of list,
    each element of list is also a list which consits of k-hop multi-relations
    neighbor nodes' feature tensor.
    """ 
    x_list = []
    for i in range(len(alias_list)): # len(alias_list)=1+hop_num
        neigh_hop_i_alias_list = alias_list[i]   # i=0 means self
        neigh_hop_i_node_type_list = node_type_path[i]
        nbr_list = []
        for j in range(len(neigh_hop_i_alias_list)):
            alias = neigh_hop_i_alias_list[j]              
            node_type = neigh_hop_i_node_type_list[j][-1]
            # print('=for save node embedding, get x_list for hop: ',i,'alias: ', alias, 'node type: ',node_type)
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
    return x_list

  def _query_by_relation(self, graph, label_relation, src_node_type):
    """
      src_node_type: the src node type for GSL entrance
      k-hop neighbor sampling using different relations.
      For train, the query node name are as follows:
      root: ['train']
      1-hop neighbors: ['train_hop_0_r_0', 'train_hop_0_r_1']
      2-hop neighbors: ['train_hop_0_r_0_hop_1_r_0', 'train_hop_0_r_0_hop_1_r_0',
                        'train_hop_0_r_1_hop_1_r_0', 'train_hop_0_r_1_hop_1_r_0']
      ...
    """
    # traverse src node for specific label_relation
    q = graph.V(label_relation, node_from=gl.EDGE_SRC).batch(self._batch_size).shuffle(traverse=True).alias(label_relation + "_" + src_node_type)

    current_hop_list = [q]
    current_relation_list = [[src_node_type]]
    current_node_type = [[src_node_type]]

    relation_path = [current_relation_list]
    node_type_path = [current_node_type]

    neigh_rel_count = []
    alias_path = [[q.get_alias()]]

    for idx, hop in enumerate(self._nbrs_num):
      next_hop_list = []
      next_hop_relation = []
      next_hop_node_type = []
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
          if self._pos_relation_dict[rel][0] == hop_q_node_type[-1]:
            alias = hop_q.get_alias() + '-hop' + str(idx) + '_out_' + rel
            next_hop_list.append(hop_q.outV(rel).sample(hop).by(self._sampler).alias(alias))
            next_hop_relation.append(hop_q_relation + [rel])
            next_hop_node_type.append(hop_q_node_type + [self._pos_relation_dict[rel][1]])
            hop_q_neigh_rel_count += 1
            next_hop_alias.append(alias)

          elif self._pos_relation_dict[rel][1] == hop_q_node_type[-1]:
            alias = hop_q.get_alias() + '-hop' + str(idx) + '_in_' + rel
            next_hop_list.append(hop_q.inV(rel).sample(hop).by(self._sampler).alias(alias))
            next_hop_relation.append(hop_q_relation + [rel])
            next_hop_node_type.append(hop_q_node_type + [self._pos_relation_dict[rel][0]])
            hop_q_neigh_rel_count += 1
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

    return q.values(), relation_path, node_type_path, neigh_rel_count, alias_path
 
   
  def _query(self, graph, src_node_type):
    """ 
      src_node_type: the src node type for GSL entrance 
      k-hop neighbor sampling using different relations.
      For train, the query node name are as follows:
      root: ['train']
      1-hop neighbors: ['train_hop_0_r_0', 'train_hop_0_r_1']
      2-hop neighbors: ['train_hop_0_r_0_hop_1_r_0', 'train_hop_0_r_0_hop_1_r_0', 
                        'train_hop_0_r_1_hop_1_r_0', 'train_hop_0_r_1_hop_1_r_0']
      ...
    """
    q = graph.V(src_node_type).batch(self._batch_size).shuffle(traverse=True).alias(src_node_type)  
    
    current_hop_list = [q]  
    current_relation_list = [[src_node_type]]
    current_node_type =[[src_node_type]]

    relation_path= [current_relation_list]
    node_type_path = [current_node_type]
      
    neigh_rel_count = []
    alias_path = [[q.get_alias()]]

    for idx, hop in enumerate(self._nbrs_num):
      next_hop_list = []
      next_hop_relation = []
      next_hop_node_type = []
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
      
    return q.values(), relation_path, node_type_path, neigh_rel_count, alias_path