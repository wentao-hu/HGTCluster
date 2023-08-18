# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
#/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import json
import os
import sys

import numpy as np

try:
    # https://www.tensorflow.org/guide/migrate
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

import graphlearn as gl
import graphlearn.python.nn.tf as tfg


class HGTRgcnDataLoader:
    def __init__(self, graph, mask=gl.Mask.TRAIN, sampler='random',
                 batch_size=128, window=10, train_mode='all',
                 neg_num=5, neg_sampler="random", batch_relation=None,
                 u_nbrs_num=None, i_nbrs_num=None, relation_alias=None):
        self._neg_num = neg_num
        self._neg_sampler = neg_sampler
        self._u_nbrs_num = u_nbrs_num
        self._i_nbrs_num = i_nbrs_num
        self._relations_alias = relation_alias
        self._train_mode = train_mode

        self._sampler = sampler
        self._batch_size = batch_size
        self._q = self.query(graph, batch_relation)
        self._dataset = tfg.Dataset(self._q, window=window)
        self._iterator = self._dataset.iterator
        self._data_dict = self._dataset.get_data_dict()


    def get_egograph(self, key, neighbors=None):
        return self._dataset.get_egograph(key, neighbors)

    def get_relation_ego(self, centric_name, relation, ego_relation, node_type):
        src_name = centric_name + '_' + relation
        ego_name = centric_name + '_' + relation + '_' + self._relations_alias.get(ego_relation)
        hops = range(len(self._u_nbrs_num if node_type == "user" else self._i_nbrs_num))
        return self.get_egograph(key=src_name, neighbors=[ego_name + '_hop_' + str(i + 1) for i in hops])

    def query(self, graph, relation):
        # traverse graph to get positive and negative (u,i) samples.
        # query for specific edge_type
        batch_relation = relation
        if self._train_mode == 'incremental':
            batch_relation = 'label_' + batch_relation
            print("batch relation for " + batch_relation)
        edge = graph.E(relation).batch(self._batch_size).shuffle(traverse=True).alias('seed_' + relation)
        src = edge.outV().alias('src_' + relation)
        dst = edge.inV().alias('dst_' + relation)
        neg_dst = src.outNeg(relation).sample(self._neg_num).by(self._neg_sampler).alias('neg_dst_' + relation)
        neg_src = dst.outNeg(relation).sample(self._neg_num).by(self._neg_sampler).alias('neg_src_' + relation)
        # user=> user_follow_note_user
        # user=> user_share_note_user
        # user=> user_comment_note_user
        # note=> note_follow_user_note
        # note=> note_share_user_note
        # note=> note_comment_user_note

        for edge_type, alias in self._relations_alias.items():
            # sampling various alias neibors for specific relation
            # meta-path sampling.
            src_ego = self.meta_path_sample(src, 'user', ego_name='src_' + relation + '_' + alias, nbrs_num=self._u_nbrs_num, sampler=self._sampler, i2i=False, edge_type=edge_type)
            dst_ego = self.meta_path_sample(dst, 'note', 'dst_' + relation + '_' + alias, self._i_nbrs_num, self._sampler, False, edge_type)
            neg_dst_ego = self.meta_path_sample(neg_dst, 'note', 'neg_dst_' + relation + '_' + alias, self._i_nbrs_num, self._sampler, False,
                                           edge_type)
            neg_src_ego = self.meta_path_sample(neg_src, 'user', 'neg_src_' + relation + '_' + alias, self._u_nbrs_num,
                                                self._sampler, False,
                                                edge_type)
        return edge.values()

    def meta_path_sample(self, ego, ego_type, ego_name, nbrs_num, sampler, i2i, edge_type):
        """ creates the meta-math sampler of the input ego.
          ego: A query object, the input centric nodes/edges
          ego_type: A string, the type of `ego`, 'u' or 'i'.
          ego_name: A string, the name of `ego`.
          nbrs_num: A list, the number of neighbors for each hop.
          sampler: A string, the strategy of neighbor sampling.
          i2i: Boolean, is i2i egde exist or not.
        """
        choice = int(ego_type == 'note')
        meta_path = []
        hops = range(len(nbrs_num))
        if i2i:
            # for u is u-i-i-i..., for i is i-i-i-i...
            meta_path = ['outV' for i in hops]
        else:
            # for u is u-i-u-i-u..., for i is i-u-i-u-i....
            meta_path = [('outV', 'inV')[(i + choice) % 2] for i in hops]
        alias_list = [ego_name + '_hop_' + str(i + 1) for i in hops]
        idx = 0
        mata_path_string = ""
        for path, nbr_count, alias in zip(meta_path, nbrs_num, alias_list):
            etype = (edge_type, 'i-i')[(int(i2i) and choice) or (int(i2i) and not choice and idx > 0)]
            idx += 1
            mata_path_string += path + '(' + etype + ').'
            ego = getattr(ego, path)(etype).sample(nbr_count).by(sampler).alias(alias)
        # print("Sampling meta path for {} is {}.".format(ego_type, mata_path_string))
        return ego
