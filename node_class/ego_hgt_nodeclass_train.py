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
from trainer import LocalTrainer

from ego_hgt_nodeclass import EgoHGT
from ego_hgt_data_loader import EgoHGTDataLoader

flags = tf.app.flags
FLAGS = flags.FLAGS
# user-defined params
flags.DEFINE_integer('epochs', 40, 'training epochs')
flags.DEFINE_integer('train_batch_size', 128, 'training minibatch size')
flags.DEFINE_integer('test_batch_size', 128, 'test minibatch size')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_float('drop_out', 0.5, 'drop out rate')
flags.DEFINE_integer('hidden_dim', 128, 'hidden layer dim')
flags.DEFINE_integer('class_num', 7, 'final output embedding dim')
flags.DEFINE_integer('n_heads', 8, 'number of relations')
flags.DEFINE_boolean('use_norm', True, 'use norm for hgt aggregation')
flags.DEFINE_string('nbrs_num', '[20,20]', 'string of list, neighbor num of each hop')
flags.DEFINE_string('sampler', 'random', 'neighbor sampler strategy. random, in_degree, topk.')
flags.DEFINE_string('attr_types', None, 'node attribute types')
flags.DEFINE_string('attr_dims', None, 'node attribute dimensions')
flags.DEFINE_integer('float_attr_num', 1433, 
  'number of float attrs. If there is only float attrs, we use this flag to instead of two above flags.')

model_params= {'epoch':FLAGS.epochs,'train_bs':FLAGS.train_batch_size, 'lr': FLAGS.learning_rate, 'dropout':FLAGS.drop_out,
               'hidden_dim':FLAGS.hidden_dim,'nbrs_num':FLAGS.nbrs_num}
print("===HGT model params===:", model_params )

if FLAGS.attr_types is not None and FLAGS.attr_dims is not None:
  attr_types = json.loads(FLAGS.attr_types)
  attr_dims = json.loads(FLAGS.attr_dims)
else:
  assert FLAGS.float_attr_num > 0
  attr_types = ['float'] * FLAGS.float_attr_num
  attr_dims = [0] * FLAGS.float_attr_num


def load_graph():
  """ Load node and edge data to build graph.
    Note that node_type must be "i", and edge_type must be "r_i", 
    the number of edge tables must be the same as FLAGS.num_relations.
  """
  cur_path = sys.path[0]
  dataset_folder = os.path.abspath(os.path.join(cur_path, ".."))+ '/data/cora/'
  print('dataset folder',dataset_folder)
  g = gl.Graph()\
        .node(dataset_folder + "node_table", node_type="i",
              decoder=gl.Decoder(labeled=True,
                                 attr_types=attr_types,
                                 attr_delimiter=":"))                      \
        .edge(dataset_folder + "edge_table",
              edge_type=("i", "i", "r_0"),
              decoder=gl.Decoder(weighted=True), directed=False)           \
        .edge(dataset_folder + "edge_table_with_self_loop",
              edge_type=("i", "i", "r_1"),
              decoder=gl.Decoder(weighted=True), directed=False)           \
        .node(dataset_folder + "train_table", node_type="i",
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.TRAIN)       \
        .node(dataset_folder + "test_table", node_type="i",
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.TEST)
  relation_dict = {'r_0':['i','i'], 'r_1':['i','i']}
  return g, relation_dict

def supervised_loss(logits, labels):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  return tf.reduce_mean(loss)

def accuracy(logits, labels):
  indices = tf.math.argmax(logits, 1, output_type=tf.int32)
  correct = tf.reduce_sum(tf.cast(tf.math.equal(indices, labels), tf.float32))
  return correct / tf.cast(tf.shape(labels)[0], tf.float32)


def main(unused_argv):
  print('=========start load graph=======')
  g, relation_dict = load_graph()
  num_relations = len(relation_dict)  

  g.init()

  
  # Define Model
  input_dim = sum([1 if not i else i for i in attr_dims])
  nbrs_num = json.loads(FLAGS.nbrs_num)  # '[20]', 'string of list, neighbor num of each hop'
  num_layers = len(nbrs_num)
  model = EgoHGT(input_dim,
                  FLAGS.hidden_dim,
                  FLAGS.class_num,
                  num_layers,  
                  relation_dict,
                  n_heads=FLAGS.n_heads,
                  dropout=FLAGS.drop_out,
                  use_norm=FLAGS.use_norm)
  
  # prepare train dataset
  train_data = EgoHGTDataLoader(g, gl.Mask.TRAIN, FLAGS.sampler, FLAGS.train_batch_size,
                                 node_type='i', nbrs_num=nbrs_num, num_relations=num_relations)

  
  # # print model structure
  # print('====start model structure======')
  # total_params = 0
  # for var in tf.trainable_variables():
  #   print(var)
  #   shape = var.shape # get shape of each param: 'tensorflow.python.framework.tensor_shape.TensorShape'
  #   array = np.asarray([dim.value for dim in shape]) 
  #   mulValue = np.prod(array) 
  #   total_params += mulValue
  # print('====finish model structure======')
  # print('===total params: {}'.format(total_params) )
  
  
  # model update
  train_embedding = model.forward(train_data.x_list(), train_data.relation, nbrs_num)
  loss = supervised_loss(train_embedding, train_data.labels)
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

  # prepare test dataset
  test_data = EgoHGTDataLoader(g, gl.Mask.TEST, FLAGS.sampler, FLAGS.test_batch_size,
                                 node_type='i', nbrs_num=nbrs_num, num_relations=num_relations)
  test_embedding = model.forward(test_data.x_list(), test_data.relation, nbrs_num)
  test_acc = accuracy(test_embedding, test_data.labels)

  
  trainer = LocalTrainer()
  # train with some epochs
  trainer.train(train_data.iterator, loss, optimizer, epochs=FLAGS.epochs) 
  
  # test
  trainer.test(test_data.iterator, test_acc)
  print("===HGT model params===:", model_params )


  # finish
  g.close()


if __name__ == "__main__":
  tf.app.run()
