#coding:utf-8
import numpy as np
import time
import datetime

import graphlearn as gl
import tensorflow as tf
import graphlearn.python.nn.tf as tfg
import sys
import os
import shutil
import commands
from trainer import DistTrainer

#np.set_printoptions(threshold=sys.maxsize)
class EgoLightGCN(tfg.Module):
  """ a variation of LightGCN. https://arxiv.org/abs/2002.02126.

  Args:
    dim: int, emb dim.
    bn_fn: Batch normalization function for hidden layers' output. Default is
      None, which means batch normalization will not be performed.
    active_fn: Activation function for hidden layers' output. Default is None,
      which means activation will not be performed.
    dropout: Dropout rate for hidden layers' output. Default is None, which
      means dropout will not be performed. The optional value is a float.
  """

  def __init__(self,
               name,
               in_dim,
               out_dim,
               in_seq,
               agg_type="mean",
               com_type="add",
               layer_num=2,
               bn_fn=None,
               active_fn=None,
               dropout=None,
               **kwargs):
    super(EgoLightGCN, self).__init__()
    assert len(in_seq) == (layer_num + 1)
    self.in_seq = in_seq
    
    self.name = name
    self.out_dim = out_dim
    layers = []
    for i in range(layer_num):
      lg = []
      for j in range(layer_num - i):
        layer = tfg.EgoSAGELayer("{}_EgoSAGELayer_{}_{}".format(name, i, j),
                           input_dim=out_dim,
                           output_dim=out_dim,
                           agg_type=agg_type,
                           com_type=com_type,
                           parameter_share=False)
        lg.append(layer)
      group = tfg.EgoSAGELayerGroup(lg)
      layers.append(group)
    self.layers = layers
    self.pre_linears = []
    self.pre_linears.append(
      tfg.LinearLayer("{}_pre_linear_0".format(self.name), in_dim[0], self.out_dim, True))
    self.pre_linears.append(
      tfg.LinearLayer("{}_pre_linear_1".format(self.name), in_dim[1], self.out_dim, True))
    self.bn_func = bn_fn
    self.active_func = active_fn

    if dropout is not None:
      self.dropout_func = tf.nn.dropout(keep_prob=1-dropout)
    else:
      self.dropout_func = None

  def forward(self, graph):
    """
    Args:
      graph: EgoGraph object. An EgoGraph contains a batch of nodes and their
        n-hop neighbors. In EgoGraph, wo do not care where the neighbors come
        from, a homogeneous graph, or a heterogeneous one.

    Return:
      A tensor with shape [batch_size, out_dim]
    """
    graph = graph.forward()

    # h^{0}
    h = [graph.nodes]
    for i in range(len(self.layers)):
      h.append(graph.hop(i))
    i = 0
    H = []
    for x in h:
      x = self.pre_linears[self.in_seq[i]].forward(x)
      H.append(x)
      i += 1
    h = H

    hops = graph.expands
    h_all_layer = [h[0]]
    for i in range(len(self.layers) - 1):
      # h^{i}
      current_hops = hops if i == 0 else hops[:-i]
      h = self.layers[i].forward(h, current_hops)
      H = []
      for x in h:
        if self.bn_func:
          x = self.bn_func(x)
        if self.active_func:
          x = self.active_func(x)
        if self.dropout_func and tfg.conf.training:
          x = self.dropout_func(x)
        H.append(x)
      h_all_layer.append(H[0])
      h = H

    # The last layer
    h = self.layers[-1].forward(h, [hops[0]])
    assert len(h) == 1
    h_all_layer.append(h[0])
    h_final = tf.reduce_mean(tf.stack(h_all_layer, axis=0), axis=0, keepdims=False)
    return h_final

def build_loss(src, dst, neg):
  x1 = COSINE_SCALE * tf.nn.l2_normalize(src, axis=-1) * tf.nn.l2_normalize(dst, axis=-1)
  true_logits = tf.reduce_sum(x1, axis=-1)
  
  dim = src.shape[1]
  neg_expand = neg.shape[1]
  src = tf.tile(tf.expand_dims(src, axis=1), [1, neg_expand, 1])
  src = tf.reshape(src, [-1, dim])
  neg = tf.reshape(neg, [-1, dim])
  x2 = COSINE_SCALE * tf.nn.l2_normalize(src, axis=-1) * tf.nn.l2_normalize(neg, axis=-1)
  neg_logits = tf.reduce_sum(x2, axis=-1)

  true_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(true_logits),
        logits=true_logits)
  neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(neg_logits),
        logits=neg_logits)
  loss = tf.reduce_mean(true_loss) + tf.reduce_mean(neg_loss)
  return loss

def build_hinge_loss(src, dst, neg):
  x1 = tf.nn.l2_normalize(src, axis=-1) * tf.nn.l2_normalize(dst, axis=-1)
  true_logits = tf.reduce_sum(x1, axis=-1)
  
  dim = src.shape[1]
  neg_expand = neg.shape[1]
  src = tf.tile(tf.expand_dims(src, axis=1), [1, neg_expand, 1])
  src = tf.reshape(src, [-1, dim])
  neg = tf.reshape(neg, [-1, dim])
  x2 = tf.nn.l2_normalize(src, axis=-1) * tf.nn.l2_normalize(neg, axis=-1)
  neg_logits = tf.reduce_sum(x2, axis=-1)

  true_logits = tf.tile(true_logits, [neg_expand])
  loss = tf.reduce_mean(tf.nn.relu(- true_logits + neg_logits + MARGIN))
  return loss

def build_bpr_loss(src, dst, neg):
  x1 = tf.nn.l2_normalize(src, axis=-1) * tf.nn.l2_normalize(dst, axis=-1)
  true_logits = tf.reduce_sum(x1, axis=-1)
  
  dim = src.shape[1]
  neg_expand = neg.shape[1]
  src = tf.tile(tf.expand_dims(src, axis=1), [1, neg_expand, 1])
  src = tf.reshape(src, [-1, dim])
  neg = tf.reshape(neg, [-1, dim])
  x2 = tf.nn.l2_normalize(src, axis=-1) * tf.nn.l2_normalize(neg, axis=-1)
  neg_logits = tf.reduce_sum(x2, axis=-1)

  true_logits = tf.tile(true_logits, [neg_expand])
  loss = tf.reduce_mean(-tf.log_sigmoid(COSINE_SCALE * (true_logits-neg_logits)))
  return loss

def build_bpr_loss_adaptive_margin(src, dst, neg):
  x1 = tf.nn.l2_normalize(src, axis=-1) * tf.nn.l2_normalize(dst, axis=-1)
  true_logits = tf.reduce_sum(x1, axis=-1)
  
  dim = src.shape[1]
  neg_expand = neg.shape[1]
  src = tf.tile(tf.expand_dims(src, axis=1), [1, neg_expand, 1])
  src = tf.reshape(src, [-1, dim])
  neg = tf.reshape(neg, [-1, dim])
  x2 = tf.nn.l2_normalize(src, axis=-1) * tf.nn.l2_normalize(neg, axis=-1)
  neg_logits = tf.reduce_sum(x2, axis=-1)

  dst_1 = tf.tile(tf.expand_dims(dst, axis=1), [1, neg_expand, 1])
  dst_1 = tf.reshape(dst_1, [-1, dim])
  x3 = tf.nn.l2_normalize(dst_1, axis=-1) * tf.nn.l2_normalize(neg, axis=-1)
  adaptive_margin = tf.maximum(0.0, tf.reduce_sum(x3, axis=-1))

  true_logits = tf.reshape(tf.tile(tf.expand_dims(true_logits, axis=1), [1, neg_expand]), [-1])
  #weights = tf.reshape(tf.tile(tf.expand_dims(tf.reshape(weights, [-1]), axis=1), [1, neg_expand]), [-1])
  loss = tf.reduce_mean(-tf.log_sigmoid(COSINE_SCALE * (true_logits - neg_logits - adaptive_margin)))
  return loss

# build_model never used, should comment it
def build_model(name, layer_num, u_dim, i_dim, dims):
  layer_groups = []
  for i in range(layer_num):
    layer_group = []
    for j in range(layer_num - i):
      if j % 2 == 0:
        layer = tfg.EgoSAGELayer(name + "_EgoSAGELayer_" + str(i) + "_" + str(j), 
                             input_dim=(u_dim, i_dim) if i == 0 else dims[i],
                             output_dim=dims[i + 1],
                             agg_type="mean",
                             com_type="concat",
                             parameter_share=False)
      else:
        layer = tfg.EgoSAGELayer(name + "_EgoSAGELayer_" + str(i) + "_" + str(j), 
                             input_dim=(i_dim, u_dim) if i == 0 else dims[i],
                             output_dim=dims[i + 1],
                             agg_type="mean",
                             com_type="concat",
                             parameter_share=False)
      layer_group.append(layer)
    group = tfg.EgoSAGELayerGroup(layer_group)
    layer_groups.append(group)
  return tfg.EgoGraphSAGE(layer_groups, bn_fn=None, active_fn=tf.nn.relu, droput=DROPOUT)

def build_lightgcn(name, layer_num, u_dim, i_dim, in_seq, dim):
  return EgoLightGCN(name, (u_dim, i_dim), dim, in_seq, agg_type="mean", com_type="concat", layer_num=layer_num, \
    bn_fn=None, active_fn=None, dropout=None)

def init_graph(TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST):
  t_attr_types = ['string'] * TOPIC_FEA_NUM
  t_attr_dims = [DISCRETE_EMB_SIZE] * TOPIC_FEA_NUM
  t_decoder = gl.Decoder(labeled=True, attr_types=t_attr_types, attr_dims=t_attr_dims, attr_delimiter=":")

  # n_attr_types = ['string'] * (ITEM_FEA_NUM - 1) + [('string', None, True)]
  n_attr_types = ['string'] * ITEM_FEA_NUM
  n_attr_dims = [DISCRETE_EMB_SIZE] * ITEM_FEA_NUM
  n_decoder = gl.Decoder(labeled=True, attr_types=n_attr_types, attr_dims=n_attr_dims, attr_delimiter=":")

  print("debug data dir: {}, {}".format(TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST))

  g = gl.Graph() \
    .node(TRAINING_DATA_NODE_LIST[0], node_type='t',
          decoder=t_decoder) \
    .node(TRAINING_DATA_NODE_LIST[1], node_type='n',
          decoder=n_decoder) \
    .edge(TRAINING_DATA_EDGE_LIST[0], edge_type=('n', 'n', 'n-n'),
          decoder=gl.Decoder(weighted=True), directed=False) \
    .edge(TRAINING_DATA_EDGE_LIST[1], edge_type=('t', 'n', 't-n'),
          decoder=gl.Decoder(weighted=True), directed=True) \
    .edge(TRAINING_DATA_EDGE_LIST[2], edge_type=('n', 't', 'n-t'),
          decoder=gl.Decoder(weighted=True), directed=True) \
    .edge(TRAINING_DATA_EDGE_LIST[1], edge_type=('t', 'n', 't-n-train'),
          decoder=gl.Decoder(weighted=True), directed=True) \
    .edge(TRAINING_DATA_EDGE_LIST[3], edge_type=('t', 't', 't-t'),
          decoder=gl.Decoder(weighted=True), directed=False)
  return g


def build_query(g):
  query = g.E('t-n-train').shuffle(traverse=True).batch(BATCH_SIZE).alias('seed').each( \
          lambda e: (
            e.inV().alias('n').each(
              lambda v: (
                v.outV('n-n').sample(HOP_1_N).by('random').alias('n_n').each(
                  lambda v1: (
                    v1.outV('n-n').sample(HOP_2_N).by('random').alias('n_n_n'),
                    v1.outV('n-t').sample(HOP_N_T).by('edge_weight').alias('n_n_t')
                  )
                ),
                v.outV('n-t').sample(HOP_N_T).by('edge_weight').alias('n_t')
                  .outV('t-n').sample(HOP_1_T).by('edge_weight').alias('n_t_n')
              )
            ),
            e.outV().alias('t').each(
              lambda v: (
                v.outV('t-n').sample(HOP_1_T).by('edge_weight').alias('t_n').each(
                  lambda v1: (
                    v1.outV('n-n').sample(HOP_2_T).by('random').alias('t_n_n'),
                    v1.outV('n-t').sample(HOP_N_T).by('edge_weight').alias('t_n_t')
                  )
                ),
                v.outV('t-t').sample(HOP_1_T).by('edge_weight').alias('t_t').each(
                  lambda v1: (
                    v1.outV('t-t').sample(HOP_1_T).by('edge_weight').alias('t_t_t'),
                  )
                ),
                v.outV('t-t').sample(1).by('edge_weight').alias('tpos').each(
                  lambda v1: (
                    v1.outV('t-t').sample(HOP_1_T).by('edge_weight').alias('tpos_t').each(
                      lambda v3: (
                        v3.outV('t-t').sample(HOP_1_T).by('edge_weight').alias('tpos_t_t'),
                      )
                    ),
                    v1.outV('t-n').sample(HOP_1_T).by('edge_weight').alias('tpos_n').each(
                      lambda v3: (
                        v3.outV('n-n').sample(HOP_2_T).by('random').alias('tpos_n_n'),
                        v3.outV('n-t').sample(HOP_N_T).by('edge_weight').alias('tpos_n_t')
                      )
                    )
                  )
                ),
                v.outNeg('t-t').sample(NEG).by('in_degree').alias('tneg').each(
                  lambda v1: (
                    v1.outV('t-t').sample(HOP_1_T).by('edge_weight').alias('tneg_t').each(
                      lambda v3: (
                        v3.outV('t-t').sample(HOP_1_T).by('edge_weight').alias('tneg_t_t'),
                      )
                    ),
                    v1.outV('t-n').sample(HOP_1_T).by('edge_weight').alias('tneg_n').each(
                      lambda v3: (
                        v3.outV('n-n').sample(HOP_2_T).by('random').alias('tneg_n_n'),
                        v3.outV('n-t').sample(HOP_N_T).by('edge_weight').alias('tneg_n_t')
                      )
                    ),
                    v1.outV('t-n').sample(1).by('random').alias('nneg').each(
                      lambda v3: (
                        v3.outV('n-n').sample(HOP_1_N).by('random').alias('nneg_n').each(
                          lambda v2: (
                            v2.outV('n-n').sample(HOP_2_N).by('random').alias('nneg_n_n'),
                            v2.outV('n-t').sample(HOP_N_T).by('edge_weight').alias('nneg_n_t')
                          )
                        ),
                        v3.outV('n-t').sample(HOP_N_T).by('edge_weight').alias('nneg_t')
                          .outV('t-n').sample(HOP_1_T).by('edge_weight').alias('nneg_t_n')
                      )
                    )
                  )
                )
                # ,
                # v.outNeg('t-t').sample(NEG).by('in_degree').alias('tneg').each(
                #   lambda v1: (
                #     v1.outV('t-n').sample(HOP_1_T).by('edge_weight').alias('tneg_n').each(
                #       lambda v2: (
                #         v2.outV('n-n').sample(HOP_2_T).by('random').alias('tneg_n_n'),
                #         v2.inV('t-n').sample(HOP_N_T).by('random').alias('tneg_n_t')
                #       )
                #     )
                #   )
                # )
              )
            )
          )
  ).values()
  return query

def build_query_save_topic(g):
  query = g.V('t-n', node_from=gl.EDGE_SRC).batch(BATCH_SIZE).shuffle(traverse=True).alias('t').each(
                lambda v: (
                  v.outV('t-n').sample(HOP_1_T).by('edge_weight').alias('t_n').each(
                    lambda v1:(
                      v1.outV('n-n').sample(HOP_2_T).by('random').alias('t_n_n'),
                      v1.outV('n-t').sample(HOP_N_T).by('edge_weight').alias('t_n_t')
                    )
                  ),
                  v.outV('t-t').sample(HOP_1_T).by('edge_weight').alias('t_t').each(
                    lambda v1: (
                      v1.outV('t-t').sample(HOP_1_T).by('edge_weight').alias('t_t_t'),
                    )
                  )
                )
              ).values()
  return query
  # query = g.V('t').batch(BATCH_SIZE).shuffle(traverse=True).alias('t') \
  #          .outV('t-n').sample(HOP_1_T).by('random').alias('t_n') \
  #          .outV('n-n').sample(HOP_2_T).by('random').alias('t_n_n') \
  #          .values()
  # return query

def build_query_save_note(g):
  query = g.V('n-t', node_from=gl.EDGE_SRC).batch(BATCH_SIZE).shuffle(traverse=True).alias('n').each(
                lambda v: (
                  v.outV('n-n').sample(HOP_1_N).by('random').alias('n_n').each(
                    lambda v1:(
                      v1.outV('n-n').sample(HOP_2_N).by('random').alias('n_n_n'),
                      v1.outV('n-t').sample(HOP_N_T).by('edge_weight').alias('n_n_t')
                    )
                  ),
                  v.outV('n-t').sample(HOP_N_T).by('edge_weight').alias('n_t')
                  .outV('t-n').sample(HOP_1_T).by('edge_weight').alias('n_t_n')
                )
              ).values()
  return query
  # # query = g.V('n').batch(BATCH_SIZE).shuffle(traverse=True).alias('n') \
  # #          .outV('n-n').sample(HOP_1_N).by('random').alias('n_n') \
  # #          .outV('n-n').sample(HOP_2_N).by('random').alias('n_n_n') \
  # #          .values()
  # return query


# def build_t_eg(df):
#   t_n_n_eg = df.get_ego_graph('t', neighbors=['t_n', 't_n_n'])
#   return (t_n_n_eg,)

def build_t_eg(df, src_name):
  t_t_t_eg = df.get_ego_graph(src_name, neighbors=['{}_t'.format(src_name), '{}_t_t'.format(src_name)])
  t_n_n_eg = df.get_ego_graph(src_name, neighbors=['{}_n'.format(src_name), '{}_n_n'.format(src_name)])
  t_n_t_eg = df.get_ego_graph(src_name, neighbors=['{}_n'.format(src_name), '{}_n_t'.format(src_name)])
  return (t_t_t_eg, t_n_n_eg, t_n_t_eg)

def build_n_eg(df, src_name):
  n_n_n_eg = df.get_ego_graph(src_name, neighbors=['{}_n'.format(src_name), '{}_n_n'.format(src_name)])
  n_n_t_eg = df.get_ego_graph(src_name, neighbors=['{}_n'.format(src_name), '{}_n_t'.format(src_name)])
  n_t_n_eg = df.get_ego_graph(src_name, neighbors=['{}_t'.format(src_name), '{}_t_n'.format(src_name)])
  return (n_n_n_eg, n_n_t_eg, n_t_n_eg)
  
def forward_group(model_group, eg_group):
  res = 0
  for model, eg in zip(model_group, eg_group):
    res += model.forward(eg)
  res /= len(model_group)
  return res

def forward_get_emb(eg_group):
  res = 0
  for eg in eg_group:
    graph = eg.forward()
    h = [graph.nodes]
    res += tf.reduce_mean(tf.stack(h, axis=0), axis=0, keepdims=False)
  res /= len(eg_group)
  return res
  
    
def train():
  gl_cluster, tf_cluster, job_name, task_index = gl.get_cluster()
  tf_cluster = tf.train.ClusterSpec(tf_cluster)
  gl.set_default_string_attribute("None")
  gl.set_padding_mode(gl.CIRCULAR)
  print("gl version:{}".format(gl.__git_version__))

  # training data
  SYNC_DONE = os.path.join(FLAGS.save_model_dir, "_SYNC_DONE")
  TRAIN_DONE = os.path.join(FLAGS.train_data_base_dir, "_TRAIN_DONE")
  EMB_DONE = os.path.join(FLAGS.save_emb_dir, "_DONE")
  TRAINING_DATA_NODE_LIST = []
  TRAINING_DATA_EDGE_LIST = []

  if job_name == "graphlearn":
    # training data
    TRAINING_DATA_NODE_LIST = [os.path.join(FLAGS.train_data_base_dir, node, "{}.csv".format(task_index)) for node in FLAGS.train_data_node_dir.split(',')]
    TRAINING_DATA_EDGE_LIST = [os.path.join(FLAGS.train_data_base_dir, edge, "{}.csv".format(task_index)) for edge in FLAGS.train_data_edge_dir.split(',')]

    if task_index == 0:
      ######################################### to rm
      # training data done
      data_done_cnt = 0
      for i in range(200):
        data_done_cnt = 0
        for node in FLAGS.train_data_node_dir.split(',') + FLAGS.train_data_edge_dir.split(','):
          cmd_status, data_done = commands.getstatusoutput("ls {} | grep _DONE | wc -l".format(os.path.join(FLAGS.train_data_base_dir, node)))
          if cmd_status == 0:
            try:
              data_done_cnt += int(data_done)
            except Exception:
              break
        
        if data_done_cnt == len(TRAINING_DATA_NODE_LIST) + len(TRAINING_DATA_EDGE_LIST):
          break
        else:
          print("check training data done: {}, cnt: {}".format("ls {} | grep _DONE | wc -l".format(FLAGS.train_data_base_dir), data_done))
          time.sleep(60)
      if data_done_cnt < len(TRAINING_DATA_NODE_LIST) + len(TRAINING_DATA_EDGE_LIST):
        raise RuntimeError("check training data not done: {}".format(data_done_cnt))
      #########################################

      print("check path:")
      for training_data in TRAINING_DATA_NODE_LIST + TRAINING_DATA_EDGE_LIST:
        if not os.path.exists(training_data):
          print("training data not exists: {}".format(training_data))
          raise RuntimeError("training data not exists: {}".format(training_data))
        else:
          print("training data dir: {}".format(training_data))
      print("load_model_dir: {}".format(FLAGS.load_model_dir))
      print("save_model_dir: {}".format(FLAGS.save_model_dir))

      print("syncing ckpt model dir ...")
      """
      save_dir as ckpt_dir
      1. filter and cp load_dir save_dir
      2. rewrite save_dir checkpoint
      """
      if os.path.exists(FLAGS.save_model_dir):
        shutil.rmtree(FLAGS.save_model_dir)

      if FLAGS.load_model_dir == "":
        mkdir_status, mkdir_output = commands.getstatusoutput("mkdir -p {}".format(FLAGS.save_model_dir))
      else:
        _, ckpt_version = commands.getstatusoutput("cat {} | head -1 | awk -F'-' '{{print $NF}}' | sed 's/\"//g'".format(os.path.join(FLAGS.load_model_dir, "checkpoint")))
        print("ckpt_version: {}".format(ckpt_version))
        mkdir_status, mkdir_output = commands.getstatusoutput("mkdir -p {}".format(FLAGS.save_model_dir))
        shutil.copy(os.path.join(FLAGS.load_model_dir, "checkpoint"), os.path.join(FLAGS.save_model_dir, "checkpoint"))
      
      if os.path.exists(FLAGS.save_emb_dir):
        shutil.rmtree(FLAGS.save_emb_dir)
      mkdir_status, mkdir_output = commands.getstatusoutput("mkdir -p {}".format(FLAGS.save_emb_dir))

      os.system('touch {}'.format(SYNC_DONE))
    else:
      for i in range(200):
        if os.path.isfile(SYNC_DONE):
          break
        else:
          print("waiting ckpt model dir ... {}".format(SYNC_DONE))
          time.sleep(60)


    g = init_graph(TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST)
    g.init(cluster=gl_cluster, job_name="server", task_index=task_index)
    print("GraphLearn Server start...")
    g.wait_for_close()
    
    if task_index == 0:
      for i in range(200):
        _, emb_num = commands.getstatusoutput("ls {} | grep part | wc -l".format(FLAGS.save_emb_dir))
        if int(emb_num) == FLAGS.save_emb_num:
          os.system('touch {}'.format(TRAIN_DONE))
          os.system('touch {}'.format(EMB_DONE))
        else:
          print("check commands:{}. now save {}/{}, waiting all workers to save embedding..." \
            .format("ls {} | grep part | wc -l".format(FLAGS.save_emb_dir), int(emb_num), FLAGS.save_emb_num))
          time.sleep(5)

  elif job_name == "ps":
    # training data
    TRAINING_DATA_NODE_LIST = [os.path.join(FLAGS.train_data_base_dir, node, "{}.csv".format(task_index)) for node in FLAGS.train_data_node_dir.split(',')]
    TRAINING_DATA_EDGE_LIST = [os.path.join(FLAGS.train_data_base_dir, edge, "{}.csv".format(task_index)) for edge in FLAGS.train_data_edge_dir.split(',')]
    
    # ######################################### to rm
    # # training data done
    # for i in range(2000):
    #   data_done_cnt = 0
    #   for node in FLAGS.train_data_node_dir.split(',') + FLAGS.train_data_edge_dir.split(','):
    #     cmd_status, data_done = commands.getstatusoutput("ls {} | grep _DONE | wc -l".format(os.path.join(FLAGS.train_data_base_dir, node)))
    #     if cmd_status == 0:
    #       data_done_cnt += int(data_done)
      
    #   if data_done_cnt == len(TRAINING_DATA_NODE_LIST) + len(TRAINING_DATA_EDGE_LIST):
    #     break
    #   else:
    #     print("check training data done: {}, cnt: {}".format("ls {} | grep _DONE | wc -l".format(FLAGS.train_data_base_dir), data_done))
    #     time.sleep(60)
    # #########################################


    for i in range(200):
      if os.path.isfile(SYNC_DONE):
        break
      else:
        print("waiting ckpt model dir ... {}".format(SYNC_DONE))
        time.sleep(60)
    trainer = DistTrainer(tf_cluster, job_name, task_index, tf_cluster.num_tasks("worker"),
                          ckpt_dir=FLAGS.save_model_dir,
              profiling=False)
    print("TF PS start...")
    trainer.join()
  else:
    # training data
    TRAINING_DATA_NODE_LIST = [os.path.join(FLAGS.train_data_base_dir, node, "{}.csv".format(task_index)) for node in FLAGS.train_data_node_dir.split(',')]
    TRAINING_DATA_EDGE_LIST = [os.path.join(FLAGS.train_data_base_dir, edge, "{}.csv".format(task_index)) for edge in FLAGS.train_data_edge_dir.split(',')]
    
    # ######################################### to rm
    # # training data done
    # for i in range(2000):
    #   data_done_cnt = 0
    #   for node in FLAGS.train_data_node_dir.split(',') + FLAGS.train_data_edge_dir.split(','):
    #     cmd_status, data_done = commands.getstatusoutput("ls {} | grep _DONE | wc -l".format(os.path.join(FLAGS.train_data_base_dir, node)))
    #     if cmd_status == 0:
    #       data_done_cnt += int(data_done)
      
    #   if data_done_cnt == len(TRAINING_DATA_NODE_LIST) + len(TRAINING_DATA_EDGE_LIST):
    #     break
    #   else:
    #     print("check training data done: {}, cnt: {}".format("ls {} | grep _DONE | wc -l".format(FLAGS.train_data_base_dir), data_done))
    #     time.sleep(60)
    # #########################################

    for i in range(200):
      if os.path.isfile(SYNC_DONE):
        break
      else:
        print("waiting ckpt model dir ... {}".format(SYNC_DONE))
        time.sleep(60)

    g = init_graph(TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST)
    g.init(cluster=gl_cluster, job_name="client", task_index=task_index)

    query = build_query(g)
    query_save_topic = build_query_save_topic(g)
    query_save_note = build_query_save_note(g)

    trainer = DistTrainer(tf_cluster, job_name, task_index, tf_cluster.num_tasks("worker"), ckpt_dir=FLAGS.save_model_dir,
              profiling=False)
    trainer.ckpt_freq = FLAGS.save_checkpoints_cycle
    with trainer.context():
      df = tfg.DataFlow(query)   # get 3 DataFlow
      df_save_topic = tfg.DataFlow(query_save_topic)
      df_save_note = tfg.DataFlow(query_save_note)
      # trainer.monitor(df)
      t_dim = DISCRETE_EMB_SIZE * TOPIC_FEA_NUM
      n_dim = DISCRETE_EMB_SIZE * ITEM_FEA_NUM
      layer_num = LAYER_NUM
      # many models,model_t_n_n = build_lightgcn("t_n_n", layer_num, t_dim, n_dim, [0, 1, 1], DIM)
      model_t_t_t = build_lightgcn("t_t_t", layer_num, t_dim, n_dim, [0, 0, 0], DIM)
      model_t_n_n = build_lightgcn("t_n_n", layer_num, t_dim, n_dim, [0, 1, 1], DIM)
      model_t_n_t = build_lightgcn("t_n_t", layer_num, t_dim, n_dim, [0, 1, 0], DIM)
      model_n_n_n = build_lightgcn("n_n_n", layer_num, t_dim, n_dim, [1, 1, 1], DIM)
      model_n_n_t = build_lightgcn("n_n_t", layer_num, t_dim, n_dim, [1, 1, 0], DIM)
      model_n_t_n = build_lightgcn("n_t_n", layer_num, t_dim, n_dim, [1, 0, 1], DIM)

      def build_acc(src, dst, neg):
        x1 = tf.nn.l2_normalize(src, axis=-1) * tf.nn.l2_normalize(dst, axis=-1)
        true_logits = tf.reduce_sum(x1, axis=-1, keepdims=True)
        dim = src.shape[1]
        neg_expand = neg.shape[1]
        src = tf.tile(tf.expand_dims(src, axis=1), [1, neg_expand, 1])
        src = tf.reshape(src, [-1, dim])
        neg = tf.reshape(neg, [-1, dim])
        x2 = tf.nn.l2_normalize(src, axis=-1) * tf.nn.l2_normalize(neg, axis=-1)
        #x2 = src * neg
        neg_logits = tf.reduce_sum(x2, axis=-1)
        neg_logits = tf.reshape(neg_logits, [-1, NEG])
        all_logits = tf.concat([true_logits, neg_logits], axis=1)
        preds = tf.argmax(all_logits, 1)
        labels = tf.zeros_like(preds)
        accuracy, update_op = tf.metrics.accuracy(labels, preds)
        return accuracy, update_op


      t_egs = build_t_eg(df, 't')  # get 3 ego_graphs for src node 't', (t_t_t_eg, t_n_n_eg, t_n_t_eg)
      tpos_egs = build_t_eg(df, 'tpos')
      tneg_egs = build_t_eg(df, 'tneg')
      n_egs = build_n_eg(df, 'n')  # get 3 ego_graphs for src node 'n', (n_n_n_eg, n_n_t_eg, n_t_n_eg)
      nneg_egs = build_n_eg(df, 'nneg')

      t_embeddings = forward_group((model_t_t_t, model_t_n_n, model_t_n_t), t_egs)
      tpos_embeddings = forward_group((model_t_t_t, model_t_n_n, model_t_n_t), tpos_egs)
      tneg_embeddings = forward_group((model_t_t_t, model_t_n_n, model_t_n_t), tneg_egs)
      tneg_embeddings = tf.reshape(tneg_embeddings, [-1, NEG, DIM])

      n_embeddings = forward_group((model_n_n_n, model_n_n_t, model_n_t_n), n_egs)
      nneg_embeddings = forward_group((model_n_n_n, model_n_n_t, model_n_t_n), nneg_egs)
      nneg_embeddings = tf.reshape(nneg_embeddings, [-1, NEG, DIM])

      accuracy_train, update_op_train = tuple( 0.5*a + 0.5*b for a, b in zip(build_acc(t_embeddings, n_embeddings, nneg_embeddings), build_acc(t_embeddings, tpos_embeddings, tneg_embeddings)))
      loss = 0.5*build_bpr_loss(t_embeddings, n_embeddings, nneg_embeddings) + 0.5*build_bpr_loss(t_embeddings, tpos_embeddings, tneg_embeddings)


      t_egs_save = build_t_eg(df_save_topic, 't')  # get 3 ego_graphs for src node 't', (t_t_t_eg, t_n_n_eg, t_n_t_eg)
      n_egs_save = build_n_eg(df_save_note, 'n')   # get 3 ego_graphs for src node 'n', (n_n_n_eg, n_n_t_eg, n_t_n_eg)
      
      t_save_emb = forward_group((model_t_t_t, model_t_n_n, model_t_n_t), t_egs_save)  # t_egs_save has 3 ego_graphs (t_t_t_eg, t_n_n_eg, t_n_t_eg)
      n_save_emb = forward_group((model_n_n_n, model_n_n_t, model_n_t_n), n_egs_save)

      t_strings = t_egs_save[0].nodes.string_attrs[0, :]
      n_strings = n_egs_save[0].nodes.string_attrs[0, :]

      t_emb = tf.nn.l2_normalize(t_save_emb, axis=-1)
      n_emb = tf.nn.l2_normalize(n_save_emb, axis=-1)

      t_emb_path = os.path.join(FLAGS.save_emb_dir, 'topic_emb_part{}.txt'.format(task_index))
      n_emb_path = os.path.join(FLAGS.save_emb_dir, 'note_emb_part{}.txt'.format(task_index))
      if os.path.exists(t_emb_path):
        os.remove(t_emb_path)
      if os.path.exists(n_emb_path):
        os.remove(n_emb_path)

    trainer.train(df.iterator, loss, FLAGS.learning_rate, epochs=FLAGS.train_epochs, metrics=[accuracy_train, update_op_train])
    trainer.save_node_embedding(t_emb_path, df_save_topic.iterator, t_strings, t_emb, FLAGS.dataset_batch_size)
    trainer.save_node_embedding(n_emb_path, df_save_note.iterator, n_strings, n_emb, FLAGS.dataset_batch_size)

    trainer.close()
    g.close()
    print("graph closed")



def define_custom_flags():
    flags.DEFINE_integer(name='emb_max_partitions', default=12, help='The max_partitions for embedding variables for embedding variables partitioned by min_max_variable_partitioner')
    flags.DEFINE_integer(name='emb_min_slice_size', default=64 * 1024, help='The min_slice_size for embedding variables for embedding variables partitioned by min_max_variable_partitioner')
    flags.DEFINE_integer(name='emb_live_steps', default=30 * 1600000, help='Global steps to live for inactive keys in embedding variables')

    flags.DEFINE_string(name='train_data_base_dir', default=None, help='Training data base dir') # 填/root/data/topic_graph/data/v2_rename/train_data_end_dtm/train_data_end_hr/
    flags.DEFINE_string(name='train_data_node_dir', default=None, help='Training data node dir, split by comma') # node_topic,node_note
    flags.DEFINE_string(name='train_data_edge_dir', default=None, help='Training data edge dir, split by comma') # edge_swing,edge_t2n,edge_n2t,edge_topic

    flags.DEFINE_string(name='train_data_end_dtm', default=None, help='Training data end date')
    flags.DEFINE_string(name='train_data_end_hr', default=None, help='Training data end hour')
    flags.DEFINE_string(name='train_data_dtm_format', default='%Y%m%d', help='Format of date string')
    flags.DEFINE_integer(name='train_epochs', default=1, help='Number of epochs used to train')
    flags.DEFINE_integer(name='dataset_batch_size', default=64, help='Batch size')
    flags.DEFINE_float(name='learning_rate', default=0.0001, help='Learning rate')
    
    flags.DEFINE_string(name='load_model_dir', default="", help='Model dir for load before training') # /root/data/topic_graph/model/20220222/
    flags.DEFINE_string(name='save_model_dir', default=None, help='Model dir for save after training') # 
    flags.DEFINE_integer(name='save_checkpoints_cycle', default=36000, help='How many seconds to save checkpoint')
    flags.DEFINE_string(name='save_emb_dir', default=None, help='Save emb dir')
    flags.DEFINE_integer(name='save_emb_num', default=None, help='Save emb file num, generally comes with node type num * worker num')

if __name__ == "__main__":
    print("tf version:{}".format(tf.__version__))
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    define_custom_flags()

    assert FLAGS.train_data_base_dir is not None
    assert FLAGS.train_data_node_dir is not None
    assert FLAGS.train_data_edge_dir is not None
    assert FLAGS.train_data_end_dtm is not None
    assert FLAGS.train_data_end_hr is not None
    assert FLAGS.save_emb_dir is not None
    assert FLAGS.save_model_dir is not None
    assert FLAGS.save_emb_num is not None

    tfg.conf.emb_max_partitions = FLAGS.emb_max_partitions
    tfg.conf.emb_min_slice_size = FLAGS.emb_min_slice_size
    tfg.conf.emb_live_steps = FLAGS.emb_live_steps

    # feature side
    DISCRETE_EMB_SIZE = 16
    TOPIC_FEA_NUM = 1
    ITEM_FEA_NUM = 4

    # training side
    BATCH_SIZE = FLAGS.dataset_batch_size
    MARGIN = 2.0
    DROPOUT = 0.0
    COSINE_SCALE = 2.0
    HOP_1_T = 5
    HOP_2_T = 10
    HOP_1_N = 10
    HOP_2_N = 10
    HOP_N_T = 3
    NEG = 5
    LAYER_NUM = 2
    DIM = 32

    train()

    print("main finished")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
