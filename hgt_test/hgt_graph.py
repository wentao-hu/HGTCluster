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

from process_hgt_config import get_feature_config
from trainer import DistTrainer

gl.set_tracker_mode(0)

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
  """ Load node and edge data to build graph.
    Note that node_type must be "i", and edge_type must be "r_i",
    the number of edge tables must be the same as FLAGS.num_relations.
  """
  feature_config = get_feature_config('hgt_config_simple.json')
  user_attr_types = feature_config['userFeatures']['attr_types']
  user_attr_dims = feature_config['userFeatures']['attr_dims']
  user_decoder = gl.Decoder(labeled=False, attr_types=user_attr_types, attr_dims=user_attr_dims, attr_delimiter=":")

  author_attr_types = feature_config['authorFeatures']['attr_types']
  author_attr_dims = feature_config['authorFeatures']['attr_dims']
  author_decoder = gl.Decoder(labeled=False, attr_types=author_attr_types, attr_dims=author_attr_dims,
                              attr_delimiter=":")

  note_attr_types = feature_config['noteFeatures']['attr_types']
  note_attr_dims = feature_config['noteFeatures']['attr_dims']
  note_decoder = gl.Decoder(labeled=False, attr_types=note_attr_types, attr_dims=note_attr_dims, attr_delimiter=":")
  decoder_dict = {}
  decoder_dict['user_decoder'] = user_decoder
  decoder_dict['author_decoder'] = author_decoder
  decoder_dict['note_decoder'] = note_decoder

  print("debug data dir: {}, {}".format(TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST))

  g = gl.Graph() \
    .node(TRAINING_DATA_NODE_LIST[0], node_type='user',
          decoder=user_decoder) \
    .node(TRAINING_DATA_NODE_LIST[2], node_type='note',
          decoder=note_decoder) \
    .edge(TRAINING_DATA_EDGE_LIST[0], edge_type=('user', 'note', 'follow_note'),
          decoder=gl.Decoder(weighted=True, labeled=True), directed=False)
    # .edge(TRAINING_DATA_EDGE_LIST[1], edge_type=('user', 'user', 'share_to_user'),
    #       decoder=gl.Decoder(weighted=True, labeled=True), directed=False)
  return g, decoder_dict


def build_query(g):
  seed = g.E("follow_note").batch(BATCH_SIZE).shuffle().alias('edge')
  src = seed.outV().alias('src')
  dst = seed.inV().alias('dst')
  neg_dst = src.outNeg("follow_note").sample(NEG).by("in_degree").alias('neg_dst')
  src_ego = src.outV("follow_note").sample(10).by("edge_weight").alias('src_ego')
  dst_ego = dst.inV("follow_note").sample(10).by("edge_weight").alias('dst_ego')
  neg_dst_ego = neg_dst.inV("follow_note").sample(10).by("edge_weight").alias('neg_dst_ego')
  return seed.values()

# def build_t_eg(df):
#   t_n_n_eg = df.get_ego_graph('t', neighbors=['t_n', 't_n_n'])
#   return (t_n_n_eg,)

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


def get_node_edge_path(task_index):
  train_data_base_dir = FLAGS.train_data_base_dir
  # tmp = train_data_base_dir.split('/')
  # if tmp[-1] == '':
  #     tmp = tmp[:-2]
  # else:
  #     tmp= tmp[:-1]
  # train_data_base_dir = '/'.join(tmp)+'/'

  TRAINING_DATA_NODE_LIST = []
  for node in FLAGS.train_data_node_dir.split(','):
    cur_dir = os.path.join(train_data_base_dir, node)

    filenames = os.listdir(cur_dir)
    if '_SUCCESS' in filenames:
      filenames.remove('_SUCCESS')
    while '.ipynb_checkpoints' in filenames:
      filenames.remove('.ipynb_checkpoints')
    filenames.sort()
    print('== len of node filenames', len(filenames), filenames)

    if task_index <= len(filenames) - 1:
      target_file_name = filenames[task_index]
    else:
      target_file_name = 'part-' + str(task_index) + '.csv'
    # print('cur_dir is',cur_dir,'\n','len of filenames and its content',len(filenames),filenames,'\n','task_index is',task_index,'\n','target_file_name is:',target_file_name)
    TRAINING_DATA_NODE_LIST.append(os.path.join(cur_dir, target_file_name))

  TRAINING_DATA_EDGE_LIST = []
  for edge in FLAGS.train_data_edge_dir.split(','):
    cur_dir = os.path.join(train_data_base_dir, edge)
    filenames = os.listdir(cur_dir)
    if '_SUCCESS' in filenames:
      filenames.remove('_SUCCESS')
    while '.ipynb_checkpoints' in filenames:
      filenames.remove('.ipynb_checkpoints')
    filenames.sort()
    print('== len of edge filenames', len(filenames), filenames)
    if task_index <= len(filenames) - 1:
      target_file_name = filenames[task_index]
    else:
      target_file_name = 'part-' + str(task_index) + '.csv'
    # print('cur_dir is',cur_dir,'\n','len of filenames and its content',len(filenames),filenames,'\n','task_index is',task_index,'\n','target_file_name is:',target_file_name)
    TRAINING_DATA_EDGE_LIST.append(os.path.join(cur_dir, target_file_name))
  return TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST

def train():
  gl_cluster, tf_cluster, job_name, task_index = gl.get_cluster()
  print("gl_cluster={}, tf_cluster={}, job_name={}, task_index={}".format(
    gl_cluster, tf_cluster, job_name, task_index
  ))
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
    TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST = get_node_edge_path(task_index)
    print('TRAINING_DATA_NODE_LIST is', TRAINING_DATA_NODE_LIST, '\n', 'TRAINING_DATA_EDGE_LIST is',
          TRAINING_DATA_EDGE_LIST)

    if task_index == 0:
      ######################################### to rm
      # training data done

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


    g, decoder_dict = init_graph(TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST)
    g.init(cluster=gl_cluster, job_name="server", task_index=task_index)
    topo = g.get_topology()
    topo.print_all()
    print("GraphLearn Server start...")
    g.wait_for_close()

    # TODO: disable save node embedding
    # if task_index == 0:
    #   for i in range(200):
    #     _, emb_num = commands.getstatusoutput("ls {} | grep part | wc -l".format(FLAGS.save_emb_dir))
    #     if int(emb_num) == FLAGS.save_emb_num:
    #       os.system('touch {}'.format(TRAIN_DONE))
    #       os.system('touch {}'.format(EMB_DONE))
    #     else:
    #       print("check commands:{}. now save {}/{}, waiting all workers to save embedding..." \
    #         .format("ls {} | grep part | wc -l".format(FLAGS.save_emb_dir), int(emb_num), FLAGS.save_emb_num))
    #       time.sleep(5)

  elif job_name == "ps":

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
    TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST = get_node_edge_path(task_index)
    print('TRAINING_DATA_NODE_LIST is', TRAINING_DATA_NODE_LIST, '\n', 'TRAINING_DATA_EDGE_LIST is',
          TRAINING_DATA_EDGE_LIST)


    for i in range(200):
      if os.path.isfile(SYNC_DONE):
        break
      else:
        print("waiting ckpt model dir ... {}".format(SYNC_DONE))
        time.sleep(60)

    g, decoder_dict = init_graph(TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST)
    g.init(cluster=gl_cluster, job_name="client", task_index=task_index)

    query = build_query(g)
    # query_save_topic = build_query_save_topic(g)
    # query_save_note = build_query_save_note(g)

    trainer = DistTrainer(tf_cluster, job_name, task_index, tf_cluster.num_tasks("worker"), ckpt_dir=FLAGS.save_model_dir,
              profiling=False)
    trainer.ckpt_freq = FLAGS.save_checkpoints_cycle
    with trainer.context():
      df = tfg.DataFlow(query)   # get 3 DataFlow
      # df_save_topic = tfg.DataFlow(query_save_topic)
      # df_save_note = tfg.DataFlow(query_save_note)
      # trainer.monitor(df)
      feature_config = get_feature_config('hgt_config_simple.json')
      user_attr_dims = feature_config['userFeatures']['attr_dims']
      note_attr_dims = feature_config['noteFeatures']['attr_dims']
      user_dim = sum([1 if not i else i for i in user_attr_dims])
      note_dim = sum([1 if not i else i for i in note_attr_dims])
      print("user_dim = {}".format(user_dim))
      print("note_dim = {}".format(note_dim))
      layer_num = 1
      # many models,model_t_n_n = build_lightgcn("t_n_n", layer_num, t_dim, n_dim, [0, 1, 1], DIM)
      model_user = build_lightgcn("user", layer_num, user_dim, note_dim, [0, 1], DIM)
      model_note = build_lightgcn("note", layer_num, user_dim, note_dim, [1, 0], DIM)

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

      src_ego = df.get_ego_graph("src", neighbors=['src_ego'])
      dst_ego = df.get_ego_graph("dst", neighbors=['dst_ego'])
      neg_dst_ego = df.get_ego_graph("neg_dst", neighbors=['neg_dst_ego'])
      src_embeddings = model_user.forward(src_ego)
      dst_embeddings = model_note.forward(dst_ego)
      neg_dst_embeddings = model_note.forward(neg_dst_ego)


      print(tf.shape(src_embeddings))
      print(tf.shape(dst_embeddings))
      print(tf.shape(neg_dst_embeddings))

      neg_dst_embeddings = tf.reshape(neg_dst_embeddings, [-1, NEG, DIM])

      accuracy_train, update_op_train = tuple(build_acc(src_embeddings, dst_embeddings, neg_dst_embeddings))
      loss = build_bpr_loss(src_embeddings, dst_embeddings, neg_dst_embeddings)


      # t_egs_save = build_t_eg(df_save_topic, 't')  # get 3 ego_graphs for src node 't', (t_t_t_eg, t_n_n_eg, t_n_t_eg)
      # n_egs_save = build_n_eg(df_save_note, 'n')   # get 3 ego_graphs for src node 'n', (n_n_n_eg, n_n_t_eg, n_t_n_eg)
      #
      # t_save_emb = forward_group((model_t_t_t, model_t_n_n, model_t_n_t), t_egs_save)  # t_egs_save has 3 ego_graphs (t_t_t_eg, t_n_n_eg, t_n_t_eg)
      # n_save_emb = forward_group((model_n_n_n, model_n_n_t, model_n_t_n), n_egs_save)
      #
      # t_strings = t_egs_save[0].nodes.string_attrs[0, :]
      # n_strings = n_egs_save[0].nodes.string_attrs[0, :]
      #
      # t_emb = tf.nn.l2_normalize(t_save_emb, axis=-1)
      # n_emb = tf.nn.l2_normalize(n_save_emb, axis=-1)
      #
      # t_emb_path = os.path.join(FLAGS.save_emb_dir, 'topic_emb_part{}.txt'.format(task_index))
      # n_emb_path = os.path.join(FLAGS.save_emb_dir, 'note_emb_part{}.txt'.format(task_index))
      # if os.path.exists(t_emb_path):
      #   os.remove(t_emb_path)
      # if os.path.exists(n_emb_path):
      #   os.remove(n_emb_path)

    trainer.train(df.iterator, loss, FLAGS.learning_rate, epochs=FLAGS.train_epochs, metrics=[accuracy_train, update_op_train])
    # trainer.save_node_embedding(t_emb_path, df_save_topic.iterator, t_strings, t_emb, FLAGS.dataset_batch_size)
    # trainer.save_node_embedding(n_emb_path, df_save_note.iterator, n_strings, n_emb, FLAGS.dataset_batch_size)

    trainer.close()
    g.close()
    print("graph closed")



def define_custom_flags():
    flags.DEFINE_integer(name='emb_max_partitions', default=20, help='The max_partitions for embedding variables for embedding variables partitioned by min_max_variable_partitioner')
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
    print("FLAGS.emb_max_partitions={}, FLAGS.emb_min_slice_size={}, FLAGS.emb_live_steps={}".format(
      FLAGS.emb_max_partitions, FLAGS.emb_min_slice_size, FLAGS.emb_live_steps
    ))

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
