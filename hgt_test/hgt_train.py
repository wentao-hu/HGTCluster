# coding:utf-8
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
import json
from collections import OrderedDict

from ego_bipartite_sage import EgoBipartiteGraphSAGE
from process_hgt_config import get_feature_config
from trainer_v2 import DistTrainer
from hgt_data_loader import HGTDataLoader
from ego_hgt import EgoHGT


# np.set_printoptions(threshold=sys.maxsize)

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
    loss = tf.reduce_mean(-tf.log_sigmoid(COSINE_SCALE * (true_logits - neg_logits)))
    return loss


def init_graph(TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST):
    """ Load node and edge data to build graph.
      Note that node_type must be "i", and edge_type must be "r_i",
      the number of edge tables must be the same as FLAGS.num_relations.
    """
    feature_config = get_feature_config(FLAGS.config_path)
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
        .node(TRAINING_DATA_NODE_LIST[1], node_type='note',
              decoder=note_decoder) \
        .edge(TRAINING_DATA_EDGE_LIST[0], edge_type=('user', 'note', 'pos_comment_note'),
              decoder=gl.Decoder(weighted=True, labeled=True), directed=False) \
        .edge(TRAINING_DATA_EDGE_LIST[1], edge_type=('user', 'note', 'pos_follow_note'),
              decoder=gl.Decoder(weighted=True, labeled=True), directed=False) \
        .edge(TRAINING_DATA_EDGE_LIST[2], edge_type=('user', 'note', 'pos_share_note'),
              decoder=gl.Decoder(weighted=True, labeled=True), directed=False)
    return g, decoder_dict


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
    gl.set_tracker_mode(0)
    gl.set_timeout(300)
    gl.set_default_string_attribute("None")
    gl.set_padding_mode(gl.CIRCULAR)
    gl_cluster, tf_cluster, job_name, task_index = gl.get_cluster()
    print("gl_cluster={}, tf_cluster={}, job_name={}, task_index={}".format(
        gl_cluster, tf_cluster, job_name, task_index
    ))

    worker_count = len(tf_cluster["worker"])
    ps_hosts = tf_cluster.get("ps")
    # gl_cluster["server"] = ",".join([host.split(":")[0] + ":8889" for host in ps_hosts])
    # global settings.
    tfg.conf.emb_max_partitions = len(ps_hosts)  # embedding varible partition num.
    tf_cluster = tf.train.ClusterSpec(tf_cluster)

    print("gl version:{}".format(gl.__git_version__))

    # training data
    SYNC_DONE = os.path.join(FLAGS.save_model_dir, "_SYNC_DONE")
    TRAIN_DONE = os.path.join(FLAGS.train_data_base_dir, "_TRAIN_DONE")
    EMB_DONE = os.path.join(FLAGS.save_emb_dir, "_DONE")
    # training data
    TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST = get_node_edge_path(task_index)
    print('TRAINING_DATA_NODE_LIST is', TRAINING_DATA_NODE_LIST, '\n', 'TRAINING_DATA_EDGE_LIST is',
          TRAINING_DATA_EDGE_LIST)

    g, decoder_dict = init_graph(TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST)

    if job_name == "graphlearn":
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
                _, ckpt_version = commands.getstatusoutput(
                    "cat {} | head -1 | awk -F'-' '{{print $NF}}' | sed 's/\"//g'".format(
                        os.path.join(FLAGS.load_model_dir, "checkpoint")))
                print("ckpt_version: {}".format(ckpt_version))
                mkdir_status, mkdir_output = commands.getstatusoutput("mkdir -p {}".format(FLAGS.save_model_dir))
                shutil.copy(os.path.join(FLAGS.load_model_dir, "checkpoint"),
                            os.path.join(FLAGS.save_model_dir, "checkpoint"))

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

        g.init(cluster=gl_cluster, job_name="server", task_index=task_index)
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
        #       time.sleep(5
    elif job_name == "worker":
        g.init(cluster=gl_cluster, job_name="client", task_index=task_index)
    else:
        pass

    if job_name != "graphlearn":
        for i in range(200):
            if os.path.isfile(SYNC_DONE):
                break
            else:
                print("waiting ckpt model dir ... {}".format(SYNC_DONE))
                time.sleep(60)
        # training and save embedding.
        tf_cluster = tf.train.ClusterSpec(tf_cluster)
        trainer = DistTrainer(tf_cluster, job_name, task_index, worker_count, ckpt_dir=FLAGS.save_model_dir)
        if job_name == "worker":
            with trainer.context():
                input_dim_dict = define_feature()
                nbrs_num = json.loads(FLAGS.nbrs_num)  # string of list, neighbor num of each hop'
                num_layers = len(nbrs_num)

                # define model
                model = EgoHGT(FLAGS.hidden_dim,
                               num_layers,
                               pos_relation_dict,
                               input_dim_dict,
                               n_heads=FLAGS.n_heads,
                               dropout=FLAGS.drop_out,
                               use_norm=FLAGS.use_norm)

                iterator_list, total_loss, metrics = train_hgt(model, g, decoder_dict, pos_relation_dict, nbrs_num)

            # train
            trainer.train_list(iterator_list, total_loss, learning_rate=FLAGS.learning_rate, epochs=FLAGS.train_epochs,
                          metrics=metrics)
        else:
            print("TF PS start...")
            trainer.join()

        # query_save_topic = build_query_save_topic(g)
        # query_save_note = build_query_save_note(g)

        g.close()
        print("graph closed")

def train_hgt(model, g, decoder_dict, pos_relation_dict, nbrs_num):
    # define handler for node feature transformation
    user_feature_handler = tfg.FeatureHandler('user_feature_handler', decoder_dict['user_decoder'].feature_spec)
    author_feature_handler = tfg.FeatureHandler('author_feature_handler',
                                                decoder_dict['author_decoder'].feature_spec)
    note_feature_handler = tfg.FeatureHandler('note_feature_handler', decoder_dict['note_decoder'].feature_spec)

    # initialize loss and acc
    total_loss = tf.constant(0.0)  # initialize loss
    total_acc = tf.constant(0.0)  # initialize accuracy
    total_update_op = tf.constant(0.0)  # initialize update op
    iterator_list = []

    num_pos_neg_pairs = len(pos_relation_dict)  # or len(pos_relation_dict)
    for i in range(len(pos_relation_dict)):  # should be len(pos_relation_dict) for all training
        pos_relation = pos_relation_dict.keys()[i]
        # neg_relation = neg_relation_dict.keys()[i] , neg relation is generated by GSL, so we do not need it here
        print('===Start sampling for:', 'pos_relation:', pos_relation)

        train_data = HGTDataLoader(g, gl.Mask.TRAIN,
                                      sampler=FLAGS.sampler,
                                      neg_sampler=FLAGS.neg_sampler,
                                      batch_size=FLAGS.batch_size,
                                      pos_relation=pos_relation, neg_num=FLAGS.neg_num,
                                      pos_relation_dict=pos_relation_dict,
                                      user_feature_handler=user_feature_handler,
                                      author_feature_handler=author_feature_handler,
                                      note_feature_handler=note_feature_handler,
                                      nbrs_num=nbrs_num)
        iterator = train_data._iterator
        iterator_list.append(iterator)

        ## get neighborhood features
        pos_src_x_list, pos_dst_x_list, neg_src_x_list, neg_dst_x_list = train_data.x_list(train_data._q,
                                                                                           train_data._data_dict,
                                                                                           train_data._alias_dict,
                                                                                           train_data._node_type_path_dict)

        pos_src_relation = train_data._relation_path_dict['src']
        pos_dst_relation = train_data._relation_path_dict['dst']
        neg_src_relation = train_data._relation_path_dict['neg_src']
        neg_dst_relation = train_data._relation_path_dict['neg_dst']

        pos_src_node_type = train_data._node_type_path_dict['src']
        pos_dst_node_type = train_data._node_type_path_dict['dst']
        neg_src_node_type = train_data._node_type_path_dict['neg_src']
        neg_dst_node_type = train_data._node_type_path_dict['neg_dst']

        pos_src_relation_count = train_data._relation_count_dict['src']
        pos_dst_relation_count = train_data._relation_count_dict['dst']
        neg_src_relation_count = train_data._relation_count_dict['neg_src']
        neg_dst_relation_count = train_data._relation_count_dict['neg_dst']

        pos_src_embedding = model.forward(pos_src_x_list, pos_src_relation, pos_src_node_type,
                                          pos_src_relation_count, nbrs_num)
        pos_dst_embedding = model.forward(pos_dst_x_list, pos_dst_relation, pos_dst_node_type,
                                          pos_dst_relation_count, nbrs_num)

        neg_src_embedding = model.forward(neg_src_x_list, neg_src_relation, neg_src_node_type,
                                          neg_src_relation_count, nbrs_num)
        neg_dst_embedding = model.forward(neg_dst_x_list, neg_dst_relation, neg_dst_node_type,
                                          neg_dst_relation_count, nbrs_num)

        neg_src_embedding = tf.reshape(neg_src_embedding, [-1, FLAGS.neg_num, FLAGS.hidden_dim])
        neg_dst_embedding = tf.reshape(neg_dst_embedding, [-1, FLAGS.neg_num, FLAGS.hidden_dim])

        accuracy_train, update_op_train = tuple(0.5 * a + 0.5 * b for a, b in
                                                zip(build_acc(pos_src_embedding, pos_dst_embedding,
                                                              neg_dst_embedding),
                                                    build_acc(pos_dst_embedding, pos_src_embedding,
                                                              neg_src_embedding)))
        loss = 0.5 * build_bpr_loss(pos_src_embedding, pos_dst_embedding,
                                    neg_dst_embedding) + 0.5 * build_bpr_loss(pos_dst_embedding,
                                                                              pos_src_embedding,
                                                                              neg_src_embedding)

        total_loss += loss
        total_acc += accuracy_train
        total_update_op += update_op_train

    avg_acc = total_acc / (num_pos_neg_pairs)
    avg_update_op = total_update_op / (num_pos_neg_pairs)
    print('len of iterator list', len(iterator_list))
    metrics = [avg_acc, avg_update_op]
    return iterator_list, total_loss, metrics

def define_feature():
    # feature parameters
    feature_config = get_feature_config(FLAGS.config_path)
    user_attr_dims = feature_config['userFeatures']['attr_dims']
    author_attr_dims = feature_config['authorFeatures']['attr_dims']
    note_attr_dims = feature_config['noteFeatures']['attr_dims']
    user_input_dim = sum([1 if not i else i for i in user_attr_dims])
    author_input_dim = sum([1 if not i else i for i in author_attr_dims])
    note_input_dim = sum([1 if not i else i for i in note_attr_dims])
    input_dim_dict = OrderedDict()
    input_dim_dict['user'] = user_input_dim
    input_dim_dict['author'] = author_input_dim
    input_dim_dict['note'] = note_input_dim
    return input_dim_dict

# build pairwise acc
def build_acc(src, dst, neg):
    x1 = tf.nn.l2_normalize(src, axis=-1) * tf.nn.l2_normalize(dst, axis=-1)
    true_logits = tf.reduce_sum(x1, axis=-1, keepdims=True)
    dim = src.shape[1]
    neg_expand = neg.shape[1]
    src = tf.tile(tf.expand_dims(src, axis=1), [1, neg_expand, 1])
    src = tf.reshape(src, [-1, dim])
    neg = tf.reshape(neg, [-1, dim])
    x2 = tf.nn.l2_normalize(src, axis=-1) * tf.nn.l2_normalize(neg, axis=-1)
    # x2 = src * neg
    neg_logits = tf.reduce_sum(x2, axis=-1)
    neg_logits = tf.reshape(neg_logits, [-1, FLAGS.neg_num])
    all_logits = tf.concat([true_logits, neg_logits], axis=1)
    preds = tf.argmax(all_logits, 1)
    labels = tf.zeros_like(preds)
    accuracy, update_op = tf.metrics.accuracy(labels, preds)
    return accuracy, update_op

def define_custom_flags():
    # flags.DEFINE_integer(name='emb_max_partitions', default=20,
    #                      help='The max_partitions for embedding variables for embedding variables partitioned by min_max_variable_partitioner')
    flags.DEFINE_integer(name='emb_min_slice_size', default=64 * 1024,
                         help='The min_slice_size for embedding variables for embedding variables partitioned by min_max_variable_partitioner')
    flags.DEFINE_integer(name='emb_live_steps', default=30 * 1600000,
                         help='Global steps to live for inactive keys in embedding variables')

    # for data path
    flags.DEFINE_string(name='train_data_base_dir', default=None,
                        help='Training data base dir')  # /root/data/topic_graph/data/v2_rename/
    flags.DEFINE_string(name='train_data_node_dir', default=None,
                        help='Training data node dir, split by comma')  # node_topic,node_note
    flags.DEFINE_string(name='train_data_edge_dir', default=None,
                        help='Training data edge dir, split by comma')  # edge_swing,edge_t2n,edge_n2t,edge_topic

    # for model training
    flags.DEFINE_integer('train_epochs', default=1, help='Number of epochs used to train')
    flags.DEFINE_integer('batch_size', 32, 'training minibatch size')
    flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
    flags.DEFINE_float('drop_out', 0.5, 'drop out rate')
    flags.DEFINE_integer('hidden_dim', 64, 'hidden layer dim')
    flags.DEFINE_integer('n_heads', 8, 'number of relations')
    flags.DEFINE_boolean('use_norm', True, 'use norm for hgt aggregation')
    flags.DEFINE_string('nbrs_num', '[4,2]', 'string of list, neighbor num of each hop')
    flags.DEFINE_string('sampler', 'random', 'neighbor sampler strategy. random, in_degree, topk.')
    flags.DEFINE_string('neg_sampler', 'random', 'neighbor sampler strategy. random, in_degree, edge_weight, topk.')
    flags.DEFINE_integer('neg_num', 2, 'number of negative samples for each src/dst node in pos relation')
    flags.DEFINE_integer('max_training_step', None, 'max training steps')

    # for load and save model
    flags.DEFINE_string(name='load_model_dir', default="",
                        help='Model dir for load before training')  # /root/data/larc_plat/model/shequ_rec_retrieval/hgt/model/20230811184839/x2a_GNN_Training/
    flags.DEFINE_string(name='save_model_dir', default=None, help='Model dir for save after training')
    flags.DEFINE_integer(name='save_checkpoints_cycle', default=36000, help='How many seconds to save checkpoint')
    flags.DEFINE_string(name='save_emb_dir', default=None, help='Save emb dir')
    flags.DEFINE_string(name='train_mode', default="all", help='incremental/all')
    flags.DEFINE_string(name='config_path', default='hgt_config_with_bucket.json',
                        help='config path')  # hgt_config_with_bucket.json
    flags.DEFINE_integer(name='save_emb_num', default=None,
                         help='Save emb file num, generally comes with node type num * worker num')
    flags.DEFINE_integer(name='skip_infer', default=0,
                         help='if 1, skip infer user emb')
    flags.DEFINE_integer(name='skip_train', default=0,
                         help='if 1, skip_train')


if __name__ == "__main__":
    print("tf version:{}".format(tf.__version__))
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    define_custom_flags()

    assert FLAGS.train_data_base_dir is not None
    assert FLAGS.train_data_node_dir is not None
    assert FLAGS.train_data_edge_dir is not None
    assert FLAGS.save_emb_dir is not None
    assert FLAGS.save_model_dir is not None
    assert FLAGS.save_emb_num is not None

    tfg.conf.emb_min_slice_size = FLAGS.emb_min_slice_size
    tfg.conf.emb_live_steps = FLAGS.emb_live_steps

    print('====flags====')
    for key in tf.app.flags.FLAGS.flag_values_dict():
        print("TF_FLAGS ", key, FLAGS[key].value)


    pos_relation_dict = OrderedDict()
    pos_relation_dict['pos_comment_note'] = ['user', 'note']
    pos_relation_dict['pos_follow_note'] = ['user', 'note']
    pos_relation_dict['pos_share_note'] = ['user', 'note']
    
    COSINE_SCALE = 2.0

    train()

    print("main finished")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
