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

from ego_bipartite_sage import EgoBipartiteGraphSAGE
from hgt_rgcn_data_loader import HGTRgcnDataLoader
from process_hgt_config import get_feature_config
from trainer_saver import DistTrainer
from ego_sage_data_loader import EgoSAGEUnsupervisedDataLoader


# np.set_printoptions(threshold=sys.maxsize)

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
    loss = tf.reduce_mean(-tf.log_sigmoid(COSINE_SCALE * (true_logits - neg_logits)))
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
    # weights = tf.reshape(tf.tile(tf.expand_dims(tf.reshape(weights, [-1]), axis=1), [1, neg_expand]), [-1])
    loss = tf.reduce_mean(-tf.log_sigmoid(COSINE_SCALE * (true_logits - neg_logits - adaptive_margin)))
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
    if FLAGS.train_mode == 'all':
        g = gl.Graph() \
            .node(TRAINING_DATA_NODE_LIST[0], node_type='user',
                  decoder=user_decoder) \
            .node(TRAINING_DATA_NODE_LIST[1], node_type='note',
                  decoder=note_decoder) \
            .edge(TRAINING_DATA_EDGE_LIST[0], edge_type=('user', 'note', 'cmt_n'),
                  decoder=gl.Decoder(weighted=True, labeled=True), directed=False) \
            .edge(TRAINING_DATA_EDGE_LIST[1], edge_type=('user', 'note', 'sa_n'),
                  decoder=gl.Decoder(weighted=True, labeled=True), directed=False)  \
            .edge(TRAINING_DATA_EDGE_LIST[2], edge_type=('user', 'note', 'fo_n'),
              decoder=gl.Decoder(weighted=True, labeled=True), directed=False)
    elif FLAGS.train_mode == 'incremental':
        g = gl.Graph() \
            .node(TRAINING_DATA_NODE_LIST[0], node_type='user',
                  decoder=user_decoder) \
            .node(TRAINING_DATA_NODE_LIST[1], node_type='note',
                  decoder=note_decoder) \
            .edge(TRAINING_DATA_EDGE_LIST[0], edge_type=('user', 'note', 'cmt_n'),
                  decoder=gl.Decoder(weighted=True, labeled=True), directed=False) \
            .edge(TRAINING_DATA_EDGE_LIST[1], edge_type=('user', 'note', 'sa_n'),
                  decoder=gl.Decoder(weighted=True, labeled=True), directed=False) \
            .edge(TRAINING_DATA_EDGE_LIST[2], edge_type=('user', 'note', 'fo_n'),
                  decoder=gl.Decoder(weighted=True, labeled=True), directed=False) \
            .edge(TRAINING_DATA_EDGE_LIST[0], edge_type=('user', 'note', 'label_cmt_n'),
                  decoder=gl.Decoder(weighted=True, labeled=True), directed=False) \
            .edge(TRAINING_DATA_EDGE_LIST[1], edge_type=('user', 'note', 'label_sa_n'),
                  decoder=gl.Decoder(weighted=True, labeled=True), directed=False) \
            .edge(TRAINING_DATA_EDGE_LIST[2], edge_type=('user', 'note', 'label_fo_n'),
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
        trainer = DistTrainer(tf_cluster, job_name, task_index, worker_count, ckpt_dir=FLAGS.save_model_dir,
                              ckpt_freq=FLAGS.save_checkpoints_cycle)

        if job_name == "worker":
            with trainer.context():
                feature_config = get_feature_config(FLAGS.config_path)
                user_attr_dims = feature_config['userFeatures']['attr_dims']
                note_attr_dims = feature_config['noteFeatures']['attr_dims']

                u_input_dim = sum([1 if not i else i for i in user_attr_dims])
                i_input_dim = sum([1 if not i else i for i in note_attr_dims])

                print("user_dim = {}".format(u_input_dim))
                print("note_dim = {}".format(i_input_dim))

                iterator_list, total_loss, accuracy_train, update_op_train = train_model(g, u_input_dim, i_input_dim)

            total_loss = total_loss/len(relation_list)
            accuracy_train = accuracy_train/len(relation_list)
            update_op_train = update_op_train/len(relation_list)
            # train
            trainer.train(iterator_list, total_loss, learning_rate=FLAGS.learning_rate, epochs=FLAGS.train_epochs,
                          metrics=[accuracy_train, update_op_train])
        else:
            print("TF PS start...")
            trainer.join()

        # query_save_topic = build_query_save_topic(g)
        # query_save_note = build_query_save_note(g)

        trainer.close()
        g.close()
        print("graph closed")

def train_model(g, u_input_dim, i_input_dim):
    # many models,model_t_n_n = build_lightgcn("t_n_n", layer_num, t_dim, n_dim, [0, 1, 1], DIM)
    # two tower model for u and i.
    # user => user_follow_n_follow_user
    # user => user_share_n_share_user
    # user => user_comment_n_comment_user
    # user => user_follow_author_follow_user
    # note => note_follow_user_follow_note
    # note => note_share_user_share_note
    # note => note_comment_user_comment_note
    total_loss = 0.0
    accuracy_train = 0.0
    update_op_train = 0.0
    # define meta_path_model for various relations
    model_dict = {}
    # tfg.conf.training = True
    u_hidden_dims = [FLAGS.hidden_dim] * (len(u_nbrs_num) - 1) + [output_dim]
    i_hidden_dims = [FLAGS.hidden_dim] * (len(i_nbrs_num) - 1) + [output_dim]
    for edge_type, alias in relation_alias.items():
        model_dict[alias + "_user"] = EgoBipartiteGraphSAGE("src_" + alias,
                                                            u_input_dim,
                                                            i_input_dim,
                                                            u_hidden_dims,
                                                            agg_type=FLAGS.agg_type,
                                                            dropout=FLAGS.drop_out,
                                                            i2i=i2i_path != "")

        dst_input_dim = i_input_dim if i2i_path != "" else u_input_dim
        model_dict[alias + "_note"] = EgoBipartiteGraphSAGE("dst_" + alias,
                                                            i_input_dim,
                                                            dst_input_dim,
                                                            i_hidden_dims,
                                                            agg_type=FLAGS.agg_type,
                                                            dropout=FLAGS.drop_out,
                                                            i2i=i2i_path != "")

    iterator_list = []
    for relation in relation_list:
        src_name = 'src_' + relation
        dst_name = 'dst_' + relation
        neg_dst_name = 'neg_dst_' + relation
        src_res = 0
        dst_res = 0
        neg_dst_res = 0
        neg_src_res = 0
        dataLoader = HGTRgcnDataLoader(g, None, sampler=FLAGS.sampler, batch_relation=relation,
                                       batch_size=FLAGS.batch_size, window=10, train_mode=FLAGS.train_mode,
                                       neg_num=FLAGS.neg_num, neg_sampler=FLAGS.neg_sampler,
                                       u_nbrs_num=u_nbrs_num, i_nbrs_num=i_nbrs_num,
                                       relation_alias=relation_alias)

        iterator_list.append(dataLoader._iterator)
        for edge_type, alias in relation_alias.items():
            u_model = model_dict.get(alias + "_user")
            i_model = model_dict.get(alias + "_note")
            # prepare train dataset
            # train and save node embeddings.
            src_ego = dataLoader.get_relation_ego('src', relation, edge_type, 'user')
            dst_ego = dataLoader.get_relation_ego('dst', relation, edge_type, 'note')
            neg_dst_ego = dataLoader.get_relation_ego('neg_dst', relation, edge_type, 'note')
            neg_src_ego = dataLoader.get_relation_ego('neg_src', relation, edge_type, 'user')
            src_emb = u_model.forward(src_ego)
            dst_emb = i_model.forward(dst_ego)
            neg_dst_emb = i_model.forward(neg_dst_ego)
            neg_src_emb = u_model.forward(neg_src_ego)
            src_res += src_emb
            dst_res += dst_emb
            neg_dst_res += neg_dst_emb
            neg_src_res += neg_src_emb
        # use sampled softmax loss with temperature.
        # loss = tfg.unsupervised_softmax_cross_entropy_loss(output_embeddings, dst_emb, neg_dst_emb,
        #                                                   temperature=0.07)
        src_res = src_res / len(relation_alias)
        dst_res = dst_res / len(relation_alias)
        neg_dst_res = neg_dst_res / len(relation_alias)
        neg_src_res = neg_src_res / len(relation_alias)
        neg_dst_res = tf.reshape(neg_dst_res, [-1, FLAGS.neg_num, output_dim])
        neg_src_res = tf.reshape(neg_src_res, [-1, FLAGS.neg_num, output_dim])

        accuracy_train, update_op_train = tuple(0.5 * a + 0.5 * b for a, b in
                                                            zip(build_acc(src_res, dst_res,
                                                                          neg_dst_res),
                                                                build_acc(src_res, dst_res,
                                                                          neg_src_res)))
        loss = 0.5 * build_bpr_loss(src_res, dst_res, neg_dst_res) + 0.5 * build_bpr_loss(src_res, dst_res, neg_src_res)
        total_loss += loss
        accuracy_train += accuracy_train
        update_op_train += update_op_train
        # loss = tfg.unsupervised_softmax_cross_entropy_loss(src_embeddings, dst_embeddings, neg_dst_embeddings,temperature=0.07)
    return iterator_list, total_loss, accuracy_train, update_op_train

def define_custom_flags():
    # flags.DEFINE_integer(name='emb_max_partitions', default=20,
    #                      help='The max_partitions for embedding variables for embedding variables partitioned by min_max_variable_partitioner')
    flags.DEFINE_integer(name='emb_min_slice_size', default=64 * 1024,
                         help='The min_slice_size for embedding variables for embedding variables partitioned by min_max_variable_partitioner')
    flags.DEFINE_integer(name='emb_live_steps', default=30 * 1600000,
                         help='Global steps to live for inactive keys in embedding variables')

    flags.DEFINE_string(name='train_data_base_dir', default=None,
                        help='Training data base dir')  # 填/root/data/topic_graph/data/v2_rename/train_data_end_dtm/train_data_end_hr/
    flags.DEFINE_string(name='train_data_node_dir', default=None,
                        help='Training data node dir, split by comma')  # node_topic,node_note
    flags.DEFINE_string(name='train_data_edge_dir', default=None,
                        help='Training data edge dir, split by comma')  # edge_swing,edge_t2n,edge_n2t,edge_topic

    flags.DEFINE_string(name='train_data_end_dtm', default=None, help='Training data end date')
    flags.DEFINE_string(name='train_data_end_hr', default=None, help='Training data end hour')
    flags.DEFINE_string(name='train_data_dtm_format', default='%Y%m%d', help='Format of date string')
    flags.DEFINE_integer(name='train_epochs', default=1, help='Number of epochs used to train')
    flags.DEFINE_integer(name='batch_size', default=64, help='Batch size')
    flags.DEFINE_float(name='learning_rate', default=0.0001, help='Learning rate')
    flags.DEFINE_string(name='config_path', default='hgt_config_with_bucket.json',
                        help='config path')  # hgt_config_with_bucket.json
    flags.DEFINE_integer(name='neg_num', default=2, help='neg number')

    flags.DEFINE_string(name='sampler', default='edge_weight',
                        help='sampler strategy')
    flags.DEFINE_string(name='neg_sampler', default='in_degree',
                        help='neg sampler strategy')
    flags.DEFINE_string(name='agg_type', default='mean',
                        help='agg type')
    flags.DEFINE_float(name='drop_out', default=0.0, help='drop out rate')
    flags.DEFINE_integer(name='hidden_dim', default=64, help='hidden dim')

    flags.DEFINE_string(name='nbrs_num', default='[10,5]',
                        help='neigors number for node')
    flags.DEFINE_string(name='train_mode', default='incremental',
                        help='all train or incremental train')

    flags.DEFINE_string(name='load_model_dir', default="",
                        help='Model dir for load before training')  # /root/data/topic_graph/model/20220222/
    flags.DEFINE_string(name='save_model_dir', default=None, help='Model dir for save after training')  #
    flags.DEFINE_integer(name='save_checkpoints_cycle', default=36000, help='How many seconds to save checkpoint')
    flags.DEFINE_string(name='save_emb_dir', default=None, help='Save emb dir')
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


    relation_list = ["cmt_n", "sa_n", "fo_n"]
    relation_alias = {"cmt_n": "cmt", "sa_n": "sa", "fo_n": "fo"}


    # training side

    u_nbrs_num = json.loads(FLAGS.nbrs_num)
    i_nbrs_num = json.loads(FLAGS.nbrs_num)
    output_dim = 64
    i2i_path = ""
    MARGIN = 2.0
    COSINE_SCALE = 2.0


    train()

    print("main finished")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
