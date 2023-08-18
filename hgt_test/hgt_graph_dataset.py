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

from process_hgt_config import get_feature_config
from trainer_v2 import DistTrainer
from ego_sage_data_loader import EgoSAGEUnsupervisedDataLoader
from ego_sage import EgoGraphSAGE


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
        .edge(TRAINING_DATA_EDGE_LIST[0], edge_type=('user', 'user', 'share_to_user'),
              decoder=gl.Decoder(weighted=True, labeled=True), directed=False)
    # .edge(TRAINING_DATA_EDGE_LIST[1], edge_type=('user', 'user', 'share_to_user'),
    #       decoder=gl.Decoder(weighted=True, labeled=True), directed=False)
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
        trainer = DistTrainer(tf_cluster, job_name, task_index, worker_count)
        if job_name == "worker":
            with trainer.context():
                feature_config = get_feature_config('hgt_config_simple.json')
                user_attr_dims = feature_config['userFeatures']['attr_dims']

                user_dim = sum([1 if not i else i for i in user_attr_dims])

                print("user_dim = {}".format(user_dim))

                nbrs_num = [10, 5]
                layer_num = 1
                # many models,model_t_n_n = build_lightgcn("t_n_n", layer_num, t_dim, n_dim, [0, 1, 1], DIM)
                dims = [user_dim] + [128] * (len(nbrs_num) - 1) + [DIM]
                model = EgoGraphSAGE(dims,
                                     agg_type='mean',
                                     dropout=0.0)

                # prepare train dataset
                train_data = EgoSAGEUnsupervisedDataLoader(g, None, "random", "random", BATCH_SIZE,
                                                           node_type='user', edge_type='share_to_user',
                                                           nbrs_num=nbrs_num)
                src_embeddings = model.forward(train_data.src_ego)
                dst_embeddings = model.forward(train_data.dst_ego)
                neg_dst_embeddings = model.forward(train_data.neg_dst_ego)

                print(tf.shape(src_embeddings))
                print(tf.shape(dst_embeddings))
                print(tf.shape(neg_dst_embeddings))



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
                    neg_logits = tf.reshape(neg_logits, [-1, NEG])
                    all_logits = tf.concat([true_logits, neg_logits], axis=1)
                    preds = tf.argmax(all_logits, 1)
                    labels = tf.zeros_like(preds)
                    accuracy, update_op = tf.metrics.accuracy(labels, preds)
                    return accuracy, update_op

                neg_dst_embeddings = tf.reshape(neg_dst_embeddings, [-1, NEG, DIM])
                accuracy_train, update_op_train = tuple(build_acc(src_embeddings, dst_embeddings, neg_dst_embeddings))
                loss = build_bpr_loss(src_embeddings, dst_embeddings, neg_dst_embeddings)
                #loss = tfg.unsupervised_softmax_cross_entropy_loss(src_embeddings, dst_embeddings, neg_dst_embeddings,temperature=0.07)

            # train
            trainer.train(train_data.iterator, loss, learning_rate=FLAGS.learning_rate, epochs=FLAGS.train_epochs,
                          metrics=[accuracy_train, update_op_train])
        else:
            print("TF PS start...")
            trainer.join()

        # query_save_topic = build_query_save_topic(g)
        # query_save_note = build_query_save_note(g)

        g.close()
        print("graph closed")


def define_custom_flags():
    flags.DEFINE_integer(name='emb_max_partitions', default=20,
                         help='The max_partitions for embedding variables for embedding variables partitioned by min_max_variable_partitioner')
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
    flags.DEFINE_integer(name='dataset_batch_size', default=64, help='Batch size')
    flags.DEFINE_float(name='learning_rate', default=0.0001, help='Learning rate')

    flags.DEFINE_string(name='load_model_dir', default="",
                        help='Model dir for load before training')  # /root/data/topic_graph/model/20220222/
    flags.DEFINE_string(name='save_model_dir', default=None, help='Model dir for save after training')  #
    flags.DEFINE_integer(name='save_checkpoints_cycle', default=36000, help='How many seconds to save checkpoint')
    flags.DEFINE_string(name='save_emb_dir', default=None, help='Save emb dir')
    flags.DEFINE_integer(name='save_emb_num', default=None,
                         help='Save emb file num, generally comes with node type num * worker num')


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
