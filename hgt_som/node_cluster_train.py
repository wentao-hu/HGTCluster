import numpy as np
import time
import datetime
import argparse
import json

import graphlearn as gl
import tensorflow as tf
import graphlearn.python.nn.tf as tfg
import sys
import os
import shutil
import commands
from collections import OrderedDict

from ego_hgt_model import EgoHGT
from SOM_model import SOM
from node_dataloader import NodeDataLoader
from trainer_v3 import DistTrainer
from process_hgt_config import get_feature_config

gl.set_tracker_mode(0)
gl.set_retry_times(30)
gl.set_padding_mode(gl.CIRCULAR)


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
        .edge(TRAINING_DATA_EDGE_LIST[0], edge_type=('user', 'note', 'pos_follow_note'),
              decoder=gl.Decoder(weighted=True, labeled=True), directed=False) \
        .edge(TRAINING_DATA_EDGE_LIST[1], edge_type=('user', 'note', 'pos_comment_note'),
              decoder=gl.Decoder(weighted=True, labeled=True), directed=False) \
        .edge(TRAINING_DATA_EDGE_LIST[2], edge_type=('user', 'note', 'pos_share_note'),
              decoder=gl.Decoder(weighted=True, labeled=True), directed=False)       
    return g, decoder_dict


def get_total_user_num():
    total_user_num = 0
    train_data_base_dir = FLAGS.train_data_base_dir
    user_node = FLAGS.train_data_node_dir.split(',')[0]
    assert 'user' in user_node,'error in get_total_user_num, user node must be the first node'
    cur_dir = os.path.join(train_data_base_dir, user_node)
    filenames = os.listdir(cur_dir)
    if '_SUCCESS' in filenames:
        filenames.remove('_SUCCESS')
    while '.ipynb_checkpoints' in filenames:
        filenames.remove('.ipynb_checkpoints')
    filenames.sort()
    for filename in filenames:
        one_part_file = os.path.join(cur_dir, filename)
        one_part_total_user = sum(1 for line in open(one_part_file))-1
        total_user_num += one_part_total_user
    return total_user_num


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
    tfg.conf.emb_max_partitions = len(tf_cluster.get("ps"))
    print('====task_index is====', task_index)
    print('job_name is', job_name, 'tf_cluster is', tf_cluster, 'gl_cluster is', gl_cluster)
    tf_cluster = tf.train.ClusterSpec(tf_cluster)
    gl.set_default_string_attribute("None")
    gl.set_padding_mode(gl.CIRCULAR)

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
            # print('===save_model_dir, mkdir_status is',mkdir_status,'===mkdir_output is',mkdir_output)

            if os.path.exists(FLAGS.save_emb_dir):
                shutil.rmtree(FLAGS.save_emb_dir)
            mkdir_status, mkdir_output = commands.getstatusoutput("mkdir -p {}".format(FLAGS.save_emb_dir))
            # print('=== save_emb_dir, mkdir_status is',mkdir_status,'===mkdir_output is',mkdir_output)

            os.system('touch {}'.format(SYNC_DONE))
        else:
            for i in range(200):
                print('=on gl, running on i: ', i)
                if os.path.isfile(SYNC_DONE):
                    break
                else:
                    print("waiting ckpt model dir ... {}".format(SYNC_DONE))
                    time.sleep(5)

        # load graph
        g, decoder_dict = init_graph(TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST)
        g.init(cluster=gl_cluster, job_name="server", task_index=task_index)
        print("GraphLearn Server start...")
        g.wait_for_close()


    elif job_name == "ps":
        for i in range(200):
            if os.path.isfile(SYNC_DONE):
                break
            else:
                print("waiting ckpt model dir ... {}".format(SYNC_DONE))
                time.sleep(5)
        # training data
        TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST = get_node_edge_path(task_index)
        print('TRAINING_DATA_NODE_LIST is', TRAINING_DATA_NODE_LIST, '\n', 'TRAINING_DATA_EDGE_LIST is',
              TRAINING_DATA_EDGE_LIST)

        trainer = DistTrainer(tf_cluster, job_name, task_index, tf_cluster.num_tasks("worker"),
                              ckpt_dir=FLAGS.save_model_dir, profiling=False)
        print("TF PS start...")
        trainer.join()
    else:
        for i in range(200):
            if os.path.isfile(SYNC_DONE):
                break
            else:
                print("waiting ckpt model dir ... {}".format(SYNC_DONE))
                time.sleep(5)

        # training data
        TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST = get_node_edge_path(task_index)
        print('TRAINING_DATA_NODE_LIST is', TRAINING_DATA_NODE_LIST, '\n', 'TRAINING_DATA_EDGE_LIST is',
              TRAINING_DATA_EDGE_LIST)
        total_user_num = get_total_user_num()
        total_global_step = FLAGS.train_epochs*int(total_user_num/FLAGS.batch_size)
        print('===total user num and total global step is: ',total_user_num, total_global_step)


        g, decoder_dict = init_graph(TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST)
        g.init(cluster=gl_cluster, job_name="client", task_index=task_index)

        trainer = DistTrainer(tf_cluster, job_name, task_index, tf_cluster.num_tasks("worker"),
                              ckpt_dir=FLAGS.save_model_dir,
                              profiling=False)
        trainer.ckpt_freq = FLAGS.save_checkpoints_cycle

        with trainer.context():
        # define handler for node feature transformation
            user_feature_handler = tfg.FeatureHandler('user_feature_handler', decoder_dict['user_decoder'].feature_spec)
            author_feature_handler = tfg.FeatureHandler('author_feature_handler',
                                                        decoder_dict['author_decoder'].feature_spec)
            note_feature_handler = tfg.FeatureHandler('note_feature_handler', decoder_dict['note_decoder'].feature_spec)

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
            nbrs_num = json.loads(FLAGS.nbrs_num)  # string of list, neighbor num of each hop'
            num_layers = len(nbrs_num)

            # define HGT model
            model = EgoHGT(FLAGS.hidden_dim,
                           num_layers,
                           pos_relation_dict,
                           input_dim_dict,
                           n_heads=FLAGS.n_heads,
                           dropout=FLAGS.drop_out,
                           use_norm=FLAGS.use_norm)
            
            # define SOM for maintaining cluster_center_emb
            SOM_model = SOM(FLAGS.batch_size, FLAGS.max_cluster_num, FLAGS.hidden_dim)
            
            hgt_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for param in hgt_params:
                if 'cluster_center' in param.name:
                    hgt_params.remove(param)
                    print('done!remove',param.name)
            print('===hgt_params is',hgt_params)


            user_dataset = NodeDataLoader(g, gl.Mask.TRAIN, FLAGS.sampler, FLAGS.batch_size,
                                                src_node_type='user', pos_relation_dict=pos_relation_dict,
                                                user_feature_handler=user_feature_handler,
                                                author_feature_handler=author_feature_handler,
                                                note_feature_handler=note_feature_handler,
                                                nbrs_num=nbrs_num)
            iterator = user_dataset._iterator
            user_x_list = user_dataset.x_list(user_dataset._q, user_dataset._data_dict, user_dataset._alias_list,
                                                user_dataset._node_type_path)
            user_embedding = model.forward(user_x_list, user_dataset._relation_path, user_dataset._node_type_path,
                                            user_dataset._relation_count, nbrs_num)
            user_embedding = tf.nn.l2_normalize(user_embedding, axis=-1)  # normalize will make SOM clustering more stable


            # get decay cluster_learning_rate, decay_radius for updating SOM model
            global_step  = tf.train.get_or_create_global_step()
            decay_rate = 1
            decay_learning_rate = tf.train.inverse_time_decay(FLAGS.init_cluster_lr, global_step, total_global_step, decay_rate) 
            decay_radius = tf.train.inverse_time_decay(FLAGS.init_cluster_radius, global_step, total_global_step, decay_rate)

            # SOM forward to update cluster_center_emb
            cluster_center_emb, winner_embedding, cluster_norm = SOM_model.forward(user_embedding, decay_learning_rate, decay_radius)
            
            # use cluster loss to update HGT parameters
            cluster_loss = tf.reduce_sum((user_embedding - winner_embedding)**2, axis=-1)  # [batch_size, hidden_dim] -> [batch_size]
            cluster_loss = tf.reduce_sum(cluster_loss) # scalar

            # operations to update cluster_center_emb in SOM
            cluster_ops = [cluster_center_emb, decay_learning_rate, decay_radius, cluster_norm]


            # # create path for saving node embedding
            user_emb_path = os.path.join(FLAGS.save_emb_dir, 'user_emb_part{}.txt'.format(task_index))
            for path in [user_emb_path]:
                if os.path.exists(path):
                    os.remove(path)
            user_ids = user_dataset._data_dict['user'].ids   # we must first access all user ids and total user num and remap it into [0,1,...,total_user_num-1]


        ### train model
        print('===start trainer.train and total epochs is', FLAGS.train_epochs)
        trainer.train(iterator, cluster_loss, learning_rate=FLAGS.learning_rate, epochs=FLAGS.train_epochs,
                      metrics=[],var_list =hgt_params, global_step=global_step, cluster_ops = cluster_ops)

        # print('==start saving user embedding==')
        trainer.save_node_embedding(user_emb_path, iterator, user_ids, user_embedding, FLAGS.batch_size)

        trainer.close()
        g.close()
        print("graph closed")


def define_custom_flags():
    # for tfg.config
    flags.DEFINE_integer(name='emb_max_partitions', default=12,
                         help='The max_partitions for embedding variables for embedding variables partitioned by min_max_variable_partitioner')
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
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate for updating HGT parameters')
    flags.DEFINE_float('drop_out', 0.5, 'drop out rate')
    flags.DEFINE_integer('hidden_dim', 64, 'hidden layer dim')
    flags.DEFINE_integer('n_heads', 8, 'number of relations')
    flags.DEFINE_boolean('use_norm', True, 'use norm for hgt aggregation')
    flags.DEFINE_string('nbrs_num', '[2,2]', 'string of list, neighbor num of each hop')
    flags.DEFINE_string('sampler', 'random', 'neighbor sampler strategy. random, in_degree, topk.')
    flags.DEFINE_integer('neg_num', 2, 'number of negative samples for each src/dst node in pos relation')
    flags.DEFINE_integer('max_cluster_num', 10, 'max number of clusters, as some clusters may be empty')
    flags.DEFINE_float('init_cluster_lr', 0.1, 'initial learning rate for updating cluster center embedding')
    flags.DEFINE_float('init_cluster_radius', 0.5, 'initial radius in neighborhood function')

    # for load and save model
    flags.DEFINE_string(name='load_model_dir', default="",
                        help='Model dir for load before training')  # /root/data/hgt/model/20230722/
    flags.DEFINE_string(name='save_model_dir', default=None, help='Model dir for save after training')
    flags.DEFINE_integer(name='save_checkpoints_cycle', default=36000, help='How many seconds to save checkpoint')
    flags.DEFINE_string(name='save_emb_dir', default=None, help='Save emb dir')
    flags.DEFINE_string(name='config_path', default='hgt_config_with_bucket.json', help='config path') #hgt_config_with_bucket.json
    flags.DEFINE_integer(name='save_emb_num', default=None,
                         help='Save emb file num, generally comes with node type num * worker num')


def print_configuration_op(FLAGS):
    print('TF_FLAGS:')
    # pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        value = value.value
        if type(value) == float:
            print('flags %s:\t %f' % (name, value))
        elif type(value) == int:
            print('flags %s:\t %d' % (name, value))
        elif type(value) == str:
            print('flags %s:\t %s' % (name, value))
        elif type(value) == bool:
            print('flags %s:\t %s' % (name, value))
        else:
            print(' %s:\t %s' % (name, value))
    # for k, v in sorted(FLAGS.__dict__.items()):
    # print(f'{k}={v}\n')
    print('End of configuration')


if __name__ == "__main__":
    print("tf version:{}".format(tf.__version__))
    print("gl version:{}".format(gl.__git_version__))
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    define_custom_flags()

    print('====flags====')
    print_configuration_op(FLAGS)
    print('=save_emb_dir: ',FLAGS.save_emb_dir)
    print('=save_model_dir: ',FLAGS.save_model_dir)
    print('=save_emb_num: ',FLAGS.save_emb_num)

    assert FLAGS.save_emb_dir is not None
    assert FLAGS.save_model_dir is not None
    assert FLAGS.save_emb_num is not None

    tfg.conf.emb_min_slice_size = FLAGS.emb_min_slice_size
    tfg.conf.emb_live_steps = FLAGS.emb_live_steps

    COSINE_SCALE = 2.0

    pos_relation_dict = OrderedDict()
    pos_relation_dict['pos_follow_note'] = ['user', 'note']
    pos_relation_dict['pos_comment_note'] = ['user', 'note']
    pos_relation_dict['pos_share_note'] = ['user', 'note']

    

    train()
