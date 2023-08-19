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
from collections import OrderedDict

from ego_hgt_linkpred import EgoHGT
from link_dataloader import EgoHGTDataLoader
from local_trainer import LocalTrainer
from save_node_emb_dataloader import SaveNodeDataLoader
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
        .edge(TRAINING_DATA_EDGE_LIST[1], edge_type=('user', 'note', 'pos_share_note'),
              decoder=gl.Decoder(weighted=True, labeled=True), directed=False) \
        .edge(TRAINING_DATA_EDGE_LIST[0], edge_type=('user', 'note', 'label_edge'),
              decoder=gl.Decoder(weighted=True, labeled=True), directed=False)
    return g, decoder_dict



def train():
    gl.set_default_string_attribute("None")
    gl.set_padding_mode(gl.CIRCULAR)
 

    # training data
    TRAINING_DATA_NODE_LIST = []
    TRAINING_DATA_EDGE_LIST = []

    current_path = sys.path[0]
    local_fakedata_path = os.path.abspath(os.path.join(current_path,'..'))+'/data/local/'
    for node in ['fakeuser','fakenote']:
        TRAINING_DATA_NODE_LIST.append(local_fakedata_path+'{}/{}.csv'.format(node,node))
    for edge in ['fake_follow_note','fake_share_note']:
        TRAINING_DATA_EDGE_LIST.append(local_fakedata_path+'{}/{}.csv'.format(edge,edge))

    
    print('TRAINING_DATA_NODE_LIST is', TRAINING_DATA_NODE_LIST, '\n', 'TRAINING_DATA_EDGE_LIST is',
            TRAINING_DATA_EDGE_LIST)

    g, decoder_dict = init_graph(TRAINING_DATA_NODE_LIST, TRAINING_DATA_EDGE_LIST)

    trainer = LocalTrainer()

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

        # define model
        model = EgoHGT(FLAGS.hidden_dim,
                        num_layers,
                        pos_relation_dict,
                        input_dim_dict,
                        n_heads=FLAGS.n_heads,
                        dropout=FLAGS.drop_out,
                        use_norm=FLAGS.use_norm)

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

        if type(label_relation_dict.keys()) != list:
            label_edge = list(label_relation_dict.keys())[0]
        else:
            label_edge = label_relation_dict.keys()[0]
        # neg_relation = neg_relation_dict.keys()[i] , neg relation is generated by GSL, so we do not need it here
        print('===Start sampling for:', label_edge)

        train_data = EgoHGTDataLoader(g, gl.Mask.TRAIN, FLAGS.sampler, FLAGS.batch_size,
                                        label_relation_dict=label_relation_dict, neg_num=FLAGS.neg_num,
                                        pos_relation_dict=pos_relation_dict,
                                        user_feature_handler=user_feature_handler,
                                        author_feature_handler=author_feature_handler,
                                        note_feature_handler=note_feature_handler,
                                        nbrs_num=nbrs_num)
        iterator = train_data._iterator

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



    ### train model
    print('===start trainer.train and total epochs is', FLAGS.train_epochs)
    ## train with some epochs
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    trainer.train(iterator, loss, optimizer, epochs=FLAGS.epochs, metrics =[accuracy_train, update_op_train]) 
    trainer.train(iterator, loss, learning_rate=FLAGS.learning_rate, epochs=FLAGS.train_epochs,
                    metrics=[accuracy_train, update_op_train])


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
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_float('drop_out', 0.5, 'drop out rate')
    flags.DEFINE_integer('hidden_dim', 64, 'hidden layer dim')
    flags.DEFINE_integer('n_heads', 8, 'number of relations')
    flags.DEFINE_boolean('use_norm', True, 'use norm for hgt aggregation')
    flags.DEFINE_string('nbrs_num', '[4,2]', 'string of list, neighbor num of each hop')
    flags.DEFINE_string('sampler', 'random', 'neighbor sampler strategy. random, in_degree, topk.')
    flags.DEFINE_integer('neg_num', 2, 'number of negative samples for each src/dst node in pos relation')

    # for load and save model
    flags.DEFINE_string(name='load_model_dir', default="",
                        help='Model dir for load before training')  # /root/data/hgt/model/20230722/
    flags.DEFINE_string(name='save_model_dir', default=None, help='Model dir for save after training')
    flags.DEFINE_integer(name='save_checkpoints_cycle', default=36000, help='How many seconds to save checkpoint')
    flags.DEFINE_string(name='save_emb_dir', default=None, help='Save emb dir')
    flags.DEFINE_string(name='config_path', default='hgt_config_fake.json', help='config path') #hgt_config_with_bucket.json
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

    tfg.conf.emb_min_slice_size = FLAGS.emb_min_slice_size
    tfg.conf.emb_live_steps = FLAGS.emb_live_steps

    COSINE_SCALE = 2.0

    pos_relation_dict = OrderedDict()
    pos_relation_dict['pos_follow_note'] = ['user', 'note']
    pos_relation_dict['pos_share_note'] = ['user', 'note']

    label_relation_dict = OrderedDict()  # we only have one element in label_relation_dict to get one iterator
    label_relation_dict['label_edge'] = ['user', 'note']

    train()
