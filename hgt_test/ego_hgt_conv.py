from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import graphlearn.python.nn.tf as tfg
# from graphlearn.python.nn.tf.config import conf
from ego_layer import EgoConv


class EgoHGTConv(EgoConv):
    def __init__(self, name, in_dim, out_dim, relation_dict, input_dim_dict, n_heads, dropout=0.2, use_norm=True,
                 **kwargs):
        super(EgoHGTConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relation_dict = relation_dict
        self.input_dim_dict = input_dim_dict
        self.node_type_list = list(input_dim_dict.keys())  # ['user', 'author','note']
        self.relation_list = list(relation_dict.keys())
        self.num_relations = len(self.relation_list)  # number of relations
        self.num_types = len(self.node_type_list)  # number of node types

        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.dropout = dropout
        self.use_norm = use_norm

        self.k_linears = []
        self.q_linears = []
        self.v_linears = []
        self.a_linears = []
        self.norms = []
        self.src_type_idx = -1

        with tf.variable_scope('ego_hgt_' + name, reuse=tf.AUTO_REUSE):
            # for each node type
            for node_type in input_dim_dict.keys():
                self.k_linears.append(tfg.LinearLayer("k_linear_" + node_type, in_dim, out_dim, True)),
                self.q_linears.append(tfg.LinearLayer("q_linear_" + node_type, in_dim, out_dim, True))
                self.v_linears.append(tfg.LinearLayer("v_linear_" + node_type, in_dim, out_dim, True))
                self.a_linears.append(tfg.LinearLayer("a_linear_" + node_type, out_dim, out_dim, True))

                if use_norm:
                    norm_weight = tf.get_variable(name="norm_weight_" + node_type, shape=[out_dim, ],
                                                  initializer=tf.initializers.ones())
                    norm_bias = tf.get_variable(name="norm_bias_" + node_type, shape=[out_dim, ],
                                                initializer=tf.initializers.zeros())
                    self.norms.append([norm_weight, norm_bias])  # aka. gamma and beta

            # for each relation type
            # TODO: why relation pri is related to n_heads, should be specific to relation
            self.relation_pri = tf.get_variable(name='relation_pri', shape=[self.num_relations, self.n_heads],
                                                initializer=tf.initializers.ones())
            self.relation_att = tf.get_variable(name='reltion_att',
                                                shape=[self.num_relations, n_heads, self.d_k, self.d_k],
                                                initializer=tf.glorot_uniform_initializer())
            self.relation_msg = tf.get_variable(name='relation_msg',
                                                shape=[self.num_relations, n_heads, self.d_k, self.d_k],
                                                initializer=tf.glorot_uniform_initializer())
            self.skip = tf.get_variable(name='skip', shape=[self.num_types, ], initializer=tf.initializers.ones())

        # print('type and len of self.norm', type(self.norms), len(self.norms), type(self.norms[0])) 

    # as not all nodes will have num_relations out edges, for example use has two out edge types and note, author has only one out edge type
    # so we should change the code here
    def forward(self, x, neighbor, relation, node_type, expand):
        """
            Args:
            x: A float tensor with shape = [batch_size, input_dim].
            neighbor: A list of num_relation tensors, each tensor shape: [batch_size * expand, input_dim].
            relation: A list of str/int of relation type .
            expand: An integer, the neighbor count.
        """
        num_relations = len(
            relation)  # different node have different neighbor relation types, for example user has two relation types and user and author has one
        # in the forward function, we must use num_relations not self.num_relations, as the num_relations is different for different node types 
        k_mat = []  # k_mat: list of tensor with size num_relations, each tensor shape:[batch_size* expand, n_heads, d_k]
        v_mat = []  # v_mat: list of tensor with size num_relations, each tensor shape:[batch_size* expand, n_heads, d_k]
        relation_pri_mat = []  # relation_pri_mat: [num_relations, n_heads]

        # traverse neighbors of all relations
        for i in range(num_relations):
            rel = relation[i][-1]  # only care last relation connect to dst
            src_type = node_type[i][-2]
            dst_type = node_type[i][-1]
            # print('===prepare conv for relation: ',rel,'  src_type is',src_type,'  dst_type is:',dst_type)
            relation_type_idx = self.relation_list.index(rel)
            src_type_idx = self.node_type_list.index(src_type)
            self.src_type_idx = src_type_idx
            dst_type_idx = self.node_type_list.index(dst_type)

            k = self.k_linears[dst_type_idx](neighbor[i])  # [batch_size * expand, out_dim]
            v = self.v_linears[dst_type_idx](neighbor[i])  # [batch_size * expand, out_dim]

            k = tf.reshape(k, [-1, self.n_heads, self.d_k])  # [batch_size* expand,  n_heads, d_k]
            v = tf.reshape(v, [-1, self.n_heads, self.d_k])  # [batch_size* expand,  n_heads, d_k]

            relation_att = self.relation_att[relation_type_idx]  # [n_heads, d_k, d_k]
            relation_pri = self.relation_pri[relation_type_idx]  # [n_heads,]
            relation_pri_mat.append(
                relation_pri)  # relation_pri_mat: a list of tensor size=num_relation, each tensor shape [n_heads]
            relation_msg = self.relation_msg[relation_type_idx]  # [n_heads, d_k, d_k]

            k = tf.matmul(tf.transpose(k, [1, 0, 2]), relation_att)  # [n_heads,batch_size* expand, d_k]
            k = tf.transpose(k, [1, 0, 2])  # [batch_size* expand, n_heads, d_k]
            k_mat.append(k)

            v = tf.matmul(tf.transpose(v, [1, 0, 2]), relation_msg)  # [n_heads,batch_size* expand, d_k]
            v = tf.transpose(v, [1, 0, 2])  # [batch_size*expand, n_heads, d_k]
            v_mat.append(v)

            # print('====self.src_type_idx is', self.src_type_idx)
        assert self.src_type_idx != -1, 'src_type_idx should not be -1'
        q = self.q_linears[self.src_type_idx](x)  # [batch_size, out_dim], q is shared by all relations
        q = tf.reshape(q, [-1, self.n_heads, self.d_k])  # [batch_size, n_heads, d_k]
        q_mat = tf.expand_dims(q, axis=1)  # [batch_size, 1, n_heads, d_k]

        k_mat = tf.concat(k_mat, axis=0)  # [batch_size*expand*num_relations, n_heads, d_k]
        # print('===k_mat shape after concat',k_mat.shape)
        k_mat = tf.reshape(k_mat, [-1, expand * num_relations, self.n_heads,
                                   self.d_k])  # [batch_size, expand*num_relations, n_heads, d_k]

        # # Step 1: Heterogeneous Multi-head Attention
        relation_pri_mat = tf.concat(relation_pri_mat, axis=0)  # [n_heads*num_relations,]
        relation_pri_mat = tf.reshape(relation_pri_mat, [-1, num_relations])  # [n_heads, num_relations]
        # print('=====pri_mat shape after reshape',relation_pri_mat.shape)
        relation_pri_mat = tf.transpose(relation_pri_mat, [1, 0])  # [num_relations, n_heads]
        relation_pri_mat = tf.expand_dims(tf.tile(relation_pri_mat, [expand, 1]),
                                          0)  # [1, expand*num_relations, n_heads]

        res_att = tf.reduce_sum(q_mat * k_mat, -1) * relation_pri_mat
        res_att = res_att / self.sqrt_dk  # [batch_size, expand*num_relations, n_heads]
        res_att = tf.nn.softmax(res_att,
                                axis=1)  # [batch_size, expand*num_relations, n_heads], the weight of each neighbor at each head
        res_att = tf.reshape(res_att, [-1, expand * num_relations, self.n_heads,
                                       1])  # [batch_size, expand*num_relations, n_heads, 1]

        # Step 2: Heterogeneous Message Passing
        v_mat = tf.concat(v_mat, axis=0)  # [batch_size* expand* num_relations, n_heads, d_k]
        # print('===v_mat shape after concat',v_mat.shape)
        v_mat = tf.reshape(v_mat, [-1, expand * num_relations, self.n_heads,
                                   self.d_k])  # [batch_size, expand*num_relations, n_heads, d_k]

        agg = tf.reduce_sum(res_att * v_mat, axis=1)  # [batch_size, n_heads, d_k]
        agg = tf.reshape(agg, [-1, self.n_heads * self.d_k])  # [batch_size, out_dim]

        '''
        # Step 3: Target-specific Aggregation
        x = norm( W[node_type] * gelu( Agg(x) ) + x ), Agg(x) is agg_res
        in HGT_layer in_dim=out_dim=hidden_dim, so that we can use skip connection
        '''
        trans_out = self.a_linears[src_type_idx](agg)  # [batch_size, out_dim]
        if self.dropout and tfg.conf.training:
            trans_out = tf.nn.dropout(trans_out, keep_prob=1 - self.dropout)  # keep_prob = 1-dropout_rate

        alpha = tf.sigmoid(self.skip[src_type_idx])  # one-dim tensor 
        res = trans_out * alpha + x * (1 - alpha)  # [batch_size, out_dim], here we must ensure in_dim=out_dim
        if self.use_norm:
            eps = 1e-6
            norm_weight = self.norms[src_type_idx][0]  # gamma
            norm_bias = self.norms[src_type_idx][1]  # beta for layer normalization
            mean, variance = tf.nn.moments(res, axes=[-1], keep_dims=True)
            normalized = (res - mean) / tf.sqrt(variance + eps)
            res = norm_weight * normalized + norm_bias
        return res
