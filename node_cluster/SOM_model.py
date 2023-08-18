# SOM model for updating cluster_center_embedding

import tensorflow as tf

class SOM(object):
    def __init__(self ,batch_size, max_cluster_num,hidden_dim):
        self.batch_size = batch_size
        self.max_cluster_num = max_cluster_num
        self.hidden_dim = hidden_dim
        with tf.variable_scope('cluster_center', reuse=tf.AUTO_REUSE):
            self.cluster_center_emb = tf.get_variable(name="_emb", shape=[max_cluster_num, hidden_dim],initializer=tf.glorot_uniform_initializer())

    def forward(self, user_embedding, decay_learning_rate, decay_radius):
        # get winner cluster  for each user in a batch
        tmp_user_embedding = tf.expand_dims(user_embedding, axis=1)  # [batch_size, 1, hidden_dim]
        tmp_user_embedding = tf.tile(tmp_user_embedding, [1,self.max_cluster_num, 1])  # [batch_size, max_cluster_num, hidden_dim]
        tmp_cluster_center_emb = tf.expand_dims(self.cluster_center_emb, axis=0)  # [1, max_cluster_num, hidden_dim]
        tmp_cluster_center_emb = tf.tile(tmp_cluster_center_emb, [self.batch_size, 1, 1])  # [batch_size, max_cluster_num, hidden_dim]
        distance = tf.reduce_sum((tmp_user_embedding - tmp_cluster_center_emb)**2, axis=-1)  # [batch_size, max_cluster_num]
        winner_cluster_center = tf.arg_min(distance, -1)  # batch_size

        # update winner cluster center embedding
        cluster_learning_rate = decay_learning_rate  # learning rate for updating cluster emb, decay with global step
        winner_embedding = tf.gather(self.cluster_center_emb, winner_cluster_center)  # [batch_size , hidden_dim]
        winner_gradient = cluster_learning_rate*(user_embedding - winner_embedding)  # [batch_size , hidden_dim]
        winner_cluster_center = tf.expand_dims(winner_cluster_center, axis=-1)  # [batch_size, 1]         
        self.cluster_center_emb = tf.scatter_nd_add(self.cluster_center_emb, winner_cluster_center, winner_gradient)  # [max_cluster_num , hidden_dim]


        # get importance of each cluster with repect to winner cluster
        tmp_winner_embedding = tf.expand_dims(winner_embedding, axis=1)  # [batch_size , 1, hidden_dim]
        tmp_winner_embedding = tf.tile(tmp_winner_embedding, [1,self.max_cluster_num, 1])  # [batch_size, max_cluster_num, hidden_dim]

        radius = decay_radius # radius for neighbor cluster center, decay with learning step
        euc_distance = tf.reduce_sum((tmp_winner_embedding - tmp_cluster_center_emb)**2, axis=-1)  # [batch_size, max_cluster_num]
        neigh_importance = tf.exp(-euc_distance/(2*radius**2))  # [batch_size, max_cluster_num],  using gaussian neighbor function
        neigh_importance = tf.expand_dims(neigh_importance, axis=-1)  # [batch_size, max_cluster_num, 1]
        neigh_importance = tf.tile(neigh_importance,[1,1,self.hidden_dim])  # [batch_size, max_cluster_num, hidden_dim]

        # update neighbor cluster center embedding except winner cluster center
        cluster_center_gradient = cluster_learning_rate*(tmp_winner_embedding - tmp_cluster_center_emb)*neigh_importance  # [batch_size, max_cluster_num, hidden_dim]
        cluster_center_gradient = tf.reduce_sum(cluster_center_gradient, axis=0)  # [max_cluster_num, hidden_dim]
        self.cluster_center_emb = self.cluster_center_emb + cluster_center_gradient  # [max_cluster_num, hidden_dim]

        cluster_norm = tf.norm(self.cluster_center_emb)
        return self.cluster_center_emb, winner_embedding, cluster_norm