try:
  # https://www.tensorflow.org/guide/migrate
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
except ImportError:
  import tensorflow as tf

import ego_data_loader as ego_data
import graphlearn as gl
import graphlearn.python.nn.tf as tfg

class EgoSAGEUnsupervisedDataLoader(ego_data.EgoDataLoader):
  def __init__(self, graph, mask=gl.Mask.TRAIN, sampler='random', neg_sampler='random',
               batch_size=128, window=10,
               node_type='i', edge_type='e', nbrs_num=None, neg_num=5):
    self._neg_sampler = neg_sampler
    self._node_type = node_type
    self._edge_type = edge_type
    self._nbrs_num = nbrs_num
    self._neg_num = neg_num
    super(EgoSAGEUnsupervisedDataLoader, self).__init__(graph, mask, sampler, batch_size, window)

  @property
  def src_ego(self):
    return self.get_egograph('src')

  @property
  def dst_ego(self):
    return self.get_egograph('dst')

  @property
  def neg_dst_ego(self):
    return self.get_egograph('neg_dst')

  def _query(self, graph):
    seed = graph.E(self._edge_type).batch(self._batch_size).shuffle(traverse=True)
    src = seed.outV().alias('src')
    dst = seed.inV().alias('dst')
    neg_dst = src.outNeg(self._edge_type).sample(self._neg_num).by(self._neg_sampler).alias('neg_dst')
    src_ego = self.meta_path_sample(src, 'src', self._nbrs_num, self._sampler, "out")
    dst_ego = self.meta_path_sample(dst, 'dst', self._nbrs_num, self._sampler, "out")
    dst_neg_ego = self.meta_path_sample(neg_dst, 'neg_dst', self._nbrs_num, self._sampler, "out")
    return seed.values()

  def meta_path_sample(self, ego, ego_name, nbrs_num, sampler, type):
    """ creates the meta-math sampler of the input ego.
    config:
      ego: A query object, the input centric nodes/edges
      ego_name: A string, the name of `ego`.
      nbrs_num: A list, the number of neighbors for each hop.
      sampler: A string, the strategy of neighbor sampling.
    """
    alias_list = [ego_name + '_hop_' + str(i + 1) for i in range(len(nbrs_num))]
    for nbr_count, alias in zip(nbrs_num, alias_list):
      if (type == "out"):
        ego = ego.outV(self._edge_type).sample(nbr_count).by(sampler).alias(alias)
      else:
        ego = ego.inV(self._edge_type).sample(nbr_count).by(sampler).alias(alias)
    return ego