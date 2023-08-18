from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import graphlearn.python.nn.tf as tfg
from graphlearn.python.nn.tf.layers.linear_layer import LinearLayer
from ego_hgt_conv import EgoHGTConv


class EgoHGT(tfg.Module):
  def __init__(self,
               input_dim,
               hidden_dim,
               out_dim,
               num_layers,
               relation_dict,   # {relation_type: [src_type,dst_type]}; src is self, dst is neighbor
               n_heads,
               dropout=0.2,
               use_norm = True,
               **kwargs):
    """EgoGraph based HGT. 
  
    Args:
      input_dim: input dimension of nodes.
      out_dim: output dimension of nodes.
      relation_dict: store all relations and its source and target node type {relation_type:[src_type,dst_type]}.
      
      dropout: Dropout rate for hidden layers' output. Default is 0.0, which
        means dropout will not be performed. The optional value is a float.
    """
    node_type_set = set()
    self.relation_dict =  relation_dict
    for src, dst in relation_dict.values():
        node_type_set.add(src)
        node_type_set.add(dst)
    self.node_type_list = sorted(list(node_type_set))
    relation_list = []
    for rel in relation_dict.keys():
      relation_list.append(rel)
    self.relation_list = relation_list
    print('relation dict: ', self.relation_dict)
    print('relation list: ', self.relation_list)
    print('node type list: ', self.node_type_list)
    
    
    self.num_relations = len(self.relation_list)
    self.num_types = len(self.node_type_list)
    self.dropout = dropout
    self.layers = []
    self.pre_linears = []
    self.classification_linear = LinearLayer("classification_linear", 
                                             hidden_dim, out_dim, use_bias=True)

    for i in range(self.num_types):
        self.pre_linears.append(LinearLayer("pre_linear_node_type_" + str(i),
                                            input_dim, hidden_dim, use_bias=True))
      
    for i in range(num_layers):
      conv = EgoHGTConv("layer_" + str(i),
                              hidden_dim,
                              hidden_dim,
                              self.relation_dict,
                              self.relation_list,
                              self.node_type_list,
                              n_heads,
                              dropout=dropout,
                              use_norm = use_norm)
      self.layers.append(conv)


  def forward(self, x_list, x_relation_list, expands):
    """ Update node embeddings.
    Args:
      x_list: A list of list, representing input nodes and their K-hop neighbors.
        The first element x_list[0] is a list with one element which means root node tensor 
        with shape`[n, input_dim]`. n is batch size(bs)
        The following element x_list[i] (i > 0) is i-th hop neighbors list with legth num_relations^i.
        It consists of different types of neighbors, and each element of x_list[i] is a tensor with 
        shape `[n * k_1 * ... * k_i, input_dim]`, where `k_i` means the neighbor count of each node 
        at i-th hop of root node. 

        Note that the elements of each list must be stored in the same order when stored 
        by relation type. For example, their are 2 relations and 2-hop neighbors, the x_list is stored
        in the following format:
        [[root_node], 
         [hop1_r0, hop1_r1], 
         [hop1_r0_hop2_r0, hop1_r0_hop2_r1, hop1_r1_hop2_r0, hop1_r1_hop2_r1]]
      
      x_relation: A list of list, representing the relation type of each hop neighbors.  For example, their are 2 relations and 2-hop neighbors,
      x_relation should store in the following format:
        [
        [[src_type]],
        [[src_type,r0],[src_type,r1]],
        [[src_type,r0,r0],[src_type,r0,r1],[src_type,r1,r0], [src_type,r1,r1]] 
        ]

      expands: An integer list of neighbor count at each hop. For the above
        x_list, expands = [k_1, k_2, ... , k_K]

    Returns:
      A tensor with shape `[n, output_dim]`.
    """
    depth = len(expands)
    assert depth == (len(x_list) - 1)
    assert depth == len(self.layers)
    assert depth >=1
    
    # use pre_linear to transform the input dim from input_dim to hidden_dim
    H = []
    src_type = x_relation_list[0][0][0]
    src_type_idx = self.node_type_list.index(src_type)
    H.append([self.pre_linears[src_type_idx](x_list[0][0])])
    for i in range(1,depth+1):
      h = []
      x_list_hop_i = x_list[i]
      x_relation_list_hop_i = x_relation_list[i]
      for j in range(len(x_list_hop_i)):
        rel = x_relation_list_hop_i[j][-1]  # we only care about dst node type
        dst_type = self.relation_dict[rel][1]
        dst_type_idx = self.node_type_list.index(dst_type)
        h.append(self.pre_linears[dst_type_idx](x_list_hop_i[j]))
      H.append(h)
      
    # covolution using HGT layer
    for layer_idx in range(len(self.layers)): # for each conv layers.
      tmp_vecs = []
      num_root = 1 # the number of root node at each hop.
      for hop in range(depth - layer_idx): # for each hop neighbor, h[i+1]->h[i]
        tmp_nbr_vecs = []
        for offset in range(num_root): # do h[i+1]->h[i] according different relations.
          src_vecs = H[hop][offset] # hop=0, bs*input_dim ; hop=1, bs*k_1*input_dim, ...
          neigh_relations = x_relation_list[hop+1][(offset*self.num_relations) : ((offset+1)*self.num_relations)]
          neigh_vecs = H[hop+1][(offset*self.num_relations) : ((offset+1)*self.num_relations)] 
          print(f'using hgt layer {layer_idx}:,hop={hop}, offset={offset}')  # check the convolution operation
          print('neigh_relations', neigh_relations)
          # neigh_vecse:list of tensor with shape: hop=0, (bs*k_1)*input_dim; hop=1, (bs*k_1*k_2)*input_dim, ...
          h = self.layers[layer_idx].forward(src_vecs, neigh_vecs, neigh_relations, expands[hop])   # n*input_dim -> n*output_dim
          tmp_nbr_vecs.append(h)
        num_root *= self.num_relations # the root node of the next hop is expand by num_relations.
        tmp_vecs.append(tmp_nbr_vecs)
      H = tmp_vecs

      # classification layer
      out = self.classification_linear(H[0][0])
    return out