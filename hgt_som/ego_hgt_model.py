# ego_hgt model for link prediction
import tensorflow as tf
import graphlearn.python.nn.tf as tfg
from ego_hgt_conv import EgoHGTConv


class EgoHGT(tfg.Module):
    def __init__(self,
                 hidden_dim,
                 num_layers,
                 relation_dict,  
                 input_dim_dict,
                 n_heads,
                 dropout=0.2,
                 use_norm=True,
                 **kwargs):
        """EgoGraph based HGT for link prediction.

        Args:
          input_dim_dict: {'author': author_input_dim, 'note': note_input_dim, 'user': user_input_dim}
          out_dim: output dimension of nodes.
          relation_dict: store all relations and its source and target node type {relation_type:[src_type,dst_type]}.
        """
        self.node_type_list = list(input_dim_dict.keys())  # ['user', 'author','note']
        self.relation_list = list(relation_dict.keys())

        self.dropout = dropout
        self.layers = []
        self.pre_linears = []

        for node_type in self.node_type_list:
            input_dim = input_dim_dict[node_type]
            self.pre_linears.append(tfg.LinearLayer("pre_linear_" + node_type, input_dim, hidden_dim, use_bias=True))

        for i in range(num_layers):
            conv = EgoHGTConv("layer_" + str(i),
                              hidden_dim,
                              hidden_dim,
                              relation_dict,
                              input_dim_dict,
                              n_heads,
                              dropout=dropout,
                              use_norm=use_norm)
            self.layers.append(conv)


    def forward(self, x_list, x_relation_list, node_type_path, relation_count, expands):
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
            [[src_type,r0],[src_type,r1], [src_type,r2]],...,
            [[src_type,r0,r0],[src_type,r0,r1],[src_type,r1,r0], [src_type,r1,r1], [src_type, r2, r0]], ...,
            ]
          node_type_path:
          [
          [[user]],
          [[user, note], [user, note], [user, user]],
          [[user, note, user], [user,note,user], [user, note, user], [user, note, user], [user, user, note]]
          ]
          x_relation_count: the number of relation types that can hop for each step
          [[3],[2,2,1]]

          expands: An integer list of neighbor count at each hop. For the above
            x_list, expands = [k_1, k_2, ... , k_K]

        Returns:
          A tensor with shape `[n, output_dim]`.
        """
        depth = len(expands)
        assert depth == (len(x_list) - 1)
        assert depth == len(self.layers)
        assert depth >= 1

        # use pre_linear to transform the input dim from input_dim to hidden_dim
        H = []
        for i in range(0, depth + 1):
            h = []
            x_list_hop_i = x_list[i]
            x_node_type_hop_i = node_type_path[i]  # hop0 is self
            for j in range(len(x_list_hop_i)):
                dst_type = x_node_type_hop_i[j][-1]
                dst_type_idx = self.node_type_list.index(dst_type)
                h.append(self.pre_linears[dst_type_idx](x_list_hop_i[j]))
            H.append(h)

        # # # # covolution using HGT layer
        for layer_idx in range(len(self.layers)):  # for each conv layers.
            tmp_vecs = []
            num_root = 1  # the number of root node at each hop.

            for hop in range(depth - layer_idx):  # for each hop neighbor, h[i+1]->h[i]
                cur_hop_relation_count = relation_count[hop]
                tmp_nbr_vecs = []
                cursor = 0
                for offset in range(num_root):  # do h[i+1]->h[i] according different relations.
                    src_vecs = H[hop][offset]  # offset means the number of source node type at this hop
                    neigh_vecs = H[hop + 1][cursor: (cursor + cur_hop_relation_count[offset])]
                    neigh_relations = x_relation_list[hop + 1][cursor:(cursor + cur_hop_relation_count[offset])]
                    neigh_node_type = node_type_path[hop + 1][cursor:(cursor + cur_hop_relation_count[offset])]
                    # print('====start conv for ', 'layer', layer_idx , 'hop', hop, 'offset',offset)
                    # print('neigh_relation is', neigh_relations)
                    # print('neigh_node_type is', neigh_node_type)
                    h = self.layers[layer_idx].forward(src_vecs, neigh_vecs, neigh_relations, neigh_node_type,
                                                       expands[hop])  # n*input_dim -> n*output_dim
                    tmp_nbr_vecs.append(h)
                    cursor += cur_hop_relation_count[offset]
                num_root = sum(relation_count[layer_idx])
                tmp_vecs.append(tmp_nbr_vecs)
            H = tmp_vecs

            # for link prediction, we directly return hidden vectors
        return H[0][0]
