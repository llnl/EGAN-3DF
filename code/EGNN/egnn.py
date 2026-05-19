################################################################################
# Copyright 2024-2025 Lawrence Livermore National Security, LLC and other
# Heesung Shim (shim2@llnl.gov)
# See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# EGNN model architecture used for training.
################################################################################

from torch import nn
import torch
from torch_geometric.nn import global_add_pool

class E_GCL(nn.Module):
   
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):

        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN_MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EGNN_MLP, self).__init__()

        print("in_channels/out_channels: ", in_channels, out_channels)
        self.output = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / 1.5)),
            nn.ReLU(),
            nn.Linear(int(in_channels / 1.5), int(in_channels / 2)),
            nn.ReLU(),
            nn.Linear(int(in_channels / 2), out_channels),
        )

    def forward(self, data, return_hidden_feature=False):
        if return_hidden_feature:
            # changed the order (pred, feat1, feat2) to be compatible with PCN and other models
            return self.output(data), self.output[:-2](data), self.output[:-4](data)
        else:
            return self.output(data)


class EGNN(nn.Module):
    def __init__(
                self, 
                in_channels: int,
                out_channels: int,
                in_edge_nf=1,
                distance_cutoff=1.5,
                act_fn=nn.SiLU(), 
                n_layers=4,
                residual=True, 
                attention=True, 
                normalize=False, 
                tanh=False,
                get_feature_only=False):
       
        super(EGNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = 20
        self.hidden_nf = 20
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(self.in_channels, self.hidden_channels)
        self.embedding_out = nn.Linear(self.hidden_nf, self.hidden_channels)
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.pool = global_add_pool
        self.relu = nn.ReLU()
        self.mlp = EGNN_MLP(self.hidden_channels, self.out_channels)
        self.get_feature_only = get_feature_only

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_channels, self.hidden_nf, self.hidden_nf, edges_in_d=0,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
            self.add_module("tanh_%d" % i, nn.Tanh())
            self.add_module("tanh_c_%d" % i, nn.Tanh())
            self.add_module("bn_%d" % i, nn.BatchNorm1d(self.hidden_nf))
            self.add_module("bn_c_%d" % i, nn.BatchNorm1d(3))

        self.pair_distance = nn.PairwiseDistance(p=2)
        self.distance_cutoff = distance_cutoff
        self.n_layers = n_layers
    
    def freeze(self):
        for i in range(0, self.n_layers):
            self._modules["gcl_%d" % i].requires_grad=False
            self._modules["tanh_%d" %i].requires_grad=False
            self._modules["bn_%d" %i].requires_grad=False
            self._modules["tanh_c_%d"%i].requires_grad=False
            self._modules["bn_c_%d" %i].requires_grad=False

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        out = self.embedding_in(x)
        coords = data.pos
        # compute distances
        distances = self.pair_distance(coords[edge_index[0]],
                                       coords[edge_index[1]]) + 1e-4
        edge_index = edge_index[:, distances < self.distance_cutoff]
        edge_attr = data.edge_attr[distances < self.distance_cutoff].view(-1,1)
        for i in range(0, self.n_layers):
            out, coords, _ = self._modules["gcl_%d" % i](out, edge_index, coords, edge_attr=edge_attr)
            out = self._modules["tanh_%d" %i](out)
            out = self._modules["bn_%d" %i](out)
            coords = self._modules["tanh_c_%d"%i](coords)
            coords = self._modules["bn_c_%d" %i](coords)
        
        out = self.embedding_out(out)
        # h = self.bn(h)
        
        out = self.pool(out, data.batch)
        if self.get_feature_only:
            return out
        else:
            return self.mlp(self.relu(out), True)


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)