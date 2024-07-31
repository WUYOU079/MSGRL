import torch
from ogb.graphproppred.mol_encoder import full_atom_feature_dims, BondEncoder, AtomEncoder
from torch_geometric.data.data import BaseData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

import torch.nn.functional as F


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearEncoder, self).__init__()
        self.linear = torch.nn.Linear(in_channel, out_channel)
        self.bn = torch.nn.BatchNorm1d(out_channel, eps=1e-06, momentum=0.1)
        self.act = act

    def forward(self, x):
        x = self.linear(x)
        size = x.size()
        x = x.view(-1, x.size()[-1], 1)
        x = self.bn(x)
        x = x.view(size)
        if self.act is not None:
            x = self.act(x)
        return x


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, bond_feature_size, emb_dim, encode_edge=True):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        if bond_feature_size:
            self.bond_encoder = LinearEncoder(bond_feature_size, emb_dim)
        else:
            self.bond_encoder = BondEncoder(emb_dim=emb_dim)
        self.encode_edge = encode_edge

        self.GRU_update = torch.nn.GRUCell(emb_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        if self.encode_edge and edge_attr is not None and edge_attr.numel() > 0:
            edge_embedding = self.bond_encoder(edge_attr)
        else:
            edge_embedding = edge_attr

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        if edge_attr.numel() == 0:
            return norm.view(-1, 1) * F.relu(x_j)
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out, x):
        aggr_out = self.GRU_update(aggr_out, x)
        # torch.nn.GRUCell(args.hid_dim, args.hid_dim)
        return aggr_out


# endregion

class GNN_encoder(torch.nn.Module):

    def __init__(self, num_layer, atom_feature_size, bond_feature_size, emb_dim, drop_ratio=0.5, JK="last",
                 residual=False):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_encoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        if atom_feature_size:
            self.atom_encoder = LinearEncoder(atom_feature_size, emb_dim)
        else:
            self.atom_encoder = AtomEncoder(emb_dim)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(GCNConv(bond_feature_size, emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        if isinstance(batched_data, BaseData):
            x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
        elif len(batched_data) == 3:
            x, edge_index, edge_attr = batched_data[0], batched_data[1], batched_data[2]

        temp_h = [self.atom_encoder(x)]
        h_list = temp_h
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        assert torch.isfinite(node_representation).all(), "node_representation has NaNs or Infs"

        return node_representation
