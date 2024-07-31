import numpy as np
import torch.nn.functional as F
import torch
from torch import nn

from model.base_model import GNN_encoder


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
        self.activation = activation

    def forward(self, x, edge_index, num_nodes):
        adj_tensor = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
        adj_tensor[edge_index[0], edge_index[1]] = 1
        adj_tensor[edge_index[1], edge_index[0]] = 1

        assert torch.isfinite(self.weight).all(), "Weight has NaNs or Infs"

        x = torch.mm(x, self.weight)
        x = torch.mm(adj_tensor, x)
        outputs = self.activation(x)
        return outputs

    def glorot_init(self, dim1, dim2):
        init_range = np.sqrt(6.0 / (dim1 + dim2))
        initial = torch.rand(dim1, dim2) * 2 * init_range - init_range
        return nn.Parameter(initial)



class Motify_Classifier(torch.nn.Module):

    def __init__(self, atom_feature_size=None, bond_feature_size=None, num_layer=5, emb_dim=300, residual=False, drop_ratio=0.5, JK="last"):

        super(Motify_Classifier, self).__init__()

        self.log_std = None
        self.mean = None
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.motif_encoder = GNN_encoder(self.num_layer, atom_feature_size, bond_feature_size, self.emb_dim, JK=self.JK,
                                         drop_ratio=self.drop_ratio,
                                         residual=residual)

        self.gcn_mean = GraphConvSparse(self.emb_dim, self.emb_dim, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(self.emb_dim, self.emb_dim, activation=lambda x: x)

    def forward(self, batched_data):
        h_motif = self.motif_encoder(batched_data)
        self.mean = self.gcn_mean(h_motif, batched_data.edge_index, batched_data.num_nodes)
        self.log_std = self.gcn_logstddev(h_motif, batched_data.edge_index, batched_data.num_nodes)


        gaussian_noise = torch.randn(batched_data.x.size(0), self.emb_dim)

        sampled_z = gaussian_noise * torch.exp(self.log_std) + self.mean

        a_pred = torch.sigmoid(torch.matmul(sampled_z, sampled_z.t()))

        return a_pred

    def motif_encoder_load(self, encoder):
        self.motif_encoder = encoder
