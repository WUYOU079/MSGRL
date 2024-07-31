import pandas as pd
import os.path as osp
import torch
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.io.read_graph_pyg import read_graph_pyg
from rdkit import Chem

import torch.utils.data

from Utils.motif_utils import motif_decomp, motifgraph_featconstruct

class PygGraphPropPredDataset_Smiles(PygGraphPropPredDataset):
    def __init__(self, name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''
        super(PygGraphPropPredDataset_Smiles, self).__init__(name, root, transform, pre_transform, meta_dict)

    @property
    def mapping_dir(self) -> str:
        return osp.join(self.root, 'mapping')

    def read_graph_smiles(self):
        try:
            smiles_file = pd.read_csv(osp.join(self.mapping_dir, 'mol.csv.gz'), compression='gzip', header=None,
                                      skiprows=1).values  # 去掉第一行title
        except FileNotFoundError:
            smiles_file = None
        smiles_list = smiles_file[:, self.num_tasks]
        return smiles_list

    def process(self):
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        data_list = read_graph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge,
                                   additional_node_files=additional_node_files,
                                   additional_edge_files=additional_edge_files, binary=self.binary)

        smiles_list = self.read_graph_smiles()
        for i, g in enumerate(data_list):
            if i >= len(smiles_list):
                pass
            smiles = smiles_list[i]
            mol = Chem.MolFromSmiles(smiles_list[i])
            if len(mol.GetAtoms()) != g.num_nodes:
                raise RuntimeError('Smiles file does not match the graph data')
            g.smiles = smiles
            g.mol = mol
            g.cliques = motif_decomp(g.mol)
            edge_index_motifgraph, edge_attr_motifgraph, edge_index_cliques, attr_cliques, num_edge_cliques, g_motifs, g_motif_graph = motifgraph_featconstruct(
                g.cliques, g.x, g.edge_index, g.edge_attr, g.edge_attr.device)
            g.g_motif_graph = g_motif_graph

            g.motif_list = g_motifs

            g.edge_index_motifgraph = edge_index_motifgraph
            g.edge_attr_motifgraph = edge_attr_motifgraph

            g.edge_index_cliques = edge_index_cliques
            g.attr_cliques = attr_cliques
            g.num_edge_cliques = num_edge_cliques

        if self.task_type == 'subtoken prediction':
            graph_label_notparsed = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip',
                                                header=None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip',
                                          header=None).values

            has_nan = np.isnan(graph_label).any()

            for i, g in enumerate(data_list):
                if 'classification' in self.task_type:
                    if has_nan:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)
                    else:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.long)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])
