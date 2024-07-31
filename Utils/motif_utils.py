import torch
from rdkit.Chem import BRICS
from torch_geometric.data import Data


def flatten_2d_list_with_batch(input_list, device):
    flattened_list = []
    batch_tensor = torch.tensor([], device=device, dtype=torch.long)
    for i, sublist in enumerate(input_list):
        for item in sublist:
            flattened_list.append(item)
            batch_tensor = torch.cat((batch_tensor, torch.tensor([i], device=device, dtype=torch.long)))  # 将批次信息添加到集合中
    return flattened_list, batch_tensor


def motif_only_construct(cliques, x, edge_index, edge_attr, y, device):
    g_motifs = []
    for clique in cliques:
        g_motif = Data()
        g_motif.num_nodes = len(clique)
        g_motif.y = y
        edge_clique = [[], []]
        attr_clique = []
        for i, a_atom in enumerate(clique):
            for j, b_atom in enumerate(clique):
                if i == j:
                    continue
                else:
                    for i_edge, row_col in enumerate(zip(edge_index[0], edge_index[1])):
                        row, col = row_col
                        if a_atom == row and b_atom == col:
                            edge_clique[0].append(i)
                            edge_clique[1].append(j)
                            attr_clique.append(edge_attr[i_edge])
                            break

        g_motif.edge_index = torch.tensor(edge_clique).to(device=device, dtype=torch.long)
        if len(attr_clique) > 0:
            g_motif.edge_attr = torch.stack(attr_clique).to(device=device, dtype=torch.long)
        else:
            g_motif.edge_attr = torch.tensor(attr_clique).to(device=device, dtype=torch.long)
        g_motif.x = x[clique]
        g_motifs.append(g_motif)

    return g_motifs


def motifgraph_featconstruct(cliques, x, edge_index, edge_attr, device):
    edge_motif = [[], []]
    attr_motif = []
    for i_cliques in range(len(cliques)):
        for j_cliques in range(len(cliques)):
            if i_cliques == j_cliques:
                continue
            else:
                for a_atom in cliques[i_cliques]:
                    for b_atom in cliques[j_cliques]:
                        for i, row_col in enumerate(zip(edge_index[0], edge_index[1])):
                            row, col = row_col
                            if a_atom == row and b_atom == col:
                                edge_motif[0].append(i_cliques)
                                edge_motif[1].append(j_cliques)
                                attr_motif.append(edge_attr[i])
                                break

    edge_motif = torch.tensor(edge_motif).to(device=device, dtype=torch.long)
    if len(attr_motif) > 0:
        attr_motif = torch.stack(attr_motif).to(device=device, dtype=torch.long)
    else:
        attr_motif = torch.tensor([]).to(device=device, dtype=torch.long)

    g_motif_graph = Data()
    g_motif_graph.num_nodes = len(cliques)
    g_motif_graph.x = torch.tensor([]).to(device=device, dtype=torch.long)
    g_motif_graph.edge_attr = attr_motif
    g_motif_graph.edge_index = edge_motif

    edge_cliques = []
    attr_cliques = []
    num_edge_cliques = []
    g_motifs = []
    for clique in cliques:
        g_motif = Data()
        g_motif.num_nodes = len(clique)

        edge_clique = [[], []]
        attr_clique = []
        for i, a_atom in enumerate(clique):
            for j, b_atom in enumerate(clique):
                if i == j:
                    continue
                else:
                    for i_edge, row_col in enumerate(zip(edge_index[0], edge_index[1])):
                        row, col = row_col
                        if a_atom == row and b_atom == col:
                            edge_clique[0].append(i)
                            edge_clique[1].append(j)
                            attr_clique.append(edge_attr[i_edge])
                            break

        g_motif.edge_index = torch.tensor(edge_clique).to(device=device, dtype=torch.long)
        if len(attr_clique) > 0:
            g_motif.edge_attr = torch.stack(attr_clique).to(device=device, dtype=torch.long)
        else:
            g_motif.edge_attr = torch.tensor(attr_clique).to(device=device, dtype=torch.long)
        g_motif.x = x[clique]
        g_motifs.append(g_motif)

        num_edge_cliques.append(len(attr_clique))
        edge_cliques.append(torch.tensor(edge_clique).to(device=device, dtype=torch.long))
        if len(attr_clique) > 0:
            attr_cliques.append(torch.stack(attr_clique))
        else:
            attr_cliques.append(torch.tensor(attr_clique))

    edge_index_cliques_tensor = torch.tensor([[], []]).to(device=device, dtype=torch.long)
    attr_cliques_tensor = torch.tensor([]).to(device=device, dtype=torch.long)
    if len(edge_cliques) > 0:
        edge_index_cliques_tensor = torch.cat(edge_cliques, dim=1).to(device=device, dtype=torch.long)
        attr_cliques_tensor = torch.cat(attr_cliques, dim=0).to(device=device, dtype=torch.long)

    return edge_motif, attr_motif, edge_index_cliques_tensor, attr_cliques_tensor, num_edge_cliques, g_motifs, g_motif_graph


def motif_decomp(mol):
    graph_mol = mol
    n_atoms = graph_mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    cliques = []

    for bond in graph_mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(graph_mol))

    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms > len(c) > 0]

    cliques = [c for c in cliques if n_atoms > len(c) > 0]
    if len(cliques) == 0:
        return [[c for c in range(n_atoms)]]
    return cliques


