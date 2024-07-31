import logging

import numpy as np
import torch
import torch_geometric
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score

from tqdm import tqdm
import argparse
import torch.nn.functional as F

from Utils.args_utils import ArgsInit
from dataset_motifclassifier import PygGraphPropPredDataset_motifclassifier

from model.motifyclassifier_model import Motify_Classifier

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def get_scores(edges_pos, edges_neg, adj_rec, adj_orig):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def sparse_to_tuple(sparse_mx):
    if not sparse_mx.is_sparse:
        sparse_mx = sparse_mx.to_sparse()

    sparse_coo = sparse_mx.coalesce().coo()

    coords = torch.stack((sparse_coo.row, sparse_coo.col), dim=0).cpu().numpy()
    values = sparse_coo.data.cpu().numpy()
    shape = sparse_coo.shape

    return coords, values, shape

def train(model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration {step}")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()

            adj = torch.zeros(batch.num_nodes, batch.num_nodes, dtype=torch.float)
            adj[batch.edge_index[0], batch.edge_index[1]] = 1
            adj[batch.edge_index[1], batch.edge_index[0]] = 1
            pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
            norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

            adj_label = adj + torch.eye(adj.shape[0])
            weight_mask = adj_label.to_dense().view(-1) == 1
            weight_tensor = torch.ones(weight_mask.size(0))
            weight_tensor[weight_mask] = pos_weight

            loss = norm * F.binary_cross_entropy(pred.view(-1), adj_label.to_dense().view(-1),
                                                 weight=weight_tensor)
            kl_divergence = 0.5 / pred.size(0) * (
                    1 + 2 * model.log_std - model.mean ** 2 - torch.exp(model.log_std) ** 2).sum(1).mean()
            loss -= kl_divergence
            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # region args
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--drop_ratio', type=float, default=0.05,
                        help='dropout ratio (default: 0.5)')

    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--hidden_channels', type=int, default=300,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molbace",
                        help='dataset name')
    parser.add_argument('--lr', type=float, default=0.0001,  # 修改lr后效果变好
                        help='learning rate set for optimizer.')
    parser.add_argument('--JK', type=str, default="last",
                        help='sum, last')
    parser.add_argument('--residual', type=bool, default=False,
                        help='residual')

    # save model
    parser.add_argument('--model_save_path', type=str, default='model',
                        help='the directory used to save models')
    parser.add_argument('--model_encoder_load_path', type=str, default='',
                        help='the path of trained encoder')

    # load trained model for test
    parser.add_argument('--model_load_path', type=str, default='',
                        help='the path of trained model')
    parser.add_argument('--model_direct_load_path', type=str, default='',
                        help='the path of trained model')
    parser.add_argument('--save', type=str, default='MOTIFY-Pretrain', help='experiment name')

    args = ArgsInit(parser).save_exp4pretrain()
    # endregion

    log = logging.getLogger()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    dataset = PygGraphPropPredDataset_motifclassifier(root='pretrain/dataset', name=args.dataset)

    split_idx = dataset.get_idx_split()

    train_loader = torch_geometric.loader.DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=args.num_workers)

    model = Motify_Classifier(num_layer=args.num_layer,
                                  emb_dim=args.hidden_channels,
                                  drop_ratio=args.drop_ratio, JK=args.JK, residual=args.residual).to(device)

    if not args.model_load_path == '':
        model = torch.load(args.model_load_path)
    if not args.model_encoder_load_path == '':
        model.motif_encoder_load(torch.load(args.model_encoder_load_path))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        log.info("=====Epoch {}".format(epoch))
        if args.model_load_path == '':
            log.info('Training...')
            train(model, device, train_loader, optimizer, log)

    log.info('Finished training!')

    if not args.model_direct_load_path == '':
        torch.save(model.motif_encoder, args.model_direct_load_path)
    if not args.model_save_path == '':
        torch.save(model.motif_encoder, args.model_save_path)


if __name__ == "__main__":
    main()
