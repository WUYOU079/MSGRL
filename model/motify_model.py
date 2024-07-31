import pytorch_lightning
import torch
from torch import optim
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, Set2Set, AttentionalAggregation

from Utils.motif_utils import flatten_2d_list_with_batch
from model.base_model import GNN_encoder

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


class Motify_GNN(pytorch_lightning.LightningModule):

    def __init__(self, num_tasks, atom_feature_size=None, bond_feature_size=None, num_layer=5, emb_dim=300,
                 residual=False, drop_ratio=0.5, JK="last",
                 task_type="classification", evaluator=None, lr=0.0001):

        super(Motify_GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        # region pytorch-lightning
        self.task_type = task_type
        self.evaluator = evaluator
        self.lr = lr

        self.y_true_val = []
        self.y_pred_val = []
        self.y_true_test = []
        self.y_pred_test = []
        # endregion
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.motif_encoder = GNN_encoder(self.num_layer, atom_feature_size, bond_feature_size, self.emb_dim, JK=self.JK,
                                         drop_ratio=self.drop_ratio,
                                         residual=residual)

        self.graph_encoder = GNN_encoder(self.num_layer, self.emb_dim, bond_feature_size, self.emb_dim, JK=self.JK,
                                         drop_ratio=self.drop_ratio,
                                         residual=residual)

        self.pool = AttentionalAggregation(
            gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                        torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        self.motif_pool = AttentionalAggregation(
            gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                        torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))

        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):

        motif_list, motif_batch = flatten_2d_list_with_batch(batched_data.motif_list,
                                                             batched_data.x.device)
        batched_motifs = Batch.from_data_list(motif_list)

        batched_motifsgraph = Batch.from_data_list(batched_data.g_motif_graph)

        h_motif = self.motif_encoder(batched_motifs)

        h_motif_graph = self.motif_pool(h_motif, batched_motifs.batch)

        h_graph_node = self.graph_encoder(
            [h_motif_graph, batched_motifsgraph.edge_index, batched_motifsgraph.edge_attr])

        h_graph = self.pool(h_graph_node, batched_motifsgraph.batch)
        h_final = self.graph_pred_linear(h_graph)
        return h_final

    def motif_encoder_load(self, encoder):
        self.motif_encoder = encoder
        for param in self.motif_encoder.parameters():
            param.requires_grad = False

    # region pytorch-lightning
    def training_step(self, batch):
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = self(batch)
            is_labeled = batch.y == batch.y
            if "classification" in self.task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            self.log('train_loss', loss)
            return loss

    def validation_step(self, batch):
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = self(batch)

        self.y_true_val.append(batch.y.view(pred.shape).detach().cpu())
        self.y_pred_val.append(pred.detach().cpu())

    def test_step(self, batch):
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = self(batch)

        self.y_true_test.append(batch.y.view(pred.shape).detach().cpu())
        self.y_pred_test.append(pred.detach().cpu())

    def on_validation_epoch_end(self):
        y_true = torch.cat(self.y_true_val, dim=0).numpy()
        y_pred = torch.cat(self.y_pred_val, dim=0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        loss = self.evaluator.eval(input_dict)
        self.log('val_loss', loss[[self.evaluator.eval_metric]])

    def on_test_epoch_end(self):
        y_true = torch.cat(self.y_true_test, dim=0).numpy()
        y_pred = torch.cat(self.y_pred_test, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        loss = self.evaluator.eval(input_dict)
        self.log('test_loss', loss[self.evaluator.eval_metric])

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    # endregion
