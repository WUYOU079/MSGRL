import logging

import pytorch_lightning
import torch
import torch_geometric


from tqdm import tqdm
import argparse
import numpy as np

from ogb.graphproppred import Evaluator

from Utils.args_utils import ArgsInit
from dataset_smiles import PygGraphPropPredDataset_Smiles

from model.motify_model import Motify_GNN

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration {step}")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
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
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate set for optimizer.')
    parser.add_argument('--JK', type=str, default="last",
                        help='last、sum')
    parser.add_argument('--residual', type=bool, default=False,
                        help='residual')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molbace",
                        help='dataset name')

    # save model
    parser.add_argument('--model_save_path', type=str, default='model',
                        help='the directory used to save models')
    # load pre-trained model for test
    parser.add_argument('--model_load_path', type=str, default='',
                        help='the path of trained model')
    parser.add_argument('--model_pretrain_load_path', type=str,
                        default='',)
    parser.add_argument('--save', type=str, default='MOTIFY', help='experiment name')

    args = ArgsInit(parser).save_exp()
    # endregion
    log = logging.getLogger()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    dataset = PygGraphPropPredDataset_Smiles(name=args.dataset)

    split_idx = dataset.get_idx_split()

    evaluator = Evaluator(args.dataset)

    train_loader = torch_geometric.loader.DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=args.num_workers)
    valid_loader = torch_geometric.loader.DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size,
                                                     shuffle=False,
                                                     num_workers=args.num_workers)
    test_loader = torch_geometric.loader.DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers)

    # init model
    model = Motify_GNN(num_tasks=dataset.num_tasks, num_layer=args.num_layer,
                           emb_dim=args.hidden_channels, residual=args.residual,
                           drop_ratio=args.drop_ratio, JK=args.JK,
                           evaluator=evaluator, lr=args.lr, task_type=dataset.task_type).to(device)

    if not args.model_load_path == '':
        model = torch.load(args.model_load_path)
    if args.model_pretrain_load_path != '':
        model.motif_encoder_load(torch.load(args.model_pretrain_load_path))

    trainer = pytorch_lightning.Trainer(max_epochs=args.epochs, accelerator="cpu")

    trainer.fit(model=model, train_dataloaders=train_loader)
    valid_curve = trainer.test(model=model, dataloaders=valid_loader)
    test_curve = trainer.test(model=model, dataloaders=test_loader)

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))

    log.info('Finished training!')
    log.info('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    log.info('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.model_save_path == '':
        torch.save(model, args.model_save_path)


if __name__ == "__main__":
    main()
