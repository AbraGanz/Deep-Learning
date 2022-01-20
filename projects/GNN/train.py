################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-17
################################################################################
from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import *
from networks import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
# import wandb
# wandb.init(project="my-test-project")

def permute_indices(molecules: Batch) -> Batch:
    """permute the atoms within a molecule, but not across molecules

    Args:
        molecules: batch of molecules from pytorch geometric

    Returns:
        permuted molecules
    """
    # Permute the node indices within a molecule, but not across them.
    ranges = [
        (i, j) for i, j in zip(molecules.ptr.tolist(), molecules.ptr[1:].tolist())
    ]
    permu = torch.cat([torch.arange(i, j)[torch.randperm(j - i)] for i, j in ranges])

    n_nodes = molecules.x.size(0)
    inits = torch.arange(n_nodes)
    # For the edge_index to work, this must be an inverse permutation map.
    translation = {k: v for k, v in zip(permu.tolist(), inits.tolist())}

    permuted = deepcopy(molecules)
    permuted.x = permuted.x[permu]
    # Below is the identity transform, by construction of our permutation.
    permuted.batch = permuted.batch[permu]
    permuted.edge_index = (
        permuted.edge_index.cpu()
        .apply_(translation.get)
        .to(molecules.edge_index.device)
    )
    return permuted


def compute_loss(
    model: nn.Module, molecules: Batch, criterion: Callable
) -> torch.Tensor:
    """use the model to predict the target determined by molecules. loss computed by criterion.

    Args:
        model: trainable network
        molecules: batch of molecules from pytorch geometric
        criterion: callable which takes a prediction and the ground truth 

    Returns:
        loss
    """

    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    torch.no_grad()

    edge_index = molecules.edge_index
    edge_attr = molecules.edge_attr
    batch_idx = molecules.batch

    labels = get_labels(molecules)
    # labels = labels.to(device)

    if args.model == 'gnn':
        predictions = model.forward(molecules, edge_index, edge_attr, batch_idx)
    else:
        # pdb.set_trace()
        predictions = model.forward(molecules).squeeze()
        # predictions = (out, dim=0) #What does output of network look like? Should it be this dim?
    predictions = predictions.squeeze()
    labels = labels.squeeze()
    loss = criterion(predictions, labels) #Check correct sizes here (see terminal)

    return loss


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, criterion: Callable, permute: bool
) -> float:
    """
    Performs the evaluation of the model on a given dataset.

    Args:
        model: trainable network
        data_loader: The data loader of the dataset to evaluate.
        criterion: loss module, i.e. torch.nn.MSELoss()
        permute: whether to permute the atoms within a molecule
    Returns:
        avg_loss: scalar float, the average loss of the model on the dataset.

    
    TODO: conditionally permute indices
          calculate loss
          average loss independent of batch sizes
    """

    num_results = 0
    total_loss = 0

    for x in data_loader:
        if permute:
            x = permute_indices(x) #Also need to permute y?
        loss = compute_loss(model, x, criterion) #Does this not already take batch size into account? Need to do individually?
        total_loss += loss
        # pdb.set_trace()
        num_results += x.y.shape[0] #What is correct dimension? Currently calculating batch size, but maybe should be number of batches

    avg_loss = total_loss/num_results

    return avg_loss


def train(
    model: nn.Module, lr: float, batch_size: int, epochs: int, seed: int, data_dir: str
):
    """a full training cycle of an mlp / gnn on qm9.

    Args:
        model: a differentiable pytorch module which estimates the U0 quantity
        lr: learning rate of optimizer
        batch_size: batch size of molecules
        epochs: number of epochs to optimize over
        seed: random seed
        data_dir: where to place the qm9 data

    Returns:
        model: the trained model which performed best on the validation set
        test_loss: the loss over the test set
        permuted_test_loss: the loss over the test set where atomic indices have been permuted
        val_losses: the losses over the validation set at every epoch
        logging_info: general object with information for making plots or whatever you'd like to do with it

    TODO:
    - Implement the training of both the mlp and the gnn in the same function
    - Evaluate the model on the whole validation set each epoch.

    """
    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Loading the dataset
    train, valid, test = get_qm9(data_dir, model.device)
    train_dataloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        exclude_keys=["pos", "idx", "z", "name"],
    )
    valid_dataloader = DataLoader(
        valid, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )
    test_dataloader = DataLoader(
        test, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(device)
    # TODO: Initialize loss module and optimizer
    criterion = nn.MSELoss()
    # criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters()) #Add amsgrad? #Use learning rate?

    # TODO: Training loop including validation, using evaluate_model

    val_losses = []

    for epoch in range(epochs):
        for batch in train_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, batch, criterion)
            # loss.requires_grad = True
            loss.backward() #How do this?
        val_loss = evaluate_model(model, valid_dataloader, criterion, False)
        print('val_loss', val_loss)
        val_losses.append(val_loss)
        # wandb.log({'loss': val_loss})
        if val_loss <= min(val_losses):
            best_state_dict = model.state_dict()  # ???
            torch.save(best_state_dict, args.model)

    # TODO: Do optimization, used adam with amsgrad. (many should work)
    # val_losses = ...
        optimizer.step()
        # scheduler.step()

    model.load_state_dict(torch.load(args.model))
    # TODO: Test best model
    test_loss = evaluate_model(model = model, data_loader = test_dataloader, criterion = criterion, permute = False)
    print('test loss', test_loss)
    # TODO: Test best model against permuted indices
    permuted_test_loss = evaluate_model(model = model, data_loader = test_dataloader, criterion = criterion, permute = True)
    print('permuted loss', permuted_test_loss)
    # TODO: Add any information to save for plotting
    logging_info = {'losses:': val_losses, 'test loss': test_loss, 'permuted test loss': permuted_test_loss} #Accuracies?

    return model, test_loss, permuted_test_loss, val_losses, logging_info


def main(**kwargs):
    """main handles the arguments, instantiates the correct model, tracks the results, and saves them."""
    which_model = kwargs.pop("model")
    mlp_hidden_dims = kwargs.pop("mlp_hidden_dims")
    gnn_hidden_dims = kwargs.pop("gnn_hidden_dims")
    gnn_num_blocks = kwargs.pop("gnn_num_blocks")

    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if which_model == "mlp":
        model = MLP(FLAT_INPUT_DIM, mlp_hidden_dims, 1)
    elif which_model == "gnn":
        model = GNN(
            n_node_features=Z_ONE_HOT_DIM,
            n_edge_features=EDGE_ATTR_DIM,
            n_hidden=gnn_hidden_dims,
            n_output=1,
            num_convolution_blocks=gnn_num_blocks,
        )
    else:
        raise NotImplementedError("only mlp and gnn are possible models.")

    model.to(device)
    model, test_loss, permuted_test_loss, val_losses, logging_info = train(
        model, **kwargs
    )


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--model",
        default="mlp",
        type=str,
        choices=["mlp", "gnn"],
        help="Select between training an mlp or a gnn.",
    )
    parser.add_argument(
        "--mlp_hidden_dims",
        default=[128, 128, 128, 128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the mlp. To specify multiple, use " " to separate them. Example: "256 128"',
    )
    parser.add_argument(
        "--gnn_hidden_dims",
        default=64,
        type=int,
        help="Hidden dimensionalities to use inside the mlp. The same number of hidden features are used at every layer.",
    )
    parser.add_argument(
        "--gnn_num_blocks",
        default=2,
        type=int,
        help="Number of blocks of GNN convolutions. A block may include multiple different kinds of convolutions!",
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")

    # Technical
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the qm9 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
