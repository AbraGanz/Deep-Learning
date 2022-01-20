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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from data import get_mlp_features, get_node_features, get_dense_adj
import pdb


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_outputs):
        """
        Initializes MLP object.
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_outputs: This number is required in order to specify the
                     output dimensions of the MLP
        TODO: 
        - define a simple MLP that operates on properly formatted QM9 data
        """

        # DOes this perform multinomial logisitc regression if len(n_hidden) = 0?

        super(MLP, self).__init__()

        # Initialise
        self.ReLU = nn.ReLU() #Should we be using ReLU?
        self.layers = []

        for i in range(len(n_hidden) + 1):
            if i == 0:
                self.linear = nn.Linear(n_inputs, n_hidden[-1])
                nn.init.xavier_normal_(self.linear.weight)
                self.layers.append(self.linear)
                # if use_batch_norm: #Could be good to do anyway
                #     self.layers.append(nn.BatchNorm1d(n_hidden[-1]))
                self.layers.append(self.ReLU)
            elif i == len(n_hidden):
                self.linear = nn.Linear(n_hidden[0], n_outputs)
                nn.init.kaiming_normal_(self.linear.weight) #Done already in nn.linear?
                self.layers.append(self.linear)
            else:
                self.linear = nn.Linear(n_hidden[-i], n_hidden[-(i + 1)])
                nn.init.kaiming_normal_(self.linear.weight)
                self.layers.append(self.linear)
                # if use_batch_norm:
                #     self.layers.append(nn.BatchNorm1d(n_hidden[-(i + 1)]))
                self.layers.append(self.ReLU)

        self.model = nn.Sequential(*self.layers)  # *layers?

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
            x: input to the network
        Returns:
            out: outputs of the network
        """
        x = get_mlp_features(x)
        out = self.model(x)

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device


class GNN(nn.Module):
    """implements a graphical neural network in pytorch. In particular, we will use pytorch geometric's nn_conv module so we can apply a neural network to the edges.
    """

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        n_hidden: int,
        n_output: int,
        num_convolution_blocks: int,
    ) -> None:
        """create the gnn

        Args:
            n_node_features: input features on each node
            n_edge_features: input features on each edge
            n_hidden: hidden features within the neural networks (embeddings, nodes after graph convolutions, etc.)
            n_output: how many output features
            num_convolution_blocks: how many blocks convolutions should be performed. A block may include multiple convolutions
        
        TODO: 
        - define a GNN which has the following structure: node embedding -> [ReLU -> RGCNConv -> ReLU -> MFConv] x num_convs -> Add-Pool -> Linear -> ReLU -> Linear
        - Once the data has been pooled, it may be beneficial to apply another MLP on the pooled data before predicing the output.
        """
        
        super(GNN, self).__init__()
        #embedding
        self.node_embedding = nn.Linear(n_node_features, n_hidden)

        # x num_cons = so many layers.

        layers = []

        for i in range(num_convolution_blocks):
            layers += [nn.ReLU(), geom_nn.RGCNConv(n_hidden, n_hidden, n_edge_features), nn.ReLU(), geom_nn.MFConv(n_hidden, n_hidden)]

        self.layers = nn.ModuleList(layers)

        self.mlp = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output))


    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            edge_attr: edge attributes (pytorch geometric notation)
            batch_idx: Index of batch element for each node

        Returns:
            prediction
        """
        
        nodes = get_node_features(x)
        edge_type = edge_attr.argmax(1)
        x = self.node_embedding(nodes)
        for layer in self.layers:
            if isinstance(layer, geom_nn.RGCNConv):
                # print('isinstance(layer, geom_nn.RGCNConv)')
                # pdb.set_trace()
                x = layer(x, edge_index, edge_type) # edge_attr in third position?
            elif isinstance(layer, geom_nn.MFConv):
                # print('isinstance(layer, geom_nn.MFConv)')
                # pdb.set_trace()
                x = layer(x, edge_index)
            else:
                # print('else')
                # pdb.set_trace()
                x = layer(x)
        y = geom_nn.global_add_pool(x, batch=batch_idx)  # Should it be like this here?
        out = self.mlp(y)

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
