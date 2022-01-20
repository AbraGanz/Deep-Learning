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
# Date Created: 2021-11-01
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ReLU layer.

        TODO:
        Implement module setup of the network.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ is use_batch_norm is True.
        
        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss module for loss calculation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super(MLP, self).__init__()

        if isinstance(n_classes,list):
            n_classes=n_classes[0]

        # Initialise
        self.ReLU = nn.ReLU()
        self.layers = []

        for i in range(len(n_hidden)+1):
            if i == 0:
                self.linear = nn.Linear(n_inputs, n_hidden[-1])
                nn.init.xavier_normal_(self.linear.weight)
                self.layers.append(self.linear)
                if use_batch_norm:
                    self.layers.append(self.BatchNorm1d(n_hidden[-1]))
                self.layers.append(self.ReLU)
            elif i == len(n_hidden):
                self.linear = nn.Linear(n_hidden[0], n_classes)
                nn.init.kaiming_normal_(self.linear.weight)
                self.layers.append(self.linear)
            else:
                self.linear = nn.Linear(n_hidden[-i], n_hidden[-(i+1)])
                nn.init.kaiming_normal_(self.linear.weight)
                self.layers.append(self.linear)
                if use_batch_norm:
                    self.layers.append(self.BatchNorm1d(n_hidden[0]))
                self.layers.append(self.ReLU)

        self.model = nn.Sequential(*self.layers) #*layers?

        # self.linear = nn.Linear(n_inputs, n_hidden)
        # nn.init.xavier_normal_(self.linear.weight)
        #
        # self.linear2 = nn.Linear(n_hidden, n_classes)
        # nn.init.kaiming_normal_(self.linear2.weight)

        # self.model_512 = nn.Sequential(
        #     nn.Linear(n_inputs, 512),
        #     nn.BatchNorm1d(512) if use_batch_norm,
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256) if use_batch_norm,
        #     nn.ReLU(),
        #     nn.Linear(216, 128),
        #     nn.BatchNorm1d(128) if use_batch_norm,
        #     nn.ReLU(),
        #     nn.Linear(128, n_classes)
        # )
        #
        # self.model_256 = nn.Sequential(
        #     nn.Linear(n_inputs, 256),
        #     nn.BatchNorm1d(256) if use_batch_norm,
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128) if use_batch_norm,
        #     nn.ReLU(),
        #     nn.Linear(128, n_classes)
        # )
        #
        # self.model_128 = nn.Sequential(
        #     nn.Linear(n_inputs, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Linear(128, n_classes)
        # )
        #
        # self.n_hidden = n_hidden


        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        x = x.reshape(x.shape[0], 3072)
        out = self.model(x)

        # if len(self.n_hidden) == 1:
        #     out = self.model_128(x)
        #
        # if len(self.n_hidden) == 2:
        #     out = self.model_256
        #
        # if len(self.n_hidden) == 3:
        #     out = self.model_512

        # x = x.reshape(x.shape[0], 3072)
        # x = self.linear(x)
        # #Batch normalisation
        # if self.use_batch_norm:
        #     x = self.batchnorm #Should affine be false?
        # x = self.ReLU(x)
        # out = self.linear2(x)
        # # loss = self.CrossEntropy(x)
        # self.forw = out
        #
        # x = x.reshape(x.shape[0], 3072)
        # for layer in self.layers:
        #     x = layer(x)


        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
    
