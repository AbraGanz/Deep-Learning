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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    preds = torch.argmax(predictions, axis=1)
    results = []
    for i in range(len(preds)):
        if preds[i] == targets[i]:
            results.append(1)
        else:
            results.append(0)
    # pdb.set_trace()
    return sum(results) / len(results)


    #######################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval() # Set model to eval mode Necessary?
    true_preds, num_preds = 0., 0.

    for data_inputs, targets in data_loader:
        data_inputs = data_inputs.reshape(data_inputs.shape[0], 3072).T
        predictions = model.forward(data_inputs)
        pred_labels = torch.argmax(predictions, axis=0)
        for i in range(len(pred_labels)):
            if pred_labels[i] == targets[i]:
                true_preds += 1
        # true_preds += (pred_labels == data_labels).sum()
        num_preds += targets.shape[0]

    avg_accuracy = true_preds / num_preds
    # print(f"Accuracy of the model: {100.0*acc:4.2f}%")

    #######################
    # END OF YOUR CODE    #
    #######################
    
    return avg_accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    #Uncomment?

    # if torch.cuda.is_available():  # GPU operation have separate seed
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.determinstic = True
    #     torch.backends.cudnn.benchmark = False

    # # Set default device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    model = MLP(n_inputs=3072, n_hidden=hidden_dims, n_classes=10)
    loss_module = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), lr)

    # cifar10_loader['validation']

    # TODO: Training loop including validation
    #Training
    # Forward pass, backwards pass
    for epoch in tqdm(range(epochs)):
        accuracies = []
        losses = []
        val_accuracies = []
        val_losses = []
        predictions = torch.empty(size=(0,10))#naughty
        labels = []
        print('training epoch', epoch)
        for x, y in cifar10_loader['train']:
            # Forward Pass
            out = model.forward(x)
            loss = loss_module.forward(out, y)
            losses.append(loss.item())

            ## Calculate accuracies
            acc = accuracy(out, y)
            accuracies.append(acc)

            # Backward Pass
            model.zero_grad()  # Is this necessary? Or with .SGD.zero_grad()?
            loss.backward()

            ## Update parameters
            optimiser.step()  # What are the params? Are they not calculated automatically?

    # for epoch in tqdm(range(epochs)):
    #     for x, y in cifar10_loader:
    #
    #         ## Forward Pass
    #         out = model(x)
    #         # preds = preds.squeeze(dim=1)  # Output is [Batch size, 1], but we want [Batch size]
    #         # What should this be?
    #
    #         ## Loss
    #         loss = loss_module(out, y) # .float()?
    #
    #         ## Backwards Pass
    #         nn.optim.zero_grad() #Is this necessary? Or with .SGD.zero_grad()?
    #         loss.backward()
    #
    #         ## Update parameters
    #         nn.optim.SGD(model.parameters(), lr) #What are the params? Are they not calculated automatically?

        ## Validation
        for valx, valy in cifar10_loader['validation']:
            valout = model.forward(valx)
            predictions = torch.cat((predictions, valout), 0)
            valloss = loss_module.forward(valout, valy)
            val_losses.append(valloss.item())
            labels.append(valy)
        # predictions = np.concatenate(predictions, 0)
        # predictions = torch.from_numpy(predictions)
        labels = torch.cat(labels)
        valacc = accuracy(predictions=predictions, targets=labels)

        ## Printing results each epoch
        print('training loss:', np.mean(losses))
        print('training accuracies:', np.mean(accuracies))
        print('val accuracy:', valacc)
        print('val loss:', np.mean(val_losses))

    # TODO: Do optimization with the simple SGD optimizer
         # Done earlier?

    # TODO: Test best model
    # test_accuracy = evaluate_model(model, cifar10_loader)
    # TODO: Add any information you might want to save for plotting
    logging_info = {'val_accuracies': val_accuracies, 'val_losses': val_losses}

    # print(test_accuracy)

    # state_dict = model.state_dict()
    # torch.save(state_dict, "best_model.tar")
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies,  logging_info #test_accuracy,
#

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    