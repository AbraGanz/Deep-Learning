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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy

import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils

import torch

import pdb
import random
import matplotlib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#DELETE LATER
import wandb
wandb.init(project="DL-Lab-1", entity="somniavero")


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Need to calculate percentage of targets that are correct
    # For each row, check which column max probability in
    # If correct column as target number, corrections + 1

    preds = np.argmax(predictions, axis=1)
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

    #Which batches will we use here?
    #Make sure batch-size taken into account
    #One element of train is already a batch!

    # accuracies = []
    # # testx, testy = next(iter(cifar10_loader['validation']))
    # for batch_x, batch_y in data_loader:
    #     batch_x = batch_x.reshape((batch_x.shape[0], 3072)) #WHY does W now have first dimension 128 isntead of 3072?
    #     predictions = model.forward(batch_x) #The problem must be here with the model, with its import.
    #     acc = accuracy(predictions, batch_y)
    #     accuracies.append(acc)
    # avg_accuracy = np.mean(accuracies)
    #
    true_preds, num_preds = 0., 0.

    for data_inputs, data_labels in data_loader:
        data_inputs = data_inputs.reshape(data_inputs.shape[0],3072).T
        predictions = model.forward(data_inputs)
        pred_labels = np.argmax(predictions, axis=0)
        for i in range(len(pred_labels)):
            if pred_labels[i] == targets[i]:
                true_preds += 1
        # true_preds += (pred_labels == data_labels).sum()
        num_preds += data_labels.shape[0]

    avg_accuracy = true_preds / num_preds
    # print(f"Accuracy of the model: {100.0*acc:4.2f}%")

    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
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

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    # One element of train is already a batch! A tuple of data, then labels

    #######################
    # PUT YOUR CODE HERE  #
    #######################


    # print('shape of valx:', valx)


    # TODO: Initialize model and loss module
    model = MLP(n_inputs=3072, n_hidden= hidden_dims, n_classes=10)
    loss_module = CrossEntropyModule() #What is the point of this?

    # TODO: Training loop including validation

    # Forward pass, backwards pass
    for epoch in range(epochs):
        accuracies = []
        losses = []
        val_accuracies = []
        val_losses = []
        predictions = []
        labels = []
        print('training epoch',epoch)
        for x, y in cifar10_loader['train']: # Shoudl be using next(iter(***))?
            #Forward Pass

            out = model.forward(x)

            #Backward Pass
            loss = loss_module.forward(out, y)
            losses.append(loss)
            dout = loss_module.backward(out, y)
            model.backward(dout)

        #   Updating the weights and bias
        #     pdb.set_trace()
            acc = accuracy(out, y)
            accuracies.append(acc)

            model.Linear.params['weight'] -= lr*model.Linear.grads['weight'] #Or should this be +? model.params[weight] += lr*weight_grad
            model.Linear.params['bias'] -= lr*model.Linear.grads['bias']

            model.Linear2.params['weight'] -= lr * model.Linear2.grads['weight']  # Or should this be +? model.params[weight] += lr*weight_grad
            model.Linear2.params['bias'] -= lr * model.Linear2.grads['bias']

    # Validation:
        for valx, valy in cifar10_loader['validation']:
            valout = model.forward(valx)
            predictions.append(valout)
            valloss = loss_module.forward(valout, valy)
            val_losses.append(valloss)
            labels.append(valy)
        predictions = np.concatenate(predictions, 0)
        labels = np.concatenate(labels)
        valacc = accuracy(predictions=predictions, targets=labels)
        print('training loss:', np.mean(losses))
        print('training accuracies:', np.mean(accuracies))
        print('val accuracy:', valacc)
        print('val loss:', np.mean(val_losses))

        wandb.log({'training loss': np.mean(losses), 'training accuracies': np.mean(accuracies), "validation loss": np.mean(val_losses), 'validation accuracies': valacc})


    # print(model.Linear.params['weight'])
    # print(model.Linear.params['bias'])
    # print(model.Linear2.params['weight'])
    # print(model.Linear2.params['bias'])
    #Saving the best model via deepcopy:
    best_model = copy.deepcopy(model) # Will this be our model at the end?
    best_params = copy.deepcopy(model.Linear.params, model.Linear2.params) #Is this how to do it?

    print(val_accuracies)
    print(val_losses)


    # TODO: Test best model
    test_accuracy = evaluate_model(model=model, data_loader=cifar10_loader['test']) #Can one just specify 'model' here?
    print(test_accuracy)
    # TODO: Add any information you might want to save for plotting
    logging_dict = {'val_losses': val_losses, 'accuracies': accuracies, 'val_accuracies': val_accuracies, 'best model': best_model, 'best params': best_params}# add 'test_accuracy:': test_accuracy}
    # torch.save(logging_dict, 'Users/Zaatar/Documents/MoL/Deep-Learning/')

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict # Check if this order is correct


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')

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

#PLOTTING

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 10,
  "batch_size": 128
}


# Functions for testing and understanding
#
# from modules import *
# random.seed(42)
# torch.manual_seed(42)

## Loading the dataset
# cifar10 = cifar10_utils.get_cifar10('data/')
# cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=128,
#                                                   return_numpy=True)
# model = MLP(n_inputs=3072, n_hidden=128, n_classes=10)
#
# x = random.choice(list(cifar10_loader['train'])[0])
# x = x.reshape((x.shape[0],3072))
# y =
# print(random.choice(list(cifar10_loader['train'])[0]))
# linear = LinearModule(in_features=3072, out_features=128, input_layer = True)
# linearT = torch.nn.modules.Linear(3072,128)
# torch.nn.init.xavier_normal(linearT.weight)
# ReLU = ReLUModule()
# SoftMax = SoftMaxModule()
# CrossEntropy = CrossEntropyModule()
# ReLUT = torch.nn.modules.ReLU()
# softmaxT = torch.nn.modules.Softmax()
# x = linear.forward(x)
# y = linearT.forward(y)
# print('MY MODULE:', x)
# print('TORCH', y)



