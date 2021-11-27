###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Created: 2021-11-11
###############################################################################
"""
Main file for Question 1.2 of the assignment. You are allowed to add additional
imports if you want.
"""
import os
import json
import argparse
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms

from augmentations import gaussian_noise_transform, gaussian_blur_transform, contrast_transform, jpeg_transform
from cifar10_utils import get_train_validation_set, get_test_set

import pickle
from tqdm.auto import tqdm


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name, num_classes=10):
    """
    Returns the model architecture for the provided model_name. 

    Args:
        model_name: Name of the model architecture to be returned. 
                    Options: ['debug', 'vgg11', 'vgg11_bn', 'resnet18', 
                              'resnet34', 'densenet121']
                    All models except debug are taking from the torchvision library.
        num_classes: Number of classes for the final layer (for CIFAR10 by default 10)
    Returns:
        cnn_model: nn.Module object representing the model architecture.
    """
    if model_name == 'debug':  # Use this model for debugging
        cnn_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32*32*3, num_classes)
            )
    elif model_name == 'vgg11':
        cnn_model = models.vgg11(num_classes=num_classes)
    elif model_name == 'vgg11_bn':
            cnn_model = models.vgg11_bn(num_classes=num_classes)
    elif model_name == 'resnet18':
        cnn_model = models.resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        cnn_model = models.resnet34(num_classes=num_classes)
    elif model_name == 'densenet121':
        cnn_model = models.densenet121(num_classes=num_classes)
    else:
        assert False, f'Unknown network architecture \"{model_name}\"'
    return cnn_model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
    Returns:
        model: Model that has performed best on the validation set.

    TODO:
    Implement the training of the model with the specified hyperparameters
    Save the best model to disk so you can load it later.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Initialise? When is this needed & when not?
    # Set model to training mode
    best_model = model
    loss_module = nn.CrossEntropyLoss()

    # Move to device
    model = model.to(device)
    loss_module = loss_module.to(device)

    model.train()

    # Load the datasets
    cifar10_train, cifar10_val= get_train_validation_set(data_dir)
    cifar10_loader_train = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    cifar10_loader_val = torch.utils.data.DataLoader(cifar10_val, batch_size=batch_size, shuffle=True)

    # Initialize the optimizers and learning rate scheduler. 
    # We provide a recommend setup, which you are allowed to change if interested.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)
    
    # Training loop with validation after each epoch. Save the best model, and remember to use the lr scheduler.
    for epoch in tqdm(range(epochs)):
        accuracies = []
        losses = []
        val_accuracies = []
        val_losses = []
        predictions = torch.empty(size=(0, 10))  # naughty
        predictions = predictions.to(device)
        labels = []
        print('Training epoch', epoch)
        for x, y in cifar10_loader_train:
            x = x.to(device)
            y = y.to(device)
            x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
            # Forward Pass
            out = model.forward(x)
            loss = loss_module.forward(out, y)
            losses.append(loss.item())

            ## Caluclating Accuracy
            results = []
            train_preds = torch.argmax(out, axis=1)
            for i in range(len(train_preds)):
                if train_preds[i] == y[i]:
                    results.append(1)
                else:
                    results.append(0)
            accuracy = sum(results) / len(results)
            accuracies.append(accuracy)

            # Backward Pass
            model.zero_grad()
            loss.backward()

            ## Update parameters
            optimizer.step()

            ## Validation
        for valx, valy in cifar10_loader_val:
            valx.reshape((valx.shape[0], valx.shape[1]*valx.shape[2]*valx.shape[3]))
            valx = valx.to(device)
            valy = valy.to(device)
            valout = model.forward(valx)
            predictions = torch.cat((predictions, valout), 0)
            valloss = loss_module.forward(valout, valy)
            val_losses.append(valloss.item())
            labels.append(valy)
        labels = torch.cat(labels)

        ## Calculating Validation Accuracies
        val_results = []
        val_preds = torch.argmax(predictions, axis=1)
        for i in range(len(val_preds)):
            if val_preds[i] == labels[i]:
                val_results.append(1)
            else:
                val_results.append(0)
        valacc = sum(val_results) / len(val_results)
        print(valacc)
        val_accuracies.append(valacc)

        #Check if better model
        if valacc >= max(val_accuracies):
            # best_model = deepcopy(model) #Save to disk as pt, model state dict
            best_state_dict = model.state_dict() #???
            torch.save(best_state_dict, checkpoint_name)


        scheduler.step()



    # Load best model and return it.
    model.load_state_dict(torch.load(checkpoint_name))
    model = best_model
    
    #######################
    # END OF YOUR CODE    #
    #######################
    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    TODO:
    Implement the evaluation of the model on the dataset.
    Remember to set the model in evaluation mode and back to training mode in the training loop.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model = model.to(device)
    model.eval()

    test_predictions = torch.empty(size=(0, 10))
    test_labels = []
    for testx, testy in data_loader:
        testx = testx.to(device)
        testy = testy.to(device)
        testx.reshape((testx.shape[0], testx.shape[1]*testx.shape[2]*testx.shape[3]))
        testout = model.forward(testx)
        test_predictions = torch.cat((test_predictions, testout), 0)
        test_labels.append(testy)
    test_labels = torch.cat(test_labels)

    test_preds = torch.argmax(test_predictions, axis=1)
    test_results = []
    for i in range(len(test_preds)):
        if test_preds[i] == test_labels[i]:
            test_results.append(1)
        else:
            test_results.append(0)
    accuracy =  sum(test_results) / len(test_results)



    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def test_model(model, batch_size, data_dir, device, seed):
    """
    Tests a trained model on the test set with all corruption functions.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Evaluate the model on the plain test set. Make use of the evaluate_model function.
    For each corruption function and severity, repeat the test. 
    Summarize the results in a dictionary (the structure inside the dict is up to you.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(seed)
    test_results = {}

    #Load data
    cifar10_test = get_test_set(data_dir) #How to use batchsize??
    cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size, True)


    accuracy = evaluate_model(model, cifar10_test_loader, device)
    test_results['base'+model] = accuracy

    ## Corruption Functions?
    # Need to corrupt data, and after each time pass it to evaluate_model
    pass


    #######################
    # END OF YOUR CODE    #
    #######################
    return test_results


def main(model_name, lr, batch_size, epochs, data_dir, seed):
    """
    Function that summarizes the training and testing of a model.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Load model according to the model name.
    Train the model (recommendation: check if you already have a saved model. If so, skip training and load it)
    Test the model using the test_model function.
    Save the results to disk.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(seed)

    ## Load the model
    model = get_model(model_name)

    ## Train the model if not already saved
    # if #don't already have model:
    train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name=model_name, device=device)

    ## Test the Model
    test_results = test_model(model, batch_size, data_dir, device, seed)

    # Save test results to disk
    # Should they be binary(file) or text(txt?) or pkl? Or sth else?
    with open("test-results.pkl", "wb") as f:
        pickle.dump(test_results, f, pickle.HIGHEST_PROTOCOL)

    #######################
    # END OF YOUR CODE    #
    #######################



if __name__ == '__main__':
    """
    The given hyperparameters below should give good results for all models.
    However, you are allowed to change the hyperparameters if you want.
    Further, feel free to add any additional functions you might need, e.g. one for calculating the RCE and CE metrics.
    """
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--model_name', default='debug', type=str,
                        help='Name of the model to train.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=150, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)