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
# Date Adapted: 2021-11-11
###############################################################################

from datetime import datetime
import argparse
from tqdm.auto import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset, text_collate_fn
from model import TextGenerationModel

# Setting up tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import random

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


def train(args):
    """
    Trains an LSTM model on a text dataset
    
    Args:
        args: Namespace object of the command line arguments as 
              specified in the main function.
        
    TODO:
    Create the dataset.
    Create the model and optimizer (we recommend Adam as optimizer).
    Define the operations for the training loop here. 
    Call the model forward function on the inputs, 
    calculate the loss with the targets and back-propagate, 
    Also make use of gradient clipping before the gradient step.
    Recommendation: you might want to try out Tensorboard for logging your experiments.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(args.seed)
    clip = args.clip_grad_norm
    # Load dataset
    # The data loader returns pairs of tensors (input, targets) where inputs are the
    # input characters, and targets the labels, i.e. the text shifted by one.

    dataset = TextDataset(args.txt_file, args.input_seq_length)
    data_loader = DataLoader(dataset, args.batch_size, 
                             shuffle=True, drop_last=True, pin_memory=True,
                             collate_fn=text_collate_fn)

    device = args.device
    args.vocabulary_size = dataset.vocabulary_size

    # Create model
    model = TextGenerationModel(args)
    model = model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters())

    loss_module = nn.CrossEntropyLoss()
    loss_module = loss_module.to(device)

    sentences = {}

    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        losses = []
        accuracies = []
        results = []
        epoch_loss = 0
        curr_results = 0
        num_results = 0
        # Training loop
        j = 0
        for x, y in data_loader:
            j += 1
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            predictions = [] #Should be tensor?
            for time in range(x.shape[0]):
                out = model.forward(x[:time+1])
                label = y[time][:]
                prediction = torch.squeeze(torch.argmax(out, dim = 1)[-1, :]) #What shape will this be, and what of it do we want? Should it be argmaxed?
                pred = torch.squeeze(out[-1, :]).T
                predictions.append(pred)
                loss = loss_module(pred, label) #Check y shape properly
                total_loss += loss.item()
                curr_results += sum((prediction == label).float())
                # if prediction[-1] == label: #find index of prediction
                #     results.append(1)
                # else:
                #     results.append(0)

                #     pred = pred.permute(1,0)

            # Need to have loss calculated over all predictions for backwards propagation
            predictions = torch.stack(predictions)
            predictions_loss = predictions.view((-1, args.vocabulary_size))
            labels_loss = y.view((-1,))
            loss = loss_module(predictions_loss, labels_loss)
            # print('loss for back prop', loss)
            # sentences[epoch] = model.sample()
            # for i in range(len(sentences[epoch])):
            #     sentences[epoch][i] = dataset.convert_to_string(sentences[epoch][i])
            #     print('sentence', sentences[epoch][i])
            loss.backward()
            epoch_loss += total_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            num_results += x.shape[0] * x.shape[1]
        accuracy = curr_results/num_results
        losses.append(epoch_loss / (30*args.batch_size))
        print(epoch_loss / (30*args.batch_size))
        accuracies.append(accuracy)
        print('accuracy', accuracy)
        if curr_results/num_results >= max(accuracies):
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, 'LSTM')

        writer.add_scalar("Loss/train", epoch_loss / (30*args.batch_size), epoch)
        writer.add_scalar("Accuracy", accuracy, epoch)

        # Sentence Generation
        if epoch == 1 or 5 or 20:
            for temperature in [0, 0.5, 1.0, 2.0]:
                sentences[epoch] = {}
                sentences[epoch][temperature] = model.sample()
                for i in range(len(sentences[epoch])):
                    sentences[epoch][temperature][i] = dataset.convert_to_string(sentences[epoch][i])
                print(sentences)

    ## Recommendation: you might want to try out Tensorboard for logging your experiments.

    return sentences, accuracies, losses
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_hidden_dim', type=int, default=1024, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding_size', type=int, default=256, help='Dimensionality of the embeddings.')

    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train with.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for.')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm')

    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU

    train(args)

    #Plotting
    writer.flush()

