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

import math
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """
    def __init__(self, lstm_hidden_dim, embedding_size):
        """
        Initialize all parameters of the LSTM class.

        Args:
            lstm_hidden_dim: hidden state dimension.
            embedding_size: size of embedding (and hence input sequence).

        TODO:
        Define all necessary parameters in the init function as properties of the LSTM class.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.Wgx = None
        self.Wgh =None
        self.Wix =None
        self.Wih =None
        self.Wfx =None
        self.Wfh =None
        self.Wox =None
        self.Woh =None

        self.Whx = None
        self.Whh = None

        self.h = None

        self.bg =None
        self.bi =None
        self.bf =None
        self.bo =None
        self.bh =None
        #######################
        # END OF YOUR CODE    #
        #######################
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all your above-defined parameters,
        with a uniform distribution with desired bounds (see exercise sheet).
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        torch.FloatTensor(self.hidden_dim, self.embed_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))

        self.Wgx = torch.FloatTensor(self.hidden_dim, self.embed_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))
        self.Wix = torch.FloatTensor(self.hidden_dim, self.embed_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))
        self.Wfx = torch.FloatTensor(self.hidden_dim, self.embed_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))
        self.Wox = torch.FloatTensor(self.hidden_dim, self.embed_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))

        self.Wgh = torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))
        self.Wih = torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))
        self.Wfh = torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))
        self.Woh = torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))
        self.Whx = torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))
        self.Whh = torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))


        self.h = torch.zeros((self.Wgh.shape[1], 1))

        self.bg = torch.zeros((self.Wgh.shape[1], 1))
        self.bi = torch.zeros((self.Wih.shape[1], 1))
        self.bf = torch.ones((self.Wfh.shape[1], 1)) #Initialised with extra ones
        self.bo = torch.zeros((self.Woh.shape[1], 1))
        self.bh = torch.zeros((self.Whh.shape[1], 1))

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, embeds):
        """
        Forward pass of LSTM.

        Args:
            embeds: embedded input sequence with shape [input length, batch size, hidden dimension].

        TODO:
          Specify the LSTM calculations on the input sequence.
        Hint:
        The output needs to span all time steps, (not just the last one),
        so the output shape is [input length, batch size, hidden dimension].
        """
        #
        #
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        X = embeds



        self.g = torch.tanh(self.Wgx @ X + self.Wgh @ self.h + self.bg)
        self.i = torch.sigmoid(self.Wix @ X + self.Wih @ self.h + self.bi)
        self.f = torch.sigmoid(self.Wfx @ X + self.Wfh @ self.h + self.bf) #Check shapes
        self.o = torch.sigmoid(self.Wox @ X + self.Woh @ self.h + self.bo)
        self.c = torch.sigmoid(self.g * self.i + self.c * self.f)
        self.h = torch.tanh(self.c) * self.o

        #######################
        # END OF YOUR CODE    #
        #######################


class TextGenerationModel(nn.Module):
    """
    This module uses your implemented LSTM cell for text modelling.
    It should take care of the character embedding,
    and linearly maps the output of the LSTM to your vocabulary.
    """
    def __init__(self, args):
        """
        Initializing the components of the TextGenerationModel.

        Args:
            args.vocabulary_size: The size of the vocabulary.
            args.embedding_size: The size of the embedding.
            args.lstm_hidden_dim: The dimension of the hidden state in the LSTM cell.

        TODO:
        Define the components of the TextGenerationModel,
        namely the embedding, the LSTM cell and the linear classifier.
        """
        super(TextGenerationModel, self).__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        ## Embedding
        self.embedding = nn.Embedding(args.vocabulary_size, args.embedding_size)

        ## LSTM
        self.LSTM_cell = LSTM(args.lstm_hidden_dim, args.embedding_size) #How to do this?

        ## Linear Classifier
        # self.Wph = torch.distributions.uniform.Uniform(-1 / math.sqrt(args.lstm_hidden_dim), 1 / math.sqrt(args.lstm_hidden_dim),
        #                 (args.lstm_hidden_dim, args.lstm_hidden_dim))
        # self.bp = torch.zeros((self.Wph.shape[1], 1))

        self.linear = nn.Linear(args.lstm_hidden_dim, args.lstm_hidden_dim)

        self.y = None

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input

        TODO:
        Embed the input,
        apply the LSTM cell
        and linearly map to vocabulary size.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        ##Does one need to do one-hot encoding here?
        # x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size) #Not possible to use torch encoding?
        # Don't need to with embedding function?

        #Embedding
        x = self.embedding(x)

        #Apply LSTM
        self.LSTM_cell.forward(x)
        ## Teacher forcing
        # for j in range(x.shape[0]) #need to get right axis
        #     self.LSTM_cell.forward(x[:j][:][:])

        #Linearly map to vocab size
            # self.p = self.Wph @ self.LSTM_cell.h + self.bp
        self.p = self.linear(self.LSTM_cell.h) #But should it be applied this way? Since it is h we want to apply it to.
        self.y = torch.softmax(self.p)

        ##Teacher forcing
        # Need to take in labels?

        #######################
        # END OF YOUR CODE    #
        #######################

    def sample(self, batch_size=4, sample_length=30, temperature=0.):
        """
        Sampling from the text generation model.

        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function should default to argmax sampling,
        else to softmax sampling with specified temperature.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        ## Generate Sentences starting with random character

        if temperature==0:
            ## argmax sampling
            pass
        else:
            ## softmax sampling
            pass
        ## Ignore temperature for now until everything else working

        #######################
        # END OF YOUR CODE    #
        #######################
