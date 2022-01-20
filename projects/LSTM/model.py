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
import random
from dataset import TextDataset


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
        self.Wph = None

        self.bg =None
        self.bi =None
        self.bf =None
        self.bo =None
        self.bh =None

        self.h = None
        self.g = None
        self.i = None
        self.f = None
        self.o = None
        self.c = None
        self.p = None

        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all above-defined parameters,
        with a uniform distribution with desired bounds.
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """

        # print('hidden dims:', self.hidden_dim)
        # print('embed dims:', self.embed_dim) ##make into params
        self.Wfx = nn.Parameter(torch.FloatTensor( self.hidden_dim, self.embed_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)))
        self.Wgx = nn.Parameter(torch.FloatTensor( self.hidden_dim, self.embed_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)))
        # print('Wgx',self.Wgx.shape)
        self.Wix = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.embed_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)))
        self.Wix = nn.Parameter(torch.FloatTensor( self.hidden_dim, self.embed_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)))
        self.Wox = nn.Parameter(torch.FloatTensor( self.hidden_dim, self.embed_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)))

        self.Wgh = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)))
        self.Wih = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)))
        self.Wfh = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)))
        self.Woh = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)))
        self.Whx = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)))
        self.Whh = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)))

        self.bg = nn.Parameter(torch.zeros((self.Wgh.shape[1], 1)))
        self.bi = nn.Parameter(torch.zeros(( self.Wih.shape[1], 1)))
        self.bf = nn.Parameter(torch.ones(( self.Wfh.shape[1], 1)))
        self.bo = nn.Parameter(torch.zeros(( self.Woh.shape[1], 1)))
        self.bh = nn.Parameter(torch.zeros(( self.Whh.shape[1], 1)))

        self.Wph = nn.Parameter(torch.FloatTensor().uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)))
        self.bp = nn.Parameter(torch.zeros((self.Whh.shape[1], 1)))

        # params = [self.Wgx, self.Wix, self.Wix, self.Wox, self.Wgh, self.Wih, self.Wfh, self.Woh, self.Whx, self.Whh, self.Wph,
        # self.h, self.bg, self.bi, self.bf, self.bo, self.bh, self.bp]


    def forward(self, embeds):
        """
        Forward pass of LSTM.

        Args:
            embeds: embedded input sequence with shape [input length, batch size, embedding size].

        TODO:
          Specify the LSTM calculations on the input sequence.

        """

        X = embeds
        device = embeds.device
        X = X.to(device)

        if len(X.shape) == 3:
            self.h = torch.zeros((self.hidden_dim, X.shape[1]))
            self.c = torch.zeros((self.hidden_dim, X.shape[1]))
            outputs = torch.empty((X.shape[0], 1024, X.shape[1]), device=device)
        else:
            self.h = torch.zeros((self.hidden_dim, 1))
            self.c = torch.zeros((self.hidden_dim, 1))
            outputs = torch.empty((X.shape[0], 1024, 1), device=device)

        self.h = self.h.to(device)
        self.c = self.c.to(device)



        for j in range(X.shape[0]):
            x = X[j][:][:].T
            if len(x.shape) == 1:
                x = x.unsqueeze(-1)
            self.g = torch.tanh(self.Wgx @ x + self.Wgh @ self.h + self.bg)
            self.i = torch.sigmoid(self.Wix @ x + self.Wih @ self.h + self.bi)
            self.f = torch.sigmoid(self.Wfx @ x + self.Wfh @ self.h + self.bf) #Check shapes
            self.o = torch.sigmoid(self.Wox @ x + self.Woh @ self.h + self.bo)
            self.c = self.g * self.i + self.c * self.f
            self.h = torch.tanh(self.c) * self.o
            # self.p = self.Wph @ self.LSTM_cell.h + self.bp

            # torch.unsqueeze(self.h, 0)
            outputs[j][:][:] = self.h
        return outputs


class TextGenerationModel(nn.Module):
    """
    This module uses the implemented LSTM cell for text modelling.
    It takes care of the character embedding,
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

        ## Embedding
        self.embedding = nn.Embedding(args.vocabulary_size, args.embedding_size)

        ## LSTM
        self.LSTM_cell = LSTM(args.lstm_hidden_dim, args.embedding_size)

        ## Linear Classifier

        self.linear = nn.Linear(args.lstm_hidden_dim, args.vocabulary_size)
        # self.linear = nn.MapTable().add(nn.Linear(args.lstm_hidden_dim, args.vocabulary_size))

        self.vocab_size = args.vocabulary_size


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

        #Embedding
        x = self.embedding(x)

        #Apply LSTM
        # self.LSTM_cell.forward(x)
        ## Teacher forcing
        outputs = self.LSTM_cell.forward(x)

    #Linearly map to vocab size
        outputs = outputs.permute(0, 2, 1)
        self.p = self.linear(outputs)
        self.pred = self.p.permute(0, 2, 1)
 
        return self.pred

    def sample(self, batch_size=4, sample_length=30, temperature=0.):
        """
        Sampling from the text generation model.

        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function defaults to argmax sampling,
        else to softmax sampling with specified temperature.
        """

        ## Generate Sentences starting with random character
        sentences = {}
        if temperature == 0:
            max = torch.argmax
            div = 1
        else:
            max = torch.softmax
            div = temperature

        for i in range(batch_size):
            self.eval()
            sentence = []
            # Choose random first letter
            sentence.append(random.randint(0, self.vocab_size))
            # Generate sentence of length 30
            for j in range(sample_length):
                pred = self.forward(torch.IntTensor(sentence))
                letter = torch.squeeze(max(pred/div, dim=1)[-1, :])
                sentence.append(letter.item())
            sentences[i] = sentence
        return sentences
