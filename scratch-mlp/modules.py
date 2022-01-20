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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import pdb
import random
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization.
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #See video: https://www.youtube.com/watch?v=X5m7bC4xCLY&ab_channel=UvADeepLearningcourse


        if input_layer:
            if isinstance(out_features, list):
                out_features = out_features[0]
            std = np.sqrt(1/in_features)
        else:
            if isinstance(in_features, list):
                in_features = in_features[0]
            std = np.sqrt(2/in_features)



        weights = np.random.normal(0, std, size = (out_features, in_features))

        bias = np.zeros((1, out_features), dtype=np.float64)
        params = {'weight': weights, 'bias': bias}
        # pdb.set_trace()
        self.params = params
        self.grads = {'weight': np.zeros_like(weights), 'bias': np.zeros_like(bias)}








            # Initialise gradient with zeros?


        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################


        W = self.params['weight']
        B = self.params['bias']
        out = x @ W.T + B

        self.forw = out
        self.x = x

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        dx = dout @ self.params['weight']

        self.grads['weight'] = dout.T @ self.x
        self.grads['bias'] = np.ones((1, dout.shape[0])) @ dout

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads = None #Is this necessary?
        self.forw = None
        self.x = None

        #######################
        # END OF YOUR CODE    #
        #######################


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        out = x * (x>0)
        self.forw = out

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################


        dx = dout * (self.forw > 0)


        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.forw = None

        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:n
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        a = np.exp(x - np.max(x, axis = 1, keepdims=True))
        out = a / np.sum(a, axis = 1, keepdims=True) #Do we need to add axis =1? #Should we be using this division symbol? Element-wise
        self.forw = out

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # print('Y:', self.forw.shape, 'dout', dout.shape)
        # dx = self.forw * (dout - (dout * self.forw).sum(axis=-1, keepdims=True))

        dx = self.forw * (dout - ((dout * self.forw) @ np.outer(np.ones(dout.shape[1]), np.ones(dout.shape[1]))))



        # print(dx.shape)
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.forw = None
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Setting up one-hot matrix
        # Is there a better way to do this? can at least only loop through y instead of T
        # T = np.tile(y, (x.shape[1],1))
        # T = T.T
        # for i in range(len(y)): #Is there a way to do this without a loop?
        #     for j in range(x.shape[1]):
        #         if T[i][j] == j:
        #             T[i][j] = 1
        #         else:
        #             T[i][j] = 0

        # One-hot encoding using indexing T[x, y] = 1
        # Stack - convert array of indices to 1-hot encoded numpy

        # Try an alternative method:
        # T = np.zeroes_like(x)
        # log = np.log(x)
        # T[np.arange(log.size, y] = 1
        # print(T)
        # out = 1/len(y) * np.sum(-T)



        T = np.identity(x.shape[1])[y]

        # SOMETHING WRONG HERE
        out = 1/y.shape[0] * np.sum(-np.sum(T*np.log(x), axis = 1))

        # pdb.set_trace()



        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # T = np.tile(y, (x.shape[1], 1))
        # T = T.T
        # for i in range(len(y)):  # Is there a way to do this without a loop?
        #     for j in range(x.shape[1]):
        #         if T[i][j] == j:
        #             T[i][j] = 1
        #         else:
        #             T[i][j] = 0


        T = np.identity(x.shape[1])[y]
        dx = -1/len(y) * (T / (x + 1e-8))


        #######################
        # END OF YOUR CODE    #
        #######################

        return dx