# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:20:57 2025

@author: Acer
"""

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        # Below code store the size of the input layer
        self.input_size = input_size
        
        # below code store the size of the hidden layer
        self.hidden_size = hidden_size
        
        # below code store the size of the output layer
        self.output_size = output_size
        
        # below code use to add activation function
        self.activation_name = activation
        
        # below code initialize the weights for the input to hidden layer
        self.weights_input_hidden = np.random.randn(
            self.input_size, self.hidden_size) * 0.01
        
        #below code initialize the weights for hidden to output layer
        self.weights_input_output = np.random.randn(
            self.hidden_size, self.output_size) * 0.01
        
        # below code initialize the bias for hidden layer
        self.bias_hidden = np.zeros((1, self.hidden_size))
        
        # below code initialize the bias for output layer
        self.bias_output = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def activate(self, x):
        if self.activation_name == 'relu':
            return self.relu(x)
        elif self.activation_name == 'sigmoid':
            return self.sigmoid(x)
        
    def activate_derivative(self, x):
        if self.activation_name == 'relu':
            return self.relu_derivative(x)
        elif self.activation_name == 'sigmoid':
            return self.sigmoid_derivative(x)
        
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def mse_derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
    def feedforward(self, X):
        # below code calculates activation for hidden layer
        self.hidden_activation = np.dot(
            X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.activate(self.hidden_activation)
        
        # below code activate the function to hidden layer
        self.output_activation = np.dot(self.hidden_output, self.weights_input_output) + self.bias_output
        
        # below code applies the activation function to output layer
        #self.predicted_output = self.sigmoid(self.output_activation)
        self.predicted_output = self.output_activation
        
        return self.predicted_output
    
    def backward(self, X, y, learning_rate):
        # below code calculates the error at the output layer
        #output_error = y - self.predicted_output
        
        # below code calculates the delta for the output layer
        #output_delta = output_error * \
            #self.sigmoid_derivative(self.predicted_output)
            
        #compute gradient of MSE
        output_delta = self.mse_derivative(y, self.predicted_output)
            
        # below code calculates the error at the hidden layer
        hidden_error = np.dot(output_delta, self.weights_input_output.T)
        #hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        hidden_delta = hidden_error * self.activate_derivative(self.hidden_output)
        
        # below code updates the weights between hidden and output layers
        self.weights_input_output -= np.dot(self.hidden_output.T,
                                            output_delta) * learning_rate
        self.bias_output -= np.sum(output_delta, axis=0,
                                   keepdims=True) * learning_rate
        
        # below code updates weights between input and hidden layers
        self.weights_input_hidden -= np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden -= np.sum(hidden_delta, axis=0,
                                   keepdims=True) * learning_rate
        
class SGD:
    def __init__(self, learning_rate = 0.01, batch_size = 32):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
    def get_batches(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, n_samples, self.batch_size):
            end = i + self.batch_size
            yield X_shuffled[i:end], y_shuffled[i:end]