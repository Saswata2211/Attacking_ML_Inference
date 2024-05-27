from layer import Layer
import numpy as np
from fxpmath import Fxp

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
    
    from fxpmath import Fxp

    
    # returns output for a given input
    def forward_propagation(self, input_data, layer_index):
        self.input = input_data
        self.input = Fxp(self.input, n_word=16, n_frac=4)
        self.weights = Fxp(self.weights, n_word=16, n_frac=4)
        self.bias = Fxp(self.bias, n_word=16, n_frac=4)
        self.bias = np.array(self.bias)
        self.bias=self.bias.reshape(-1 , 1)
        self.bias=self.bias.T
        print("Input:", self.input.shape)
        print("Weight:", self.weights.shape)
        #print("Bias:",self.bias.shape)
        self.output = np.dot(self.input, self.weights) + self.bias
        self.output = Fxp(self.output,n_word=16, n_frac=4)
        print("Output:",self.output.shape)
        print("Output:",self.output.dtype )
        
        #choose layer to change
        '''
        if layer_index == 1:
            #print(self.output.shape)
            self.output=np.array(self.output)  
            additional_matrix = np.zeros_like(self.output)
            additional_matrix[0, 1] = 9
            #additional_matrix[1, 1] = 5    #Add the error
            self.output += additional_matrix
        '''
        if layer_index == 1:
            additional_matrix = np.zeros_like(self.output)
            additional_matrix = Fxp(additional_matrix, n_word=16, n_frac=4)  # Convert to Fxp
            additional_matrix[0, 1] = Fxp(15, n_word=16, n_frac=4)
            additional_matrix[0, 3] = Fxp(-6, n_word=16, n_frac=4)
            self.output += additional_matrix
        if layer_index == 5:
            additional_matrix = np.zeros_like(self.output)
            additional_matrix = Fxp(additional_matrix, n_word=16, n_frac=4)  # Convert to Fxp
            additional_matrix[0, 1] = Fxp(9, n_word=16, n_frac=4)
            additional_matrix[0, 3] = Fxp(4, n_word=16, n_frac=4)
            self.output += additional_matrix    
        return self.output


       
    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        print("Shape of self.input:", self.input.shape)
        print("Shape of output_error:", output_error.shape)
        self.input=self.input.reshape(1, -1)
        weights_error = np.dot(self.input, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error