import numpy as np
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
    
    # Function to load weights from files
    '''
    def load_weights(self, directory_path):
        for i, layer in enumerate(self.layers):
            weights = np.load(directory_path + f'layer_{i}_weights.npy')
            bias = np.load(directory_path + f'layer_{i}_bias.npy')
            layer.weights = weights
            layer.bias = bias
    
    def load_weights(self, directory_path):
        for i, layer in enumerate(self.layers):
            weights = np.load(directory_path + f'layer_{i}_weights.npy')
            bias = np.load(directory_path + f'layer_{i}_bias.npy')
            layer.weights = weights
            layer.bias = bias
        
    '''
    
    def load_weights(self, directory_path):
    # Load weights for the first layer
        weights = np.load(directory_path + 'layer_0_weights.npy')
        print(self.layers)
        bias = np.load(directory_path + 'layer_0_bias.npy')
        self.layers[0].weights = weights
        self.layers[0].bias = bias
        
        weights = np.load(directory_path + 'layer_0_weights.npy')
        print(self.layers)
        bias = np.load(directory_path + 'layer_0_bias.npy')
        self.layers[1].weights = weights
        self.layers[1].bias = bias
        
        # Load weights for the second layer
        weights = np.load(directory_path + 'layer_1_weights.npy')
        print(self.layers)
        bias = np.load(directory_path + 'layer_1_bias.npy')
        self.layers[2].weights = weights
        self.layers[2].bias = bias
        
        

        weights = np.load(directory_path + 'layer_1_weights.npy')
        print(self.layers)
        bias = np.load(directory_path + 'layer_1_bias.npy')
        self.layers[3].weights = weights
        self.layers[3].bias = bias

        

        # Load weights for the third layer
        weights = np.load(directory_path + 'layer_2_weights.npy')
        print(weights.shape)
        bias = np.load(directory_path + 'layer_2_bias.npy')
        self.layers[4].weights = weights
        self.layers[4].bias = bias
    
         
        weights = np.load(directory_path + 'layer_2_weights.npy')
        print(weights.shape)
        bias = np.load(directory_path + 'layer_2_bias.npy')
        self.layers[5].weights = weights
        self.layers[5].bias = bias
        
    '''  
    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            layer_outputs = [output.copy()]  # Store initial input
            for layer_index, layer in enumerate(self.layers):
                #if layer_index == len(self.layers) - 1:
                    #print(f"Shape after layer {layer_index + 1}: {output.shape}")
                if layer_index == 1:  # Modify this index to choose the layer (0-based)
                    output = layer.forward_propagation(output, layer_index)
                    print(f"Output after layer {layer_index + 1}: {output}") 
                    print(f"i: {i}, layer_index: {layer_index} ")
                else:
                    output = layer.forward_propagation(output)
                    print(f"Output after layer {layer_index + 1}: {output}") 
                layer_outputs.append(output)
            result.append(output)
        return result

    '''
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []
        
        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer_index, layer in enumerate(self.layers):
                output = layer.forward_propagation(output, layer_index)
                print(f"Layer {layer_index+1} output: {output}")
            result.append(output)
        
        return result
    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer_index, layer in enumerate(self.layers):
                    output = layer.forward_propagation(output,layer_index)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))