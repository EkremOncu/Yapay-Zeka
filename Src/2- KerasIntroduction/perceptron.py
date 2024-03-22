import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def output(self, x):
        return self.sigmoid(np.dot(x, self.weights) + self.bias)
        
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** -x)
    
weights = np.array([1, 2, 3, 4, 5])
bias = 2

x = np.array([3, 2, 1, 4, 6])

neuron = Neuron(weights, bias)
result = neuron.output(x)
print(result)


