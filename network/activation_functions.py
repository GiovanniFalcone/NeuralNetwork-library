import numpy as np

class ActivationFunctions:
    @staticmethod
    def identity(x, der = 0):
        """
        Return identity function if der=0, derivative of identity function otherwise.
        """
        if der == 0:
            return x
        else:
           return 1

    @staticmethod
    def sigmoid(x, der = 0):
        """
        Return sigmoid function if der=0, derivative of sigmoid function otherwise.
        """
        # in order to avoid overflow
        # source: https://stackoverflow.com/questions/40726490/overflow-error-in-pythons-numpy-exp-function
        z = np.clip(x, -709.78, 709.78)

        if der == 0:
            return 1/(1 + np.exp(-z)) 
        else:
            y = 1/(1 + np.exp(-z)) 
            return  y * (1 - y)
    
    @staticmethod
    def tanh(x, der = 0):
        """
        Return tanh function if der=0, derivative of tanh function otherwise.
        """
        if der == 0:
            return np.tanh(x)
        else:
            return 1 - np.tanh(x)**2
        
    @staticmethod
    def relu(x, der = 0):
        """
        Return relu function if der=0, derivative of relu function otherwise.
        
        relu:
            - f(x) = x if x > 0
            - f(x) = 0 otherwise

        Derivative of relu:
            - f'(x) = 1 if x > 0
            - f'(x) = 0 otherwise
        """
        if der == 0:
            return np.maximum(0, x)
        else:
            return np.where(x > 0, 1, 0)

    @staticmethod
    def l_relu(x, der = 0, alpha = 0.01):
        """
        Return leaky relu function if der=0, derivative of leaky relu function otherwise.

        Formula:
            - f(x) = alpha * x if x <= 0
            - f(x) = x if x > 0

        Derivative of leaky relu function:
            - f'(x) = 1 if x > 0
            - f'(x) = alpha otherwise
        """
        if der == 0:
            return np.where(x > 0, x, x * alpha)  
        else:
            return np.where(x > 0, 1, alpha)
        
    @staticmethod
    def softmax(y):
        #soft max is computing considering overflow
        y_exp = np.exp(y - y.max(0))
        z = y_exp / sum(y_exp, 0) #here is soft-max
        return z