import numpy as np

from network.activation_functions import ActivationFunctions

class Loss:
    @staticmethod
    def cross_entropy_softmax(y_pred, y_true, der=0):
        """
        It computes the cross entropy applying softmax to the prediction. 
        It's used for multi-class classification problems

        Parameters
        ----------
            y_pred (matrix): a matrix with a shape of (n_classes, n_examples). 
            It contains raw values for each examples. 
            y_true (matrix): a matrix with a shape of (n_classes, n_examples) of one-hot encoded true class labels
            der (int): if 0 it returns the cross entropy, otherwise it returns the derivative. Defaults to 0.
            
        Returns
        -------
            cross_entropy (float): cross entropy of the inputs if der = 0. 
            If der > 0 it returns the derivative of loss function with respect to the predictions.
        """
        #soft max is computing considering overflow
        z = ActivationFunctions.softmax(y_pred) + 1e-10
        if der == 0:
            #here the cross-entropy with soft-max is computed
            return - np.sum(y_true * np.log(z)) 
        else:
            return z - y_true
        
    @staticmethod
    def cross_entropy(y_pred, y_true, der = 0, epsilon=1e-15):
        """
        Compute cross entropy given predictions as class probabilities and one-
        hot encoded ground truth labels.

        Parameters
        ----------
            y_pred (matrix): a matrix with a shape of (n_classes, n_examples). 
            The values represent the predicted class probabilities for each sample.
            y_true (matrix): a matrix with a shape of (n_classes, n_examples) of one-hot encoded true class labels
            der (int): if 0 it returns the cross entropy, otherwise it returns the derivative. Defaults to 0.
            epsilon (float): a constant to clip predicted probabilities to avoid taking log of zero
            
        Returns
        -------
            cross_entropy (float): cross entropy of the inputs if der = 0. 
            If der > 0 it returns the derivative of loss function with respect to the predictions.
        """
        # range [epsilon, 1. - epsilon]
        y = np.clip(y, epsilon, 1. - epsilon)
        if der == 0:
            return - np.sum(y_true * np.log(y))
        else:
            # compute derivative
            return y - y_true
        
    @staticmethod
    def sum_of_squares(y_pred, y_true, der = 0):
        """
        Loss function for regression problems; it returns the sum of squares if der = 0, derivative 
        of loss function with respect to the predictions otherwise.

        Parameters
        ----------
            y_pred (matrix): a matrix with a shape of (n_classes, n_examples). 
            y_true (matrix): a matrix with a shape of (n_classes, n_examples) containing the true target values.
            der (int): if 0 it returns the SSE, otherwise it returns the partial derivative. Defaults to 0.
            
        Returns
        -------
            SSE (float): if der = 0, a value that quantifies the level of error in the model's prediction. 
            If der > 0 the derivative of the SSE with respect to the predictions.
        """
        z= y_pred - y_true
        if der == 0:
            return 0.5 * np.sum(np.power(z, 2))
        else:
            return z