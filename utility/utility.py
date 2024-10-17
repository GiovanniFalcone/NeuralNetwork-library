import random
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate

class Utility:
    @staticmethod
    def print_info_dataset(x_tr_shape, y_tr_shape, x_val_shape, y_val_shape, x_te_shape, y_te_shape):
        data = [
            ["X_train", *x_tr_shape],
            ["Y_train", *y_tr_shape],
            ["X_val", *x_val_shape],
            ["Y_val", *y_val_shape],
            ["X_test", *x_te_shape],
            ["Y_test", *y_te_shape],
        ]
        
        headers = ["Dataset", "X shape", "Y shape"]
        print(tabulate(data, headers=headers, tablefmt="grid"), "\n")

    @staticmethod
    def print_info_network(
        input_size,
        n_hidden_layer,
        hidden_neurons,
        weights_shape,
        output_size,
        activation_function,
        error_function
    ) -> None:
        """
        Print in a tabular way the network structure.

        Parameters:
        ----------
            input_size (int): Number of input neurons.
            n_hidden_layer (int): Number of hidden layers.
            hidden_neurons (list of int): List of number of neurons for each hidden layer.
            weights_shape (list of tuple): List of tuple containing the shape for each weight matrix.
            output_size (int): Number of output neurons of network (i.e number of classes).
            activation_function (list of string): A list containing the activation function name for each layer.
            error_function (string): Name of error function.
        """
         
        if not isinstance(input_size, int) or not isinstance(output_size, int):
            raise ValueError("input_size and output_size must be integer...")

        if not isinstance(n_hidden_layer, int):
            raise ValueError("n_hidden_layer must be integer...")
        
        if not isinstance(hidden_neurons, list) or not all(isinstance(n, int) for n in hidden_neurons):
            raise ValueError("hidden_neurons must be a list of integer...")
        
        if len(hidden_neurons) != n_hidden_layer:
            raise ValueError("The length of hidden_neurons must match n_hidden_layer.")
        
        if not isinstance(weights_shape, list) or not all(isinstance(n, tuple) for n in weights_shape):
            raise ValueError("weights_shape must be a list of tuple...")
        
        if not isinstance(activation_function, list) or not all(isinstance(f, str) for f in activation_function):
            raise ValueError("activation_function must be a list of string.")
        
        if not isinstance(error_function, str):
            raise ValueError("error_function must be a string.")

        data = []

        # input layer
        data.append(["Input", input_size, None, None, None, None, None])
        # hidden layers
        for h in range(n_hidden_layer):
            data.append([f"Hidden Layer #{h + 1}", None, None, hidden_neurons[h], weights_shape[h], activation_function[h], None])
        # output layer
        data.append(["Output", None, output_size, None, weights_shape[-1], activation_function[-1], error_function])

        headers = ["Layer", "Input size", "Output size", "#N. of neurons \n of hidden layer", 
                   "Weights shape", "Activation \nfunctions", "Error \nfunctions"]
        table = tabulate(data, headers=headers, tablefmt="grid", colalign=("center",))

        # table size
        table_lines = table.split("\n")
        table_width = len(table_lines[0])

        # compute total parameters (w1[0]*w1[1] + ... wn[0]*wn[1] + b1 + ... bn)
        total_params = 0
        for w_shape in weights_shape:
            params = w_shape[0] * w_shape[1]
            n_bias = w_shape[0] # same number of neurons
            total_params += (params + n_bias)

        model_summary = ("=" * table_width + "\n"
                         f"Network summary\n\n")

        summary = (
            "=" * table_width + "\n"
            f"Total params: {total_params}\n"
            + "=" * table_width
        )

        print(f"{model_summary} {table}, \n\n{summary}\n")

    @staticmethod
    def get_random_elem(test, target, plot=True):
        """
        This function returns a randomly selected element from the set and plots it.

        Parameters:
        ----------
            test (matrix): The test set, which has shape (d, N) where d=number of feature and N number of samples.
            target (matrix): The labels associated to the test set. 
            plot (boolean): Plot the element chosen if plot=True. Defaults to True.

        Returns:
        --------
            elem (matrix): An element of test test. It has shape (d, 1).
            gold (int): the label associated to the chosen element.
        """
        if not isinstance(test, np.ndarray) or len(test.shape) != 2:
            raise ValueError("Test set must be a numpy array with a shape of (d, N)!")
        if not isinstance(target, np.ndarray) or len(target.shape) != 2:
            raise ValueError("Target set (labels of test set) must be a numpy array with a shape of (c, N)!")

        # get random element from test
        x = random.randint(0, test.shape[1] - 1)
        ix = np.reshape(test[:, x],(28,28))
        # plot it
        if plot:
            plt.figure()
            plt.imshow(ix, 'gray')
            plt.show()

        return test[:, x:x + 1], target[:, x:x + 1].argmax(0)
    
    @staticmethod
    def split_batch(Y, k):
        """
        Create a list of 'k' arrays.
        Each array represents the mini-batch. 
        Each of them will contain a permutation of elements.
        Finally, it returns this list.

        Parameters:
        ----------
            Y (np.ndarray): is the one-hot encoded matrix (c, N) of labels.
            k (int): number of mini-batch

        Returns:
        ----------
            indY (List[np.ndarray]): list of mini-batches.

        Errors:
            Raise a ValueError if 'Y' is not a numpy array with a shape of 2,
            or if k is not > 1.
        """
        if not isinstance(Y, np.ndarray) or len(Y.shape) != 2:
            raise ValueError("Y set must be a numpy array with a shape of (d, N)!")
        if not isinstance(k, int) or not k > 1:
            raise ValueError("k must be an integer > 1!")
        
        # get number of classes
        d = Y.shape[0]
        # create k sub-set (i.e k numpy.array)
        indY = [np.ndarray(0, int) for i in range(k)]
        for i in range(d):
            # For each class 'i', 'boolVector' contains 'True' for examples belonging to class 'i'
            boolVector=(Y.argmax(0) == i)
            # Get the index of True elements
            index=np.argwhere(boolVector == True).flatten()
            # create k array and distributes the indexes equally
            y = np.array_split(index, k)
            # populatee each mini-batch 
            for j in range(len(y)):
                indY[j]=np.append(indY[j], y[j])
        # shuffle all the indexes of each mini-batch
        for i in range(k):
            indY[i] = np.random.permutation(indY[i])    
        
        return indY