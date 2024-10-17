import os
import copy
import pickle
import numpy as np

from time import perf_counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from utility.utility import Utility
from network.loss_functions import Loss
from network.early_stopping import EarlyStopping
from network.activation_functions import ActivationFunctions

class NeuralNetwork:
    def __init__(self, input_size, output_size, n_hidden_layers, m_neurons_list, activation_list, error_function):
        """
        Create the neural network object.

        Attributes:
        ----------
            input_size (int): Number of input neurons (e.g for MNIST dataset input size is 784 neurons).
            output_size (int): Number of output neurons of network (i.e number of classes - in case of MNIST output is 10).
            n_hidden_layer (int): Number of hidden layers.
            m_neurons_list (list of int): List of number of neurons for each hidden layer.
            activation_function (list): A list containing the activation function name for each layer (hidden and output).
            error_function (callable): Error function to use (e.g Loss.cross_entropy_softmax).
        """
        
        if n_hidden_layers + 1 != len(activation_list):
            raise ValueError("Lenght of activation list is not equal to the number of layers...")
        
        if n_hidden_layers != len(m_neurons_list):
            raise ValueError("Lenght of neurons list is not equal to the number of hidden layers...")
        
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_hidden_layers + 1 # + 1 -> output layer
        self.m_hidden_neurons_list = m_neurons_list
        self.activation_list = activation_list
        self.error_function = error_function
        self.weights = []
        self.biases = []

        # init weights and biases
        self.__init_parameters()

    def __init_parameters(self):
        """
        Initialize the matrix of weights w=(m, d) (m=number of neurons and d=previous connection) 
        and the matrix of biases b=(m, 1) for each layer (i.e hidden and output).
        """
        sigma = 0.1

        n_prev_neuron = self.input_size
        # hidden layers
        for m_hidden_neurons in self.m_hidden_neurons_list:
            self.weights.append(np.random.normal(size=[m_hidden_neurons, n_prev_neuron]) * sigma)
            self.biases.append(np.random.normal(size=[m_hidden_neurons, 1]) * sigma)
            n_prev_neuron = m_hidden_neurons
        # output neurons
        n_target_classes = self.output_size
        self.weights.append(np.random.normal(size=[n_target_classes, n_prev_neuron]) * sigma)
        self.biases.append(np.random.normal(size=[n_target_classes, 1]) * sigma)
        
    def summary(self):
        """
        Print all network information in tabular format.
        """
        # get number of hidden layers
        n_hidden = self.n_layers - 1 
        # get all weight matrix shapes
        weights_shape = []
        for w in self.weights: weights_shape.append(w.shape)
        # get all activation functions in string format
        act_fun = []
        for activation in self.activation_list: act_fun.append(activation.__name__)
        # print in tabular style
        Utility.print_info_network(input_size=self.input_size, 
                                   n_hidden_layer=n_hidden, 
                                   hidden_neurons=self.m_hidden_neurons_list, 
                                   weights_shape= weights_shape, 
                                   output_size=self.output_size, 
                                   activation_function=act_fun, 
                                   error_function=self.error_function.__name__)

    def copy_network(self):
        """Return a copy of network created at the beginning."""
        return copy.deepcopy(self)

    ###############################################################################################################
    #                                                   MODEL                                                     #
    ###############################################################################################################

    def train(self, 
              x_train: np.ndarray, 
              y_train: np.ndarray, 
              x_val: np.ndarray, 
              y_val: np.ndarray, 
              epochs: int, 
              early_stopper: EarlyStopping,
              learning_rate = 0.001, 
              momentum = 0.0, 
              mode='batch', 
              num_mini_batches = 32, 
              f1_avg_type = None):
        """
        Fit function.

        Parameters:
        ----------
            x_train (matrix): The training set. A matrix of shape (d, N) where d is the number of feature and N the number of samples. 
            y_train (matrix): The one-hot encoded matrix which contains the true labels of training set. 
            x_val (matrix): The validation set. A matrix of shape (d, N) where d is the number of feature and N the number of samples. 
            y_val (matrix): The one-hot encoded matrix which contains the true labels of validation set. 
            epochs (int): Number of epochs to train the model.
            learning_rate (float): the eta value in [0, 1] used for the update rule. Defaults to 0.01. 
            momentum (float): a value ≥ 0 that accelerates gradient descent in the relevant direction and dampens oscillations. 
            0 is vanilla gradient descent. Defaults to 0.0.
            mode (string): a string that represents the type of learning. It can assume three values: 'online', 'batch' or 'mini-batch'.
            Defaults to 'batch'.
            num_mini_bathes (int): An integer that represents the number of mini-batches. Defaults to 32. 
            If mode != 'mini-batch' it can assume any values.
            early_stopper (EarlyStopping): the early stopping criteria (patience, GL, PQ, UP)
            f1_avg_type (string): The type of average used in f1-score. It can be [None, micro, macro, weighted].

        Returns:
        ----------
            report (dict):
                - stop: the epoch at which learning process was interrupted.
                - epoch_best: the last epoch when validation error was the minimum (i.e the best net).
                - time: a float that represents the time needed to learning.
                - train_loss_list: a list where each element represents the training set loss for each epoch.
                - val_loss_list: a list where each element represents the validation set loss for each epoch.
                - train_ac_list: a list where each element represents the training set accuracy for each epoch.
                - val_ac_list: a list where each element represents the validation set accuracy for each epoch.
                - f1_score_t: f1-score on training set
                - f1-score_v: f1-score on validation set

        Errors:
        --------
            Raise a ValueError if any of these inputs are not of the right type or do not have legal values.
        """

        # check if all inputs are valid
        self.__train_validation(x_train, y_train, x_val, y_val, epochs, learning_rate, 
                              momentum, mode, num_mini_batches)        
        # in order to return the highest performing model, which is not necessarily the one of the last epoch
        best_net, min_error = None, None
        epoch_best_net = epochs
        # how much time was spent on learning
        tot_time = 0
        # list containing training loss, f1-score and accuracy of each epoch
        train_loss_list, train_ac_list, train_f1_list = [], [], []
        # list containing validation loss and training accuracy of each epoch
        val_loss_list, val_ac_list, val_f1_list = [], [], []
        # for early stopping
        stopped_epoch, flag = None, False

        print(f"Learning mode is {mode}\n[==============================]\n")       
        
        for e in range(epochs):
            start_time = perf_counter()
            # do batch - online - minibatch based on 'mode'
            self.__learning(x_train, y_train, learning_rate, momentum, mode, num_mini_batches)
            end_time = perf_counter() - start_time
            tot_time += end_time
            # get network prediction (validation can be none if network is trained using all data)
            y_pred_train = self.__forward_propagation(x_train)
            y_pred_val = self.__forward_propagation(x_val) if x_val is not None else None
            # updating lists
            train_loss_list, val_loss_list = self.__get_train_val_loss(y_pred_train, y_train, y_pred_val, y_val, train_loss_list, val_loss_list)
            train_ac_list, val_ac_list = self.__get_train_val_accuracy(y_pred_train, y_train, y_pred_val, y_val, train_ac_list, val_ac_list)
            train_f1_list, val_f1_list = self.__get_train_val_f1_score(y_pred_train, y_train, y_pred_val, y_val, train_f1_list, val_f1_list, f1_avg_type)

            print(f"Epoch {e + 1}/{epochs}")
            print(f"[==============================] - {end_time:.2f}s/step -", 
                  f"loss: {train_loss_list[e]:.4f} - accuracy: {train_ac_list[e]:.4f} -",
                 (f"val_loss: {val_loss_list[e]:.4f} - val_accuracy: {val_ac_list[e]:.4f} - " if x_val is not None else ""),
                  f"p: {early_stopper.get_patience_counter()}", end='\n')

            # for best network (min validation error)
            if x_val is not None and (best_net is None or val_loss_list[e] < min_error):
                min_error = val_loss_list[e]
                best_net = self.copy_network()
                # the epoch of min validation error
                epoch_best_net = e  
                early_stopper.reset_patience_counter()
                flag = False
            else:
                flag = True

            # early stopping
            if x_val is not None:
                # in order to verify the early stop condition
                e_train = train_loss_list[-1]
                e_val_curr = val_loss_list[-1] if x_val is not None else None
                # if condition of early stopping is verfied stop learning
                if early_stopper.check_early_stop_condition(e, e_train, e_val_curr, min_error, flag):
                    print(f"Early stopping condition met at epoch {e}.")
                    stopped_epoch = e
                    break        

        # update best network
        if best_net != None:
            self = copy.deepcopy(best_net)
            
        print(f"Total time: {tot_time:.2f}s")

        report = {
            "stop": stopped_epoch,
            "epoch_best": epoch_best_net,
            "Accuracy_train": train_ac_list,
            "Accuracy_val": val_ac_list,
            "Loss_train": train_loss_list,
            "Loss_val": val_loss_list,
            "f1-score_t": train_f1_list,
            "f1-score_v": val_f1_list,
            "Time": f"{tot_time:.2f}"
        }
    
        return report

    def evaluate_model(self, test, target, target_names):
        """
        This function make a prediction for each digit in the test data (test) and compare it against its label (target).

        Attributes:
        ---------
            test (matrix): a matrix of shape (d, N) where d is the number of feature and N the number of samples.
            target (matrix): a matrix of shape (c, N) where c is the number of classes (i.e 10) and N is the number of samples.
            target_names (list): a list of labels. Each element is a string that should represents the label of a certain class.
        
        Returns:
        --------
            accuracy (float): the fraction of digits for which the right number was predicted.
            cm (matrix): confusion matrix.
            report (dict): a dictionary which contain for each class/row the precision, recall, f1-score and support.

        Errors:
        --------
            ValueError if data or target aren't numpy arrays or if their shape is not 2.
        """
        if not isinstance(test, np.ndarray) or len(test.shape) != 2:
            raise ValueError(f"Test must be a numpy array with a shape of (d, N)!")
        if not isinstance(target, np.ndarray) or len(target.shape) != 2:
            raise ValueError(f"Target must be a numpy array with a shape of (c, N)!")
         
        # get the prediction and the probabilities
        y_pred_test = self.__forward_propagation(test)
        z_pred_test= ActivationFunctions.softmax(y_pred_test)
        # get the class for each sample
        y_pred_classes = z_pred_test.argmax(0)
        target_classes = target.argmax(0)

        accuracy = self.__compute_accuracy(y_pred_test, target)
        cm = confusion_matrix(y_true=target_classes, y_pred=y_pred_classes, labels=np.arange(10))
        report = classification_report(y_true=target_classes, y_pred=y_pred_classes, target_names=target_names)

        return accuracy, cm, report

    def predict(self, input):
        """
        This function makes a single prediction given a certain input.

        Attributes:
        ----------
            input (matrix): The single sample, which is organized as matrix of shape (d, 1) where 'd' is the number of feature.

        Returns:
        --------
            prediction (matrix): a matrix of shape (c, 1) where 'c' is the number of classes.
            Each row represents the probability of belonging to a certain class.
        """
        if len(input.shape) != 2 or input.shape[1] != 1 or not isinstance(input, np.ndarray):
            raise ValueError("Input matrix must have a shape of (d, 1) where 'd' is the number of features!")

        y_pred = self.__forward_propagation(input)
        # use softmax in order to have the probabilities
        y_pred = ActivationFunctions.softmax(y_pred)
        
        return y_pred
    
    def save_model(self, filepath='model'):
        """
        Save the neural network in 'filepath'.
        """
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        filepath = filepath + '/model.pkl'
        model_data = {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'n_layers': self.n_layers,
            'm_hidden_neurons_list': self.m_hidden_neurons_list,
            'activation_list': [activation.__name__ for activation in self.activation_list],
            'error_function': self.error_function.__name__,
            'weights': self.weights,
            'biases': self.biases
        }
        
        # Save the model
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved at {filepath}")

    @staticmethod
    def load_model(filepath='model/model.pkl'):
        """
        Load the model in a neural network object.
        Return a NeuralNetwork object.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
    
        # create a new model
        nn = NeuralNetwork(
                        input_size=model_data['input_size'], 
                        output_size=model_data['output_size'], 
                        n_hidden_layers=model_data['n_layers'] - 1, 
                        m_neurons_list=model_data['m_hidden_neurons_list'], 
                        activation_list=[getattr(ActivationFunctions, name) for name in model_data['activation_list']], 
                        error_function=getattr(Loss, model_data['error_function'])
                    )
        nn.weights = model_data['weights']
        nn.biases = model_data['biases']
        
        return nn

    ###############################################################################################################
    #                                                 Learning                                                    #
    ###############################################################################################################

    def __learning(self, x_train, y_train, learning_rate, momentum, mode, num_mini_batches):
        """Apply the learning process based on mode's value"""
        if mode == 'batch':
            return self.__batch_learning(x_train, y_train, learning_rate, momentum)
        elif mode == 'online':
            return self.__online_learning(x_train, y_train, learning_rate, momentum)
        else:
            return self.__mini_batch_learning(x_train, y_train, learning_rate, momentum, num_mini_batches)

    def __batch_learning(self, x_train, y_train, learning_rate, momentum):
        # no shuffle since at each epoch it applies the same operations to the whole dataset
        weights_deriv, bias_deriv = self.__back_propagation(x_train, y_train)
        self.__gradient_descent(learning_rate, weights_deriv, bias_deriv, momentum)

    def __online_learning(self, x_train, y_train, learning_rate, momentum):
        """Apply the back-prop and update weights for each sample."""
        # get number of samples
        N = x_train.shape[1]
        # create a list of samples 
        SAMPLES = [[i] for i in np.arange(N)]
        # shuffle them
        SAMPLES = np.random.permutation(SAMPLES)
        for sample in SAMPLES:
            x_sample = x_train[:, sample].reshape(-1, 1)
            y_sample = y_train[:, sample].reshape(-1, 1)

            weights_deriv, bias_deriv = self.__back_propagation(x_sample, y_sample)
            self.__gradient_descent(learning_rate, weights_deriv, bias_deriv, momentum)

    def __mini_batch_learning(self, x_train, y_train, learning_rate, momentum, num_mini_batches):
        """Apply the back-prop and update weights for each batch."""
        from tqdm import tqdm
        print("")
        # get the number of batches
        n_batches = Utility.split_batch(y_train, num_mini_batches)
        bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
        # for each mini-batch apply the learning phase
        with tqdm(total=len(n_batches), desc='Processing mini-batches', bar_format=bar_format, mininterval=0.1) as pbar:
            for batch in n_batches:
                x_batch = x_train[:, batch]
                y_batch = y_train[:, batch]
                # get partial derivative
                weights_deriv, bias_deriv = self.__back_propagation(x_batch, y_batch)
                # update weights and biases
                self.__gradient_descent(learning_rate, weights_deriv, bias_deriv, momentum)
                # Update the progress bar
                pbar.update(1)
    
    ###############################################################################################################
    #                                               Computations                                                  #
    ###############################################################################################################

    def __forward_propagation(self, x):
        """
        The function apply the forward propagation (only for output).
        
        Parameters:
        ----------
            x (matrix): Input matrix of shape (d, N) where 'd' is the number of features and N is the number of samples.

        Returns:
        ---------
            z (matrix): Output matrix of shape (c, N) where 'c' is the number of classes and N is the number of samples. 
        """
        # copy input 
        z = x

        for l in range(self.n_layers):
            a = np.matmul(self.weights[l], z) + self.biases[l]
            z = self.activation_list[l](a)
        return z

    def __forward_propagation_training(self, x):
        """
        The function apply the forward propagation (only for training).
        
        Parameters:
        ----------
            x (matrix): Input matrix of shape (d, N) where 'd' is the number of feature and N is the number of samples.

        Returns:
        ---------
            z_layer (list): List of output matrix for each layer, 
            which means that each z.shape is (m, N) where m is number of neuron and N number of samples. 
            z_derivative (list): A list of the derivatives of activation funcions computed into the input of each layer.
        """
        # init the lists
        z_layer = []
        z_derivative = []

        # First layer contains data points.
        # Data point can't change (but are in fact used to train the network by adjusting the weights and biases), 
        # this layer can be considered as an activation layer. Therefore append input to z_layer
        z_layer.append(x)

        for l in range(self.n_layers):
            a = np.matmul(self.weights[l], z_layer[l]) + self.biases[l]
            g = self.activation_list[l](a)
            g_der = self.activation_list[l](a, 1)
        
            z_layer.append(g)
            z_derivative.append(g_der)

        return z_layer, z_derivative

    def __back_propagation(self, x, target):
        """
        Apply back-propagation algorithm, which include 3 steps:
            1) Compute forward propagation
            2) Compute delta of output and hidden nodes (which include the derivative of each z obtained from forward training)
            3) Compute partial derivatives

        Parameters:
        ------------
            x (matrix): Input matrix of shape (d, N) where 'd' is the number of feature and N is the number of samples.
            target (matrix): Targets matrix of shape (c, N) where c is the number of target values (i.e, it corresponds to the number
            of output neurons) and N is the number of samples.

        Returns:
        ---------
            weights_deriv (list): list of partial derivates of weights
            biases_deriv (list): list of partial derivates of biases
        """
        z_layer, z_derivative = self.__forward_propagation_training(x)
        delta_list = self.__compute_delta(z_layer, z_derivative, target)
        weights_deriv, biases_deriv = self.__compute_derivatives(delta_list, z_layer)

        return weights_deriv, biases_deriv

    def __compute_delta(self, z_layer, z_derivative, target):
        """
        Compute delta.
        Starting from the last layer:
            1) Compute the delta of the output layer and insert it into the list (pos 0)
            2) Compute the delta of hidden(s) layer and insert it into the list (first position) 
                - Therefore, if there were more hidden layers, we would have an ordered list of deltas.
        """
        delta = []

        # output layer
        z_last_layer = z_layer[-1]
        cost_function_der = self.error_function(z_last_layer, target, 1)
        der_output = z_derivative[-1]
        delta.append(cost_function_der * der_output)

        # hidden layers
        for l in range(self.n_layers - 1, 0, -1):
            # w^(l3)^T * delta^(l3) * sigmoide_der(z^l2)
            curr_delta = z_derivative[l - 1] * np.matmul(self.weights[l].transpose(), delta[0])
            delta.insert(0, curr_delta)

        return delta

    def __compute_derivatives(self, delta, z):
        """
        This function represents the last step of the back-propagation algorithm.
        It computes the partial derivatives: ∂E/∂w_ij = delta_i * z_j,
        where delta_i is the of the node 'i' and z_j is the output of node 'j'.

        Parameters:
        ----------
            delta (list): the list of the the errors of each layer.
            z (list): the list of matrices where each element represents the output of each layer.
        
        Returns:
        ---------
            weights_deriv (list): a list of derivatives of weights
            biases_deriv (list): a list of derivatives of biases
        """
        weights_deriv=[]
        biase_deriv=[]
        
        for l in range(self.n_layers):
            der_c = np.matmul(delta[l], z[l].transpose())
            weights_deriv.append(der_c)
            biase_deriv.append(np.sum(delta[l], 1, keepdims=True))

        return weights_deriv, biase_deriv

    def __gradient_descent(self, learning_rate, weights_derivative, biases_derivative, momentum):
        """
        The function apply the gradient descent.
        When momentum = 0.0 update rule is: w = w - learning_rate * g where 'g' is the gradient (i.e weights and bias derivative).
        When momentum >= 0.0 update rule is:
            i) velocity = momentum * velocity - learning_rate * g
            ii) w = w + velocity

        Parameters:
        ---------
            learning_rate (float):
            weights_derivative (list):
            biases_derivative (float):
            momentum (float): float hyperparameter >= 0 that accelerates gradient descent in the relevant direction and dampens oscillations. 
            0 is vanilla gradient descent. Defaults to 0.0.

        """

        if momentum == 0:
            # standard gradient descent
            for i in range(self.n_layers):
                self.weights[i] = self.weights[i] - (learning_rate * weights_derivative[i])
                self.biases[i] = self.biases[i] - (learning_rate * biases_derivative[i])
        else:
            velocity_w = [0] * self.n_layers
            velocity_b = [0] * self.n_layers

            for i in range(self.n_layers):
                velocity_w[i] = (momentum * velocity_w[i]) - (learning_rate * weights_derivative[i])
                velocity_b[i] = (momentum * velocity_b[i]) - (learning_rate * biases_derivative[i])
                self.weights[i] += velocity_w[i]
                self.biases[i] += velocity_b[i]

    ###############################################################################################################
    #                                                   Support                                                  #
    ###############################################################################################################

    def __compute_accuracy(self, y_net, target):
        """
        This function computes the accuracy of the neural network's predictions.

        Parameters:
        -----------
            y_net (matrix): the output of the network, which is matrix with a shape of (c, N) where 'c' is the number of 
            target classes and 'N' is the number of samples.
            target (matrix): the gold labels in one-hot encoded format.

        Returns:
        ----------
            Accuracy (float): a value between 0 and 1.
        """
        N = target.shape[1]
        # get probabilities
        z_net= ActivationFunctions.softmax(y_net)
        # For each example get the row with maximum value of the prediction and compares it to the gold label.
        # Then sum all correct prediction and divides by the number of examples N.
        return ((z_net.argmax(0) == target.argmax(0)).sum())/N

    def __get_train_val_loss(self, y_pred_train, y_train, y_pred_val, y_val, train_loss_list, val_loss_list):
        # training loss
        train_loss = self.error_function(y_pred_train, y_train)
        train_loss_list.append(train_loss)
        # validation loss
        if y_val is not None:
            val_loss = self.error_function(y_pred_val, y_val)
            val_loss_list.append(val_loss)

        return train_loss_list, val_loss_list

    def __get_train_val_accuracy(self, y_pred_train, y_train, y_pred_val, y_val, train_ac_list, val_ac_list):
        # training accuracy
        train_accuracy = self.__compute_accuracy(y_pred_train, y_train)
        train_ac_list.append(train_accuracy)
        # validation accuracy
        if y_val is not None:
            loss_accuracy = self.__compute_accuracy(y_pred_val, y_val)
            val_ac_list.append(loss_accuracy)

        return train_ac_list, val_ac_list

    def __get_train_val_f1_score(self, y_pred_train, y_train, y_pred_val, y_val, train_f1_list, val_f1_list, f1_avg_type):
        # training f1-score
        train_f1_list.append(f1_score(y_true=y_train.argmax(0), y_pred=ActivationFunctions.softmax(y_pred_train).argmax(0), average=f1_avg_type))
        # validation accuf1-score
        if y_val is not None:
            val_f1_list.append(f1_score(y_true=y_val.argmax(0), y_pred=ActivationFunctions.softmax(y_pred_val).argmax(0), average=f1_avg_type))

        return train_f1_list, val_f1_list

    def __train_validation(self, x_train, y_train, x_val, y_val, epochs, 
                         learning_rate, momentum, mode, num_mini_batches):
        """
        This function do validation for all inputs of train function.
        It raise a ValueError if any inputs are not of the right type or do not have legal values.
        """
        arrays_to_validate = [
            (x_train, 'Training set', '(d, N)', '(d) is the number of features and (N) is the number of samples!'),
            (y_train, 'Training labels', '(c, N)', '(c) is the number of target classes and (N) is the number of samples!')
        ]

        # Only validate x_val and y_val if they are not None
        if x_val is not None and y_val is not None:
            arrays_to_validate.extend([
                (x_val, 'Validation set', '(d, N)', '(d) is the number of features and (N) is the number of samples!'),
                (y_val, 'Validation labels', '(c, N)', '(c) is the number of target classes and (N) is the number of samples!')
            ])

        for arr, name, expected_shape, msg in arrays_to_validate:
            if not isinstance(arr, np.ndarray) or len(arr.shape) != 2:
                raise ValueError(f"{name} must be a numpy array with a shape of length {expected_shape} where {msg}!")

        if not isinstance(epochs, int) or (epochs <= 0):
            raise ValueError("Epochs must be an integer value > 0 ...")

        if not isinstance(learning_rate, float) or not (0 < learning_rate <= 1):
            raise ValueError("Learning rate must be a float value in [0, 1] ...")
        
        if not isinstance(momentum, float) or (momentum < 0):
            raise ValueError("Momentum must be a float value >= 0 ...")
        
        if not isinstance(mode, str):
            raise ValueError("Learning 'mode' must a string value. Possible values are 'batch', 'online' or 'mini-batch' ...")
        
        if not isinstance(num_mini_batches, int) or (num_mini_batches < 0):
            raise ValueError("Num_mini_batches must be an integer value >= 0 ...")