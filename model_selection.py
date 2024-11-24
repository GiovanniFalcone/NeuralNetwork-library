import json
import numpy as np
# import mnist dataset
from dataset.dataset import Dataset
# custom library
from utility.utility import Utility
from network.loss_functions import Loss
from network.grid_search import GridSearch
from network.random_search import RandomSearch
from network.early_stopping import EarlyStopping
from network.neural_network import NeuralNetwork
from network.activation_functions import ActivationFunctions

def grid_search(X_train, Y_train, X_val, Y_val, cv):
    # hyperparam. to evaluate
    param_grid = {
        "n_hidden_layers": [1],
        "activation_list": [[ActivationFunctions.sigmoid, ActivationFunctions.identity]],
        "error_function": [Loss.cross_entropy_softmax],
        "m_neurons_list": [20, 50, 100, 200],
        "learning_rate": [0.0002, 0.0001, 0.00009, 0.00005],
        "momentum": [0.0, 0.0, 0.0]
    }

    # tuning using grid search
    grid_search = GridSearch(NeuralNetwork, param_grid)
    grid_search.train(
        x_train=X_train, 
        y_train=Y_train, 
        x_val=X_val, 
        y_val=Y_val, 
        epochs=500,
        early_stopper = EarlyStopping(),
        mode='batch',
        f1_avg_type='macro'
    )

    # print best score
    print("Best parameters\n{}".format(json.dumps(grid_search.best_param(), indent = 4)))

def random_search(X_train, Y_train, cv):
    """
    Apply random search.
    cv is the number of folds for k-fold cross validation.
    """
    if cv <= 1 or not cv:
        raise ValueError("cv value must be greater than 1...")
    print("Random Search starts...")

    # how many random numbers to generate
    neurons_dimension = 5
    # how many random number for 'eta' and 'momentum' to generate
    other_hyperparam_dimension = 5
    # number of configurations for each value of m_neurons_list
    num_combinations_per_neuron = 2 

    # hyperparam. to evaluate
    param_grid = {
            "n_hidden_layers": 1,
            "activation_list": [ActivationFunctions.sigmoid, ActivationFunctions.identity],
            "error_function": Loss.cross_entropy_softmax,
            "m_neurons_list": [np.random.randint(low=20, high=100) for i in range(5)],
            "learning_rate": [round(np.random.uniform(low=0.0002, high=0.00005), 8) for i in range(other_hyperparam_dimension)],
            "momentum": [round(np.random.uniform(low=0.5, high=0.9), 4) for i in range(other_hyperparam_dimension)]
    }

    # tuning using grid search
    random_search = RandomSearch(NeuralNetwork, param_grid)
    random_search.train(
        Dataset_items=X_train, 
        Dataset_labels=Y_train, 
        epochs=500,
        early_stopper = EarlyStopping(),
        mode='batch',
        f1_avg_type='macro',
        cv=5,
        num_combinations_per_neuron=num_combinations_per_neuron
    )

    # print best score
    print("Best parameters\n{}".format(random_search.best_param()))

# get dataset
X_train, Y_train, X_test, Y_test = Dataset.load_mnist()
Dataset_items = np.concatenate((X_train, X_test), axis=1)
Dataset_labels = np.concatenate((Y_train, Y_test), axis=1)

# grid with cv = 1 example
# X_train, Y_train, X_val, Y_val = Dataset.train_val_split(X_train, Y_train, percentage=0.2, random_state=None)
# grid_search(X_train, Y_train, X_val, Y_val, cv = 1)

# random
random_search(Dataset_items, Dataset_labels, cv=5)
