import json
# import mnist dataset
from dataset.dataset import Dataset
# custom library
from utility.utility import Utility
from network.loss_functions import Loss
from network.grid_search import GridSearch
from network.early_stopping import EarlyStopping
from network.neural_network import NeuralNetwork
from network.activation_functions import ActivationFunctions

# get dataset
X_train, Y_train, X_test, Y_test = Dataset.load_mnist()
X_train, Y_train, X_val, Y_val = Dataset.train_val_split(X_train, Y_train, percentage=0.2, random_state=None)
Utility.print_info_dataset(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)

# hyperparam. to evaluate
param_grid = {
    "n_hidden_layers": [1],
    "activation_list": [[ActivationFunctions.sigmoid, ActivationFunctions.identity]],
    "error_function": [Loss.cross_entropy_softmax],
    "m_neurons_list": [20, 50, 100, 200, 500],
    "learning_rate": [0.0002, 0.0001, 0.00009, 0.00005],
    "momentum": [0.5, 0.75, 0.9]
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