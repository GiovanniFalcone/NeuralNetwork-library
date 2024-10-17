import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
# import mnist dataset
from dataset.dataset import Dataset
# custom library
from utility.utility import Utility
from network.loss_functions import Loss
from network.early_stopping import EarlyStopping
from network.neural_network import NeuralNetwork
from network.activation_functions import ActivationFunctions

# list used in order to save the accuracy of each model to test
acc_list = []
file_name = "accuracies.csv"

def train_best_model_obtained_from_tuning(X_Train, Y_Train, path):
    # create dir if it doesn't exists
    if not os.path.exists(path):
        os.makedirs(path)
    # shuffle training set
    random_permutation = np.random.permutation(X_Train.shape[1])
    X_Train = X_Train[:, random_permutation]
    Y_Train = Y_Train[:, random_permutation]
    # initialize the network
    nn = NeuralNetwork(input_size=X_Train.shape[0],
                    output_size=Y_Train.shape[0],
                    n_hidden_layers=1,
                    m_neurons_list=[20],
                    activation_list=[ActivationFunctions.sigmoid, ActivationFunctions.identity],
                    error_function=Loss.cross_entropy_softmax)
    # training
    report = nn.train(x_train=X_Train, 
                    y_train=Y_Train, 
                    x_val=None,
                    y_val=None,
                    learning_rate=0.0002,
                    momentum=0.9,
                    epochs=25,
                    early_stopper=EarlyStopping(),
                    mode='batch')
    # save model
    nn.save_model(path)
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    # Plot Loss
    ax1.plot(report["Loss_train"])
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend()
    # Plot Accuracy
    ax2.plot(report["Accuracy_train"])
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.legend()
    # add space and save it
    plt.tight_layout()
    plt.savefig(f"{path}/metrics_over_epochs.jpg")
    plt.close(fig)

    return nn

# if flag is true -> test all model
# otherwise test the model saved in "path"
def testing(net, path, X_test, Y_test, path_save, model_name, flag):
    # create dir if it doesn't exists
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    # if net=None then we want to testing without train again the network (i.e without calling the function above)
    if net is None: net = NeuralNetwork.load_model(path)

    if not flag:
        # get random element from test set
        elem, label = Utility.get_random_elem(X_test, Y_test)
        print(f"\nElement chosen has [{label}] and has shape {elem.shape}\n")
        # print prediction
        trained_prediction = net.predict(elem)
        print("Prediction from trained network:\n", trained_prediction, "\nNetwork prediction: ", np.argmax(trained_prediction))
        print("=" * 50)

    # evaluation of trained model
    target_names = ['Digit 0', 'Digit 1', 'Digit 2', 'Digit 3', 'Digit 4', 'Digit 5', 
                    'Digit 6', 'Digit 7', 'Digit 8', 'Digit 9',]
    accuracy, confusion_matrix, report = net.evaluate_model(X_test, Y_test, target_names)
    
    print(f"[{model_name}] Accuracy {accuracy}\n")
    # update list when more more than one model needs to be tested
    if flag: acc_list.append(accuracy)
    # print classification report when only one model is tested
    if not flag: print(f"Report:\n{report}")

    # plot confusion matrix
    display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=np.arange(10))
    display.plot()
    plt.title("Confusion matrix")
    plt.savefig(path_save + '/' + model_name + '_confusion_matrix' + '.png')
    plt.close()

    # save report
    with open(path_save + '/' + model_name + '_report.txt', 'w') as file:
        file.write(report)

    # if more than one model needs to be tested save a csv of accuracies
    if flag:
        with open(path_save + '/' + file_name, mode='w', newline='') as file:  
            writer = csv.writer(file)
            # header
            writer.writerow(["Accuracy"])
            # write row by row
            for acc in acc_list:
                writer.writerow([acc])

# get dataset
X_train, Y_train, X_test, Y_test = Dataset.load_mnist()

# Case 1: if you want to train again the network and then test it (you need to set the right hyperparameters!)
#   - create a dir ./plot/mode/Retrained/Mid
#   - save model and plot in that directory

# it's the location where the model will be saved
# path_1 = "./plot/batch/Retrained/M0"
# net = train_best_model_obtained_from_tuning(X_train, Y_train, path_1)
# testing(net, None, X_test, Y_test, path_1, 'M0', False)

####################################################################

# Case 2: Test one model
#   - load model from ./evaluation/mode/Mid/model.pkl
#   - create a dir ./plot/mode/test/Mid
#   - save model and plot in that directory

# path_load = "./evaluation/batch/0/model.pkl"
# path_save = "./plot/batch/test"
# testing(None, path_load, X_test, Y_test, path_save, 'M0', False)

####################################################################

# Case 3: Test all models of a specific learning mode
#   - load all the model from ./plot/mode
#   - create a dir ./plot/mode/Mid/Retrained
#   - save model and plot in that directory

# load dir mode
path_mode = "./evaluation/batch/"
# save dir mode
path_mode_save = "./plot/batch/test"
# (REMEMBER TO CHANGE IDS)
for i in range(0, 60):
    # get the model 'i'
    model_path = os.path.join(path_mode, str(i), "model.pkl")
    testing(None, model_path, X_test, Y_test, path_mode_save, "M" + str(i), True)
