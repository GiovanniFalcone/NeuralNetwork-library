import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
# import mnist dataset
from dataset.dataset import Dataset
# custom library
from utility.utility import Utility
from network.neural_network import NeuralNetwork

# test the model saved in "path"
def testing(net, path, X_test, Y_test, path_save, model_name, csv_path):
    # create dir if it doesn't exists
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    # if net=None then we want to testing without train again the network (i.e without calling the function above)
    if net is None: net = NeuralNetwork.load_model(path)

    # get random element from test set
    if not csv_path:
        # if you want to test all models don't print the folowing rows:
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
    print(f"Report:\n{report}")

    # plot confusion matrix
    display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=np.arange(10))
    display.plot()
    plt.title("Confusion matrix")
    plt.savefig(path_save + '/' + model_name + '_confusion_matrix' + '.png')
    plt.close()

    # save report
    with open(path_save + '/' + model_name + '_report.txt', 'w') as file:
        file.write(report)

    # update csv file with accuracies
    if csv_path:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Scrivi l'intestazione se il file non esiste
            if not file_exists:
                writer.writerow(["Accuracy"])
            # Aggiungi i dati del modello
            writer.writerow([accuracy])

# get dataset
X_train, Y_train, X_test, Y_test = Dataset.load_mnist()

# Test one model
#   - load model from ./evaluation/mode/Mid/model.pkl
#   - create a dir ./plot/mode/test/Mid
#   - save model and plot in that directory

# path_load = "./evaluation/batch/23/model.pkl"
# path_save = "./plot/batch/test"
# testing(None, path_load, X_test, Y_test, path_save, 'M23', None)

# Test all models
csv_path = "./plot/batch/test/refine/Accuracies_test.csv"

# max id of folder (e.g batch = 59 + 1)
id = 87 # change id
for i in range(60, id):
    path_load = f"./evaluation/batch/refine/{i}/model.pkl"
    path_save = "./plot/batch/test/refine"
    model_name = f"M{i}"
    print(f"Testing model {model_name}...")
    testing(None, path_load, X_test, Y_test, path_save, model_name, csv_path)
