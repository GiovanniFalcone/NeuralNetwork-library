import os
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
from network.early_stopping import EarlyStopping

class GridSearch:
    def __init__(self, model_class, param_grid):
        self.model_class = model_class
        self.param_grid = param_grid
        self.results = []

    def train(self, 
              x_train: np.ndarray, 
              y_train: np.ndarray, 
              x_val: np.ndarray, 
              y_val: np.ndarray, 
              epochs: int, 
              mode='batch', 
              num_mini_batches = 32, 
              early_stopper = EarlyStopping(),
              f1_avg_type = None):
        
        # id of combination
        id = 100
        # get hyperparameters of the grid
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        # create the combination
        combinations = list(product(*values))
        # create a dictionary of combinations
        res = {f"params_{i}": dict(zip(keys, combination)) for i, combination in enumerate(combinations)}

        # apply grid search
        for _, params in res.items():
            # creare directory for current combination
            dir = f"./evaluation/{mode}/{id}"
            if not os.path.exists(dir):
                os.makedirs(dir)
    
            # create the model with new hyperparameters (neurons, activations, etc...)
            nn = self.model_class(
                    input_size=x_train.shape[0],
                    output_size=y_train.shape[0],
                    n_hidden_layers=params['n_hidden_layers'],
                    m_neurons_list=[params['m_neurons_list']],
                    activation_list=params['activation_list'],
                    error_function=params['error_function']
                )
            
            # train the model with new hyperparameters
            report = nn.train(
                x_train=x_train, 
                y_train=y_train, 
                x_val=x_val, 
                y_val=y_val, 
                epochs=epochs,
                learning_rate=params['learning_rate'],
                momentum=params['momentum'],
                early_stopper=early_stopper,
                mode=mode,
                f1_avg_type=f1_avg_type
            )

            # plot metrics of the current combination
            self.__plot_metrics(report, dir)

            # add report to the list in order to create a dataframe
            self.__update_dataframe(report, params, id)

            # save the model
            nn.save_model(dir)

            # save metrics over epochs
            self.__save_metrics(report, dir)
            
            # next combination
            id += 1
        
        # create and save dataframe
        df = pd.DataFrame(self.results)
        df.to_excel(f"./evaluation/{mode}/tuning.xlsx", index=False)

        # create heatmap and save it
        self.__create_heatmap_of_combinations(df, mode)
    
    def best_param(self, metric='acc_val'):
        if metric not in ['acc_val', 'acc_train', 'f1_train', 'f1_val']:
            raise ValueError("Metric must be 'acc_val' or 'acc_train' or 'f1_train', 'f1_val'!")
        
        return max(self.results, key=lambda x: x[metric])

    def __save_metrics(self, report, directory):
        """
        Save the metrics in the specified directory as 'name_metric.npz'
        """
        with open(f'{directory}/error.npz', 'wb') as f:
            np.savez(f, error_train=report['Loss_train'], error_val=report['Loss_val'])
        with open(f'{directory}/accuracy.npz', 'wb') as f:
            np.savez(f, accuracy_train=report['Accuracy_train'], accuracy_val=report['Accuracy_val'])
        with open(f'{directory}/f1_score.npz', 'wb') as f:
            np.savez(f, f1_train=report['f1-score_t'], f1_val=report['f1-score_v'])

    def __plot_metrics(self, report, directory):
        """
        Plot accuracy and loss on validation and training set over epochs.
        """
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        # Plot Loss
        ax1.plot(report["Loss_train"], label="Training Loss")
        ax1.plot(report["Loss_val"], label="Validation Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss over Epochs")
        ax1.legend()
        # Plot Accuracy
        ax2.plot(report["Accuracy_train"], label="Training Accuracy")
        ax2.plot(report["Accuracy_val"], label="Validation Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy over Epochs")
        ax2.legend()
        # add space and save it
        plt.tight_layout()
        plt.savefig(f"{directory}/metrics_over_epochs.jpg")
        plt.close(fig)

    def __update_dataframe(self, report, params, id):
        """
        Append to the list the training results of combination 'id' with specified 'params'.
        """
        epoch_best = report['epoch_best']
        self.results.append({
                'id': id,
                'learning_rate': params['learning_rate'],
                'momentum': params['momentum'],
                'n_hidden_layers': params['n_hidden_layers'],
                'm_neurons_list': [params['m_neurons_list']],
                'activation_function': [getattr(act, '__name__', str(act)) for act in params['activation_list']],
                'error_function': getattr(params['error_function'], '__name__', str(params['error_function'])),
                'acc_train': report['Accuracy_train'][epoch_best],  
                'acc_val': report['Accuracy_val'][epoch_best],
                'err_train': report['Loss_train'][epoch_best],
                'err_val': report['Loss_val'][epoch_best],
                'stop': report['stop'],
                'epoch_best': report['epoch_best'],
                'time': report['Time'],
                'f1-score_tr': report['f1-score_t'][epoch_best],
                'f1-score_val': report['f1-score_v'][epoch_best]
            })

    def __create_heatmap_of_combinations(self, df, mode):
        """
        Creates 2 heatmap for each combination of neurons.
            - The first one represents the loss as the learning rate and momentum change
            - The second one represents the accuracy as the learning rate and momentum change.

        It saves the heatmap as 'Heatmap_neurons_(n_neurons,).jpg
        """
         # Convertire le liste in tuple per rendere la colonna hashable
        df['m_neurons_list'] = df['m_neurons_list'].apply(tuple)
        # Ora puoi estrarre i valori unici
        neurons_configs = df['m_neurons_list'].unique()
        # create an heatmap for combination of neurons
        for neurons in neurons_configs:
            # get the rows with specific combination and create a new df
            filtered_data = df[df['m_neurons_list'] == neurons]
            # groups the dataframe using the loss on validation set
            heatmap_data_loss = filtered_data.pivot_table(values='err_val', 
                                                        index='learning_rate', # rows
                                                        columns='momentum')    # columns
            # groups the dataframe using the accuracy on validation set
            heatmap_data_accuracy = filtered_data.pivot_table(values='acc_val', 
                                                            index='learning_rate', 
                                                            columns='momentum')
            plt.figure(figsize=(14, 6))
            # Loss heatmap
            plt.subplot(1, 2, 1)  # 1 riga, 2 colonne, primo subplot
            sns.heatmap(heatmap_data_loss, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Final Validation Loss'})
            plt.title(f"Loss Heatmap - Neurons: {neurons}")
            plt.xlabel("Momentum")
            plt.ylabel("Learning Rate")
            # Accuracy heatmap
            plt.subplot(1, 2, 2)  # 1 riga, 2 colonne, secondo subplot
            sns.heatmap(heatmap_data_accuracy, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Final Validation Accuracy'})
            plt.title(f"Accuracy Heatmap - Neurons: {neurons}")
            plt.xlabel("Momentum")
            plt.ylabel("Learning Rate")
            # save
            plt.tight_layout()
            plt.savefig(f"./evaluation/{mode}/Heatmap_neurons_{neurons}.jpg")
            plt.close()