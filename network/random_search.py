import os
import random
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
from network.early_stopping import EarlyStopping
from utility.utility import Utility

class RandomSearch:
    def __init__(self, model_class, param_grid):
        self.model_class = model_class
        self.param_grid = param_grid
        self.param_combinations = set()
        self.results = []

    def train(self, 
              Dataset_items: np.ndarray, 
              Dataset_labels: np.ndarray, 
              epochs: int, 
              mode='batch', 
              num_mini_batches = 32, 
              early_stopper = EarlyStopping(),
              f1_avg_type = None,
              num_combinations_per_neuron=2,
              cv=5):
        
        # id of combination
        id = 200

        # split in cv folds
        folds = Utility.split_batch(Dataset_labels, cv)

        # extract hyperparameters
        m_neurons_list = self.param_grid['m_neurons_list']
        learning_rates = self.param_grid['learning_rate']
        momentums = self.param_grid['momentum']

        # combinations list
        for m_neurons in m_neurons_list:
            for _ in range(num_combinations_per_neuron):
                lr = random.choice(learning_rates)
                mom = random.choice(momentums)
                self.param_combinations.add((m_neurons, lr, mom))

        print(f"Number of folds: {len(folds)}")
        print(f"Number of combinations: {len(self.param_combinations)}")

        # print all configuration
        for idx, params in enumerate(self.param_combinations):
            print(f"Combination {idx+1}: Neurons={params[0]}, Learning Rate={params[1]}, Momentum={params[2]}")

        # for each 
        for idx, params in enumerate(self.param_combinations):
            # creare directory for current combination
            dir = f"./evaluation/random_search/{mode}/{id}"
            if not os.path.exists(dir):
                os.makedirs(dir)

            print(f"Combination {idx+1}: Neurons={params[0]}, Learning Rate={params[1]}, Momentum={params[2]}")

            # list of metrics to update for each fold
            accuracies_t = []
            accuracies_v = []
            score_for_fold = []
            errors_t = []
            errors_v = []
            f1_scores_t = []
            f1_scores_v = []
            time_list = []
            epoch_list = []

            # get hyperparameters
            neurons = params[0]
            lr = params[1]
            mu = params[2] 

            # for plots
            plot_acc_tr = []
            plot_acc_val = []
            plot_err_tr = []
            plot_err_val = []

            # for each fold
            for i in range(len(folds)):
                print(f"Fold: {i + 1}")

                # test fold
                idx_test = folds[i]

                # val fold index
                if i == len(folds) - 1:
                    k = 0       # first fold
                else:
                    k = i + 1   # next fold

                # val fold
                idx_val = folds[k]
                
                # Training folds: concatenates all of them but the i-th fold
                idx_train = np.concatenate([folds[j] for j in range(len(folds)) if (j != i and j != k)])
                print(i, k, [j for j in range(len(folds)) if (j != i and j != k)])
                
                # Crea training set e validation set
                X_train = Dataset_items[:, idx_train]
                Y_train = Dataset_labels[:, idx_train]
                X_val = Dataset_items[:, idx_val]
                Y_val = Dataset_labels[:, idx_val]
                X_target = Dataset_items[:, idx_test]
                Y_target = Dataset_labels[:, idx_test]

                # create model
                nn = self.model_class(
                    input_size=Dataset_items.shape[0],
                    output_size=Dataset_labels.shape[0],
                    n_hidden_layers=self.param_grid['n_hidden_layers'],
                    m_neurons_list=[neurons],
                    activation_list=self.param_grid['activation_list'],
                    error_function=self.param_grid['error_function']
                )

                # training
                res = nn.train(x_train=X_train, 
                                y_train=Y_train, 
                                x_val=X_val, 
                                y_val=Y_val,
                                learning_rate=lr,
                                momentum=mu,
                                epochs=epochs,
                                early_stopper=early_stopper,
                                mode=mode,
                                num_mini_batches=num_mini_batches,
                                f1_avg_type=f1_avg_type
                )

                # evaluate model
                target_names = ['Digit 0', 'Digit 1', 'Digit 2', 'Digit 3', 'Digit 4', 'Digit 5', 
                    'Digit 6', 'Digit 7', 'Digit 8', 'Digit 9',]
                score, _, _ = nn.evaluate_model(X_target, Y_target, target_names)
        
                # Save scores of current model
                e = res['epoch_best']
                accuracies_t.append(res['Accuracy_train'][e])
                accuracies_v.append(res['Accuracy_val'][e])
                errors_t.append(res['Loss_train'][e]) 
                errors_v.append(res['Loss_val'][e]) 
                f1_scores_t.append(res['f1-score_t'][e])
                f1_scores_v.append(res['f1-score_v'][e])
                time_list.append(float(res['Time']))
                epoch_list.append(int(e))
                score_for_fold.append(score)

                # save lists for plotting once all folds are processed
                plot_acc_tr.append(res['Accuracy_train'])
                plot_acc_val.append(res['Accuracy_val'])
                plot_err_tr.append(res['Loss_train'])
                plot_err_val.append(res['Loss_val'])

            # plot metrics of the current combination:
            self.__plot_score(id, score_for_fold, dir)
            self.__plot_folds_metrics(id, accuracies_v, errors_v, dir)
            self.__plot_folds_over_time(id, plot_acc_tr, plot_acc_val, plot_err_tr, plot_err_val, dir, len(folds))

            # add report to the list in order to create a dataframe
            self.results.append({
                'id': id,
                'learning_rate': params[1],
                'momentum': params[2],
                'n_hidden_layers': self.param_grid['n_hidden_layers'],
                'm_neurons_list': [params[0]],
                'activation_function': [getattr(act, '__name__', str(act)) for act in self.param_grid['activation_list']],
                'error_function': getattr(self.param_grid['error_function'], '__name__', str(self.param_grid['error_function'])),
                # accuracy train
                'mean_acc_train': np.mean(accuracies_t),
                'std_acc_train': np.std(accuracies_t),
                # accuracy val
                'mean_acc_val': np.mean(accuracies_v),
                'std_acc_val': np.std(accuracies_v),
                # test set score
                'mean_score': np.mean(score_for_fold),
                'std_score': np.std(score_for_fold),
                # loss train
                'mean_err_train': np.mean(errors_t),
                'std_err_train': np.std(errors_t),
                # loss val
                'mean_err_val': np.mean(errors_v),
                'std_err_val': np.std(errors_v),
                # f1-train
                'mean_f1_train': np.mean(f1_scores_t),
                'std_f1_train': np.std(f1_scores_t),
                # f1-val
                'mean_f1_val': np.mean(f1_scores_v),
                'std_f1_val': np.std(f1_scores_v),
                # epoch
                'mean_epochs': np.mean(epoch_list),
                'std_epochs': np.std(epoch_list),
                'tot_epochs': np.sum(epoch_list),
                # time
                'mean_time': np.mean(time_list),
                'std_time': np.std(time_list),
                'tot_time': np.sum(time_list)
            })
            
            # next combination
            id += 1
        
        # create and save dataframe
        df = pd.DataFrame(self.results)
        df.to_excel(f"./evaluation/random_search/{mode}/tuning.xlsx", index=False)
    
    def best_param(self, metric='mean_score'):
        if metric not in ['mean_acc_val', 'mean_acc_train', 'mean_f1_train', 'mean_f1_val', 'mean_score']:
            raise ValueError("Metric must be 'mean_score', 'mean_acc_val' or 'mean_acc_train' or 'mean_f1_train', 'mean_f1_val'!")
        
        return max(self.results, key=lambda x: x[metric])

    def __plot_folds_over_time(self, id, plot_acc_tr, plot_acc_val, plot_err_tr, plot_err_val, directory, n_folds):
        """
        Plot accuracy and loss on validation and training set over epochs for each fold.
        """
        for fold in range(n_folds):
            # Create a figure with n_folds columns (one for each fold)
            fig, axes = plt.subplots(2, n_folds, figsize=(5 * n_folds, 10))
            fig.suptitle(f"Configuration {id + 1}", fontsize=16, weight="bold") 
            sns.set_style("darkgrid")
            
            for fold in range(n_folds):
                # Plot Loss (Row 0): Loss of current fold
                axes[0, fold].plot(plot_err_tr[fold], label="Training Loss")
                axes[0, fold].plot(plot_err_val[fold], label="Validation Loss")
                axes[0, fold].set_title(f"Fold {fold + 1} - Loss")
                axes[0, fold].set_xlabel("Epochs")
                axes[0, fold].set_ylabel("Loss")
                axes[0, fold].legend()

                # Plot Accuracy (Row 1): Accuracy of current fold
                axes[1, fold].plot(plot_acc_tr[fold], label="Training Accuracy")
                axes[1, fold].plot(plot_acc_val[fold], label="Validation Accuracy")
                axes[1, fold].set_title(f"Fold {fold + 1} - Accuracy")
                axes[1, fold].set_xlabel("Epochs")
                axes[1, fold].set_ylabel("Accuracy")
                axes[1, fold].legend()

            plt.tight_layout()
            plt.savefig(f"{directory}/metrics_over_epochs_all_folds.jpg")
            plt.close(fig)

    def __plot_folds_metrics(self, id, accuracies_v, errors_v, directory):
        data_acc = {
            'Fold': np.arange(1, len(accuracies_v) + 1), 
            'Accuracy': accuracies_v
        }

        data_err = {
            'Fold': np.arange(1, len(errors_v) + 1), 
            'Loss': errors_v
        }

        # Convert dictionaries to DataFrames
        df_acc = pd.DataFrame(data_acc)
        df_err = pd.DataFrame(data_err)

        # Plot definition
        sns.set_style("darkgrid") 
        sns.color_palette("flare")

        # Create a figure with two subplots: one for accuracy and one for loss
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot Validation Accuracy for each fold
        sns.barplot(x='Fold', y='Accuracy', data=df_acc, ax=axes[0], color="steelblue")
        axes[0].set_title(f"Accuracies")
        axes[0].set_xlabel("Fold")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_xticks(range(1, len(accuracies_v) + 1))

        # Plot Validation Loss for each fold
        sns.barplot(x='Fold', y='Loss', data=df_err, ax=axes[1], color="coral")
        axes[1].set_title(f"Loss")
        axes[1].set_xlabel("Fold")
        axes[1].set_ylabel("Loss")
        axes[1].set_xticks(range(1, len(errors_v) + 1))

        # add title and save the plot
        fig.suptitle(f"Metrics Across Folds - Configuration {id + 1}", fontsize=16, fontweight='bold')
        plt.savefig(f"{directory}/config_{id}_fold_metrics.jpg")
        plt.close(fig)

    def __plot_score(self, id, score_for_fold, directory):
        data_acc = {
            'Fold': np.arange(1, len(score_for_fold) + 1), 
            'Accuracy': score_for_fold
        }

        # dictionary to dataframe in order to plot data
        df_acc = pd.DataFrame(data_acc)

        # color
        sns.set_style("darkgrid") 
        sns.color_palette("flare")

        # Create the figure
        plt.figure(figsize=(15, 6))

        # plot the score (on test) for each fold
        sns.barplot(x='Fold', y='Accuracy', data=df_acc, color="steelblue")
        plt.title(f"Score Across Folds - Configuration {id + 1}", fontsize=16, fontweight='bold')  # Titolo
        plt.xlabel("Fold")  
        plt.ylabel("Score")
        
        # save plot
        plt.savefig(f"{directory}/score_config_{id}_fold.jpg")
        plt.close()