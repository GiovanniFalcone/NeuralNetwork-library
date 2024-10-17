"""
Reference: https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
"""

import numpy as np

class EarlyStopping:
    def __init__(self, patience=15, strip=0, alpha=0.1, type_early_stopping='patience'):
        if type_early_stopping not in ['patience', 'GL', 'PQ', 'UP']:
            raise ValueError("Early stopping can assume one of this values: ['patience', 'GL', 'PQ', 'UP']!")
        
        if type_early_stopping == 'patience' and (not isinstance(patience, int) or (patience < 0)):
            raise ValueError("Patience must be an integer value >= 0 ...")
        
        if type_early_stopping != 'patience' and (not isinstance(strip, int) or (strip <= 0)):
            raise ValueError("Strip must be an integer value > 0 ...")
        
        self.__type_early_stopping = type_early_stopping
        self.__patience = patience
        self.__patience_counter = patience
        self.__strip = strip
        self.__alpha = alpha
        self.__strip_errors = []
        self.__P_k = None
        self.__PQ = None
        # inf used only for the first epoch (UP)
        self.__val_error_last_strip = float('inf')
        self.__error_increase_strip = 0

    def get_strip(self):
        return self.__strip
    
    def get_patience_counter(self):
        return self.__patience_counter
    
    def reset_patience_counter(self):
        self.__patience_counter = self.__patience

    def check_early_stop_condition(self, epoch, e_train, e_val_curr, e_val_min, flag):
        # If Progress quotience > alpha returns True and stop learning
        if self.__type_early_stopping == "PQ":
            self.__handle_generalization_loss(e_val_curr, e_val_min)
            return self.__handle_progress_quotient(e_train)
        # If generalization loss > alpha returns True and stop learning
        elif self.__type_early_stopping == "GL":
            return self.__handle_generalization_loss(e_val_curr, e_val_min)
        # if validation error increased for k strip consecutive returns True and stop learning
        elif self.__type_early_stopping == 'UP':
            if (epoch + 1) % self.__strip == 0:
                return self.__handle_UP(e_val_curr)
        # standard early stopping condition (patience)
        else: 
            return self.__handle_patience(flag)

    def __handle_patience(self, flag):
        # the error is increasing
        if flag and self.__patience_counter > 0:
            self.__patience_counter -= 1
        # if 0 stop learning
        return self.__patience_counter == 0
    
    def __handle_UP(self, e_val_curr):
        """
        Stop after epoch 't' iff UP_{s-1} stops after epoch t-k
        and E_va(t) > E_va(t - k)
        """
        if e_val_curr > self.__val_error_last_strip:
            self.__error_increase_strip += 1  
        else:
            self.__error_increase_strip = 0 

        self.__val_error_last_strip = e_val_curr  

        # Stop learning if error increased for too many consecutive strip
        return self.__error_increase_strip >= self.__strip

    def __handle_progress_quotient(self, e_train):
        # we have a list of 'strip' errors
        self.__strip_errors.append(e_train)
        # if list is 'full' we can compute P_k
        if len(self.__strip_errors) == self.__strip:
            # mean train error in k strip
            strip_avg_loss = np.mean(self.__strip_errors)
            # min train error in k strip
            strip_min_loss = np.min(self.__strip_errors)
            self.__P_k = 1000 * ((strip_avg_loss / strip_min_loss) - 1)
            # reset strip
            self.__strip_errors = []  
            # compute PQ
            self.__PQ = self.__GL/self.__P_k 
            # stop learning if PQ > alpha
            if self.__PQ > self.__alpha:
                return True

    def __handle_generalization_loss(self, e_val_curr, e_val_min):
        if e_val_min:
            self.__GL = 100 * (e_val_curr/e_val_min - 1)
            if self.__GL > self.__alpha and self.__type_early_stopping == 'GL':
                return True
