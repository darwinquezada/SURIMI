import numpy as np

'''
Developed by Darwin Quezada 
Date: 2021-02-11
Based on: Octave code provided by Joaquín Torres-Sospedra
Repository: rep de 
'''


class DataRepresentation:
    x_train = 0
    x_test = 0
    type = ""
    def_non_det_val = 0
    new_non_det_val = 0

    def __init__(self, x_train=[], x_test=[], x_valid=[], type_rep=None, **kwargs):
        self.x_train = x_train
        self.x_test = x_test
        self.x_valid = x_valid
        self.type = type_rep

        for key, value in kwargs.items():
            if key == "def_no_val":
                self.def_non_det_val = value
            if key == "new_no_val":
                self.new_non_det_val = value

    def data_rep(self):

        if self.def_non_det_val is None or self.def_non_det_val == 0:
            print("No defined a default non detected value")
        else:
            self.x_train, self.x_test, self.x_valid = self.data_new_null_db()

        if self.type == "positive":
            x_training, x_test, x_valid = self.positive_rep()
        if self.type == "powed":
            x_training, x_test, x_valid = self.powed_rep()
        if self.type == "exponential":
            x_training, x_test, x_valid = self.exponential_rep()

        return x_training, x_test, x_valid

    def positive_rep(self):
        if np.size(self.x_valid) != 0:
            min_value = np.min(np.min(np.concatenate((self.x_train, self.x_test, self.x_valid))))
        else:
            min_value = np.min(np.min(np.concatenate((self.x_train, self.x_test))))
        x_training = self.x_train - min_value
        x_test = self.x_test - min_value
        if np.size(self.x_valid) != 0:
            x_valid = self.x_valid - min_value
        else:
            x_valid = self.x_valid
        return x_training, x_test, x_valid

    def powed_rep(self):
        if np.size(self.x_valid) != 0:
            min_value = np.min(np.min(np.concatenate((self.x_train, self.x_test, self.x_valid))))
        else:
            min_value = np.min(np.min(np.concatenate((self.x_train, self.x_test))))
        norm_value = np.power((min_value * (-1)), np.exp(1))
        x_training = np.power((self.x_train - min_value), np.exp(1)) / norm_value
        x_test = np.power((self.x_test - min_value), np.exp(1)) / norm_value
        if np.size(self.x_valid) != 0:
            x_valid = np.power((self.x_valid - min_value), np.exp(1)) / norm_value
        else:
            x_valid = self.x_valid
        return x_training, x_test, x_valid

    def exponential_rep(self):
        if np.size(self.x_valid) != 0:
            min_value = np.min(np.min(np.concatenate((self.x_train, self.x_test, self.x_valid))))
        else:
            min_value = np.min(np.min(np.concatenate((self.x_train, self.x_test))))
        norm_value = np.exp((min_value * -1) / 24)
        x_training = np.exp((self.x_train - norm_value) / 24) / norm_value
        x_test = np.exp((self.x_test - norm_value) / 24) / norm_value
        if np.size(self.x_valid) != 0:
            x_valid = np.exp((self.x_valid - norm_value) / 24) / norm_value
        else:
            x_valid = self.x_valid
        return x_training, x_test, x_valid

    def data_new_null_db(self):
        x_training = self.datarep_new_null(self.x_train, self.def_non_det_val, self.new_non_det_val)
        x_test = self.datarep_new_null(self.x_test, self.def_non_det_val, self.new_non_det_val)

        if np.size(self.x_valid) != 0:
            x_valid = self.datarep_new_null(self.x_valid, self.def_non_det_val, self.new_non_det_val)
        else:
            x_valid = self.x_valid
        return x_training, x_test, x_valid

    def datarep_new_null(self, m, old_null, new_null):
        dif_old_null = np.where(m != old_null, 1, 0)
        eq_old_ull = np.where(m == old_null, 1, 0)
        m1 = m * dif_old_null + new_null * eq_old_ull
        return m1
