import json
from colorama import init, Fore, Back, Style
from tensorflow.keras.optimizers import Adam, Adamax, Adadelta, Adagrad, Ftrl, Nadam, RMSprop, SGD


class Misc:

    def json_to_dict(self, config_file):
        # Opening JSON file
        with open(config_file) as json_file:
            dictionary = json.load(json_file)
        return dictionary

    def check_key(self, dict, list_parameters):
        for param in range(0, len(list_parameters)):
            if list_parameters[param] in dict.keys():
                pass
            else:
                print(self.log_msg("ERROR", " The following parameter is not found in the configuration file: " +
                                   list_parameters[param]))
                exit(-1)
        return True

    def log_msg(self, level, message):
        init(autoreset=True)
        if level == 'WARNING':
            return Fore.YELLOW + message
        elif level == 'ERROR':
            return Fore.RED + message
        elif level == 'INFO':
            return Style.RESET_ALL + message

    def conf_parse(self, dict):
        # These parameters are compulsory in the config file
        conf_main_param = ['path', 'dataset', 'model_config', 'gpu']
        dataset_param = ['name', 'data_representation', 'default_null_value', 'train_dataset', 'test_dataset',
                         'validation_dataset']
        model_param = ['model', 'train', 'lr', 'epochs', 'batch_size', 'optimizer']
        model_param_pos = ['model', 'train', 'lr', 'epochs', 'batch_size', 'optimizer']
        model_gan = ['model', 'train', 'epochs', 'batch_size', 'num_fake_samples', 'gan']
        data_aug = ['distance_rsamples', 'iterations']

        # Check if all the main parameters are in the config file
        if self.check_key(dict, conf_main_param):
            pass

        # Datasets parameters
        for data in dict['dataset']:
            if self.check_key(data, dataset_param):
                pass

        # for data in dict['data_augmentation']:
        #     if self.check_key(data, data_aug):
        #         pass

        # Model parameters
        for data in dict['model_config']:

            if data['model'] == 'positioning':
                if self.check_key(data, model_param_pos):
                    pass
            elif data['model'] == 'gan':
                if self.check_key(data, model_gan):
                    pass
            else:
                if self.check_key(data, model_param):
                    pass

    def get_datasets_availables(self, dict):
        list_datasets = []
        for data in dict['dataset']:
            list_datasets.append(data['name'])
        return list_datasets

    def optimizer(self, opt, lr):
        if opt == 'Adam':
            return Adam(lr)
        elif opt == 'Adamax':
            return Adamax(lr)
        elif opt == 'Adadelta':
            return Adadelta(lr)
        elif opt == 'Adagrad':
            return Adagrad(lr)
        elif opt == 'Ftrl':
            return Ftrl(lr)
        elif opt == 'Nadam':
            return Nadam(lr)
        elif opt == 'RMSprop':
            return RMSprop(lr)
        elif opt == 'SGD':
            return SGD(lr)
        else:
            return Adam(lr)


