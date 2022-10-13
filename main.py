import argparse
import os
from miscellaneous.misc import Misc
from run_cgan import run_cgan
from run_cnn_lstm import run_cnn_lstm


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--config-file', dest='config_file', action='store', default='', help='Config file')
    p.add_argument('--dataset', dest='dataset', action='store', default='', help='Dataset')
    p.add_argument('--algorithm', dest='algorithm', action='store', default='', help='Algorithm')
    p.add_argument('--method', dest='method', action='store', default='', help='Method GAN')

    args = p.parse_args()

    # Check if the the config file exist
    config_file = str(args.config_file)
    datasets = str(args.dataset)
    algorithm = str(args.algorithm)
    method = str(args.method)

    misc = Misc()

    if config_file == '':
        print(misc.log_msg("ERROR", "Please specify the config file \n"
                                    " e.g., python main.py --config-file config.json \n"
                                    "or \n python main.py --config-file config.json --dataset DSI1,DSI2"))
        exit(-1)

    if os.path.exists(config_file):
        pass
    else:
        print(misc.log_msg("ERROR", "Oops... Configuration file not found. Please check the name and/or path."))
        exit(-1)

    # Config file from .json to dict
    config = misc.json_to_dict(config_file)

    # Check if all the parameters are present in the configuration file
    misc.conf_parse(config)

    # Get all the datasets availables in the config file
    list_datasets = misc.get_datasets_availables(config)

    if datasets != '':
        datasets = datasets.split(',')
        for i, dataset in enumerate(datasets):
            for j in range(0, len(config['dataset'])):
                if dataset == config['dataset'][j]['name']:
                    main_path = config['path']['data_source']
                    if os.path.exists(os.path.join(main_path, dataset,
                                                   config['dataset'][j]['train_dataset'])) and \
                            config['dataset'][j]['train_dataset'] != "":
                        pass
                    else:
                        print(misc.log_msg("ERROR",
                                           "Training set not found."))
                        exit(-1)

                    if os.path.exists(os.path.join(main_path, dataset,
                                                   config['dataset'][j]['test_dataset'])) and \
                            config['dataset'][j]['test_dataset'] != "":
                        pass
                    else:
                        print(misc.log_msg("ERROR",
                                           "Test set not found."))
                        exit(-1)

                    if os.path.exists(os.path.join(main_path, dataset,
                                                   config['dataset'][j]['validation_dataset'])):
                        pass
                    else:
                        print(misc.log_msg("ERROR",
                                           "Validation set not found."))
                        exit(-1)

                    if algorithm == 'CNN-LSTM':
                        run_cnn_lstm(dataset_name=dataset, path_config=config['path'], dataset_config=config['dataset'][j],
                                     building_config=config['model_config'][0], floor_config=config['model_config'][1],
                                     positioning_config=config['model_config'][2], algorithm=algorithm)
                    else:
                        if method != "":
                            if method in ['FLOOR', 'BUILDING', 'FULL-DB']:
                                pass
                            else:
                                print(misc.log_msg("ERROR",
                                                   "Oops this method is not available. The available methods are: "
                                                   "{FLOOR|BUILDING|FULL-DB}"))
                                exit(-1)
                        else:
                            method = "FULL-DB"
                        run_cgan(dataset_name=dataset, path_config=config['path'], dataset_config=config['dataset'][j],
                                 building_config=config['model_config'][0], floor_config=config['model_config'][1],
                                 positioning_config=config['model_config'][2], gan_general_config=config['model_config'][3],
                                 data_augmentation=config['data_augmentation'], algorithm=algorithm, method=method)
    else:
        for dataset in config['dataset']:
            dataset_name = dataset['name']
            if algorithm == 'CNN-LSTM':
                run_cnn_lstm(dataset_name=dataset_name, path_config=config['path'], dataset_config=dataset,
                             building_config=config['model_config'][0], floor_config=config['model_config'][1],
                             positioning_config=config['model_config'][2], algorithm=algorithm)
            else:
                if method != "":
                    if method in ['FLOOR', 'BUILDING', 'FULL-DB']:
                        pass
                    else:
                        print(misc.log_msg("ERROR",
                                           "Oops this method is not available. The available methods are: "
                                           "{FLOOR|BUILDING|FULL-DB}"))
                        exit(-1)
                else:
                    method = "FULL-DB"

                run_cgan(dataset_name=dataset_name, path_config=config['path'], dataset_config=dataset,
                         building_config=config['model_config'][0], floor_config=config['model_config'][1],
                         positioning_config=config['model_config'][2], gan_general_config=config['model_config'][3],
                         data_augmentation=config['data_augmentation'], algorithm=algorithm, method=method)
