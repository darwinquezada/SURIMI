#necessary Libraries
import numpy as np
import pandas as pd
import scipy.io as sci
import os

# Original datasets path (.mat files)
source_path = os.path.join(os.getcwd(),'original_datasets')

# Target path
target_path = os.path.join(os.getcwd(),'datasets')

# Add here the list of dataset to be converted (the names of the datasets have to be the same as the names of the .mat files).
# Additionally, if a new dataset is added, it have to be added to the config file e.g.,
# {
#    "name": "UJI1",
#    "data_representation": "powed",
#    "default_null_value": 100,
#    "train_dataset": "Train.csv",
#    "test_dataset": "Test.csv",
#    "validation_dataset": ""
#},
list_datasets =  [ 'DSI1', 'DSI2', 'LIB1', 'LIB2', 'MAN1', 'MAN2', 'TUT1', 'TUT2', 'TUT3', 'TUT4', 'TUT5', 'TUT6', 'TUT7','UJI1']

for i in range(0,(len(list_datasets))):
    dataset = list_datasets[i]
    print("------ " + dataset + "------")
    path_csv_datasets = os.path.join(target_path , dataset)

    if not os.path.exists(path_csv_datasets):
        os.makedirs(path_csv_datasets)

    mat = sci.loadmat( os.path.join(source_path, dataset+".mat"))
    trainingMacs = mat['database']['trainingMacs'].item()
    trainingLabels = mat['database']['trainingLabels'].item()

    testMacs = mat['database']['testMacs'].item()
    testLabels = mat['database']['testLabels'].item()
    df_x_train = pd.DataFrame(trainingMacs)
    df_y_train = pd.DataFrame(trainingLabels, columns=['LONGITUDE', 'LATITUDE', 'ALTITUDE', 'FLOOR', 'BUILDINGID'])
    df_x_train['LONGITUDE'] = df_y_train['LONGITUDE']
    df_x_train['LATITUDE'] = df_y_train['LATITUDE']
    df_x_train['ALTITUDE'] = df_y_train['ALTITUDE']
    df_x_train['FLOOR'] = df_y_train['FLOOR']
    df_x_train['BUILDINGID'] = df_y_train['BUILDINGID']

    df_x_train.to_csv(path_csv_datasets + '/' + 'Train.csv', index=False)

    df_x_test = pd.DataFrame(testMacs)
    df_y_test = pd.DataFrame(testLabels, columns=['LONGITUDE', 'LATITUDE', 'ALTITUDE', 'FLOOR', 'BUILDINGID'])
    df_x_test['LONGITUDE'] = df_y_test['LONGITUDE']
    df_x_test['LATITUDE'] = df_y_test['LATITUDE']
    df_x_test['ALTITUDE'] = df_y_test['ALTITUDE']
    df_x_test['FLOOR'] = df_y_test['FLOOR']
    df_x_test['BUILDINGID'] = df_y_test['BUILDINGID']

    df_x_test.to_csv(path_csv_datasets + '/' + 'Test.csv', index=False)
