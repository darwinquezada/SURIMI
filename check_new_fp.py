import os

import numpy as np
import argparse
from collections import Counter
from miscellaneous.misc import Misc
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--dataset', dest='dataset', action='store', default='', help='Dataset')
    p.add_argument('--option', dest='option', action='store', default='', help='Option TRAIN/TEST')
    p.add_argument('--algorithm', dest='algorithm', action='store', default='', help='Algorithm')
    p.add_argument('--building', dest='building', action='store', default='', help='Building')
    p.add_argument('--floor', dest='floor', action='store', default='', help='Floor')

    args = p.parse_args()

    # Check if the the config file exist
    floor = int(args.floor)
    building = int(args.building)
    option = str(args.option)
    dataset = str(args.dataset)
    algorithm = str(args.algorithm)

    saved_model_dir = os.path.join('saved_model', dataset, algorithm)
    dataset_dir = os.path.join('dataset', dataset)

    if option == 'TRAIN':
        df_full_real = pd.read_csv(dataset_dir + '/Train.csv')
        df_y_real = df_full_real.iloc[:, -5:].copy()
    else:
        df_full_real = pd.read_csv(dataset_dir + '/Test.csv')
        df_y_real = df_full_real.iloc[:, -5:].copy()

    df_y_fake = pd.read_csv(saved_model_dir + '/TrainingData_y_augmented.csv')

    misc = Misc()
    print(misc.log_msg("WARNING", 'Floor:' + str(Counter(df_y_real['FLOOR']))))
    print(misc.log_msg("WARNING", 'Building: ' + str(Counter(df_y_real['BUILDINGID']))))

    df_y_real = df_y_real[(df_y_real['FLOOR'] == floor) & (df_y_real['BUILDINGID'] == building)].copy()
    df_y_real['REAL-FAKE'] = np.zeros((np.shape(df_y_real)[0], 1))
    df_y_fake_fp = df_y_fake[(df_y_fake['FLOOR'] == floor) & (df_y_fake['BUILDINGID'] == building)].copy()
    df_y_fake_fp['REAL-FAKE'] = np.ones((np.shape(df_y_fake_fp)[0], 1))

    frames = [df_y_fake_fp, df_y_real]

    result = pd.concat(frames)

    result.plot(kind="scatter", x="LONGITUDE", y="LATITUDE", alpha=0.4, figsize=(10, 7),
                c="REAL-FAKE", cmap=plt.get_cmap("jet"), sharex=False)

    plt.show()



