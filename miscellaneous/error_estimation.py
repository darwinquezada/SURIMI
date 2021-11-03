import pandas as pd
import os
import numpy as np
import logging
from sklearn.metrics import confusion_matrix
from miscellaneous.plot_confusion_matrix import plot_confusion_matrix
from miscellaneous.misc import Misc


def error_estimation(database_name=None, path_config=None, prediction=None, test=None, algorithm=None, **kwargs):

    misc = Misc()
    lati_error = np.mean(np.sqrt(np.square(prediction['LATITUDE'].values[:] - test['LATITUDE'].values[:])))
    long_error = np.mean(np.sqrt(np.square(prediction['LONGITUDE'].values[:] - test['LONGITUDE'].values[:])))
    alti_error = np.mean(np.sqrt(np.square(prediction['ALTITUDE'].values[:] - test['ALTITUDE'].values[:])))

    mean_error = np.mean(np.sqrt(np.square(prediction['LONGITUDE'].values[:] - test['LONGITUDE'].values[:]) +
                                 np.square(prediction['LATITUDE'].values[:] - test['LATITUDE'].values[:]) +
                                 np.square(prediction['ALTITUDE'].values[:] - test['ALTITUDE'].values[:])
                                 ))
    mean_error_dosd = np.mean(np.sqrt(np.square(prediction['LONGITUDE'].values[:] - test['LONGITUDE'].values[:]) +
                                      np.square(prediction['LATITUDE'].values[:] - test['LATITUDE'].values[:])
                                      ))

    if 'conf_augment' in kwargs:
        prediction_path = os.path.join(path_config['results'], database_name, algorithm, kwargs.get("method"),
                                       kwargs.get("conf_augment"))
    else:
        prediction_path = os.path.join(path_config['results'], database_name, algorithm)

    # Save prediction
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    datestr = "%m/%d/%Y %I:%M:%S %p"
    logging.basicConfig(
        filename=prediction_path + '/results.log',
        level=logging.INFO,
        filemode="w",
        datefmt=datestr,
    )
    # CM plot
    cmb = confusion_matrix(y_true=test['BUILDINGID'].values[:], y_pred=prediction['BUILDING'].values[:])
    #labels = np.unique(test['BUILDINGID'].values[:])
    #plot_confusion_matrix(cm=cmb, normalize=False, target_names=labels, title="Prediction Building",
    #                      title_figure="CM")

    test_floor = test.round({'FLOOR': 2})
    prediction_floor = prediction.round({'FLOOR': 2})

    cmf = confusion_matrix(y_true=test_floor['FLOOR'].values[:], y_pred=prediction_floor['FLOOR'].values[:])
    #labels = np.unique(test['FLOOR'].values[:])
    #plot_confusion_matrix(cm=cmf, normalize=False, target_names=labels, title="Prediction Floor",
    #                      title_figure="CM")

    accuracy_building = (np.trace(cmb) / float(np.sum(cmb)))*100
    accuracy_floor = (np.trace(cmf) / float(np.sum(cmf)))*100

    logging.info("------------------ Mean positioning error ------------------------------")
    logging.info("Mean Longitude error: {:.3f}".format(long_error))
    logging.info("Mean Latitude error: {:.3f}".format(lati_error))
    logging.info("Mean Altitude error: {:.3f}".format(alti_error))
    logging.info("Mean positioning 2D error: {:.3f}".format(mean_error_dosd))
    logging.info("Mean positioning 3D error: {:.3f}".format(mean_error))
    logging.info("Building hit rate: {:.2f}%".format(accuracy_building))
    logging.info("Floor hit rate: {:.2f}%".format(accuracy_floor))

    print(misc.log_msg("WARNING", "------------------ Mean positioning error ------------------------------"))
    print(misc.log_msg("WARNING", "Longitude error: {:.3f}".format(long_error)))
    print(misc.log_msg("WARNING", "Latitude error: {:.3f}".format(lati_error)))
    print(misc.log_msg("WARNING", "Altitude error: {:.3f}".format(alti_error)))
    print(misc.log_msg("WARNING", "Mean positioning 2D error: {:.3f}".format(mean_error_dosd)))
    print(misc.log_msg("WARNING", "Mean positioning 3D error: {:.3f}".format(mean_error)))
    print(misc.log_msg("WARNING", "Building hit rate: {:.2f}%".format(accuracy_building)))
    print(misc.log_msg("WARNING", "Floor hit rate: {:.2f}%".format(accuracy_floor)))
