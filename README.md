<br />
<p align="center"> 
  <h3 align="center">SURIMI: Supervised Radio Map Augmentation with Deep Learning and a Generative Adversarial Network for Fingerprint-based Indoor Positioning</h3>
</p>

```
                          _____                     
 _________        .------|__o__|------.              
:______.-':      :  .--------------.  :             
| ______  |      | :                : |             
|:______o:|      | |  A-WEAR:       | |             
|:______o:|      | |                | |             
|:______o:|      | |  Loading...    | |             
|         |      | |                | |             
|:_____:  |      | |                | |             
|    ==   |      | :                : |             
|       @ |      :  '--------------'  :             
|       o |      :'---...______...---'              
|       o |-._.-i___/' ._______.  \._              
|'-.____o_|   '-.   '-...______...-'  `-._          
:_________:      `.____________________   `-.___.-. 
                 .'.eeeeeeeeeeeeeeeeee.'.      :___:
               .'.eeeeeeeeeeeeeeeeeeeeee.'.         
              :____________________________:

```


<!-- ABOUT THE PROJECT -->
## SURIMI

Indoor Positioning based on Machine Learning has drawn increasing attention both in the academy and the industry as meaningful information from the reference data can be extracted. Many researchers are using supervised, semi-supervised, and unsupervised Machine Learning models to reduce the positioning error and offer reliable solutions to the end-users. In this article, we propose a new architecture by combining Convolutional Neural Network (CNN), Long short-term memory (LSTM) and Generative Adversarial Network (GAN) in order to increase the training data and thus improve the position accuracy. The proposed combination of supervised and unsupervised models was tested in 17 public datasets, providing an extensive analysis of its performance. As a result, the positioning error has been reduced in more than 70% of them.

Authors: Darwin Quezada-Gaibor, Joaquín Torres-Sospedra, Jari Nurmi, Yevgeni Koucheryavy, Joaquín Huerta

### Built With

This framework has been developed using:
* [Python](https://www.python.org/)

## Libraries
* pandas, numpy, seaborn, matplotlib, sklearn, colorama, tensorflow

<!-- structure -->
## Getting Started

    .
    ├── original_datasets                           # Original fingerprinting datasets (.mat files)
    │   ├── README.md                               # Source of datasets
    ├── datasets                                    # Fingerprinting datasets (.csv)
    │   ├── <dataset>                               # Dataset
    │   │   ├── Train.csv
    │   │   ├── Test.csv
    │   │   └── Validation.csv
    ├── miscellaneus
    │   ├── fingerprints_generation.py              # Synthetic fingerprint selection
    │   ├── error_estimation.py                     # Error report
    │   ├── misc.py                                 # Util functions
    │   ├── plot_confusion_matrix.py                # Plot
    │   ├── datasets_mat_to_csv.py                  # File to convert datasets from .mat to .csv
    ├── model
    │   ├── cgan_building.py                        # conditional GAN  - Method 1 
    │   ├── cgan_floor.py                           # conditional GAN  - Method 2
    │   ├── cgan_full_db.py                         # conditional GAN  - Method 3
    │   ├── cnn_lstm.py                             # Positioning model CNN-LSTM 
    ├── preprocessing
    │   ├── data_processing.py                      # Normalization, Standardization, ...
    │   ├── data_representation.py                  # Positive, Powerd, etc.
    ├── results                                     # Positioning results by dataset
    ├── config.json                                 # Configuration file
    ├── main.py                       
    ├── run_cgan.py                                 # Run the data augmentation model
    ├── run_cnn_lstm.py                             # Run postioning model CNN-LSTM
    ├── requirements.txt                            # Python libraries - requirements
    ├── license                                     
    └── README.md                                   # The most important file :)

## Datasets 
The datasets can be downloaded either from authors' repository (see README file in original_datasets folder) or from the following repository:

      "Joaquín Torres-Sospedra, Darwin Quezada-Gaibor, Germán Mendoza-Silva,
      Jari Nurmi, Yevgeny Koucheryavy, & Joaquín Huerta. (2020). Supplementary
      Materials for 'New Cluster Selection and Fine-grained Search for k-Means
      Clustering and Wi-Fi Fingerprinting' (1.0).
      Zenodo. https://doi.org/10.5281/zenodo.3751042"

## Converting datasets from .mat to .csv
1.- Copy the original datasets (.mat) into **original_datasets** folder.

2.- Modify the list of datasets in the /miscellaneous/datasets_mat_to_csv.py file (line 23) with the dataset or datasets to be converted to csv.
```py
list_datasets = [ 'DSI1', 'DSI2', 'LIB1', 'LIB2', 'MAN1', 'MAN2', 'TUT1', 'TUT2', 'TUT3', 'TUT4', 'TUT5', 'TUT6', 'TUT7','UJI1','UTS1', 'UJIB1', 'UJIB2']
```

3.- Run the /miscellaneous/datasets_mat_to_csv.py.
```sh
  python /miscellaneous/datasets_mat_to_csv.py
```
## New datasets
The new datasets have to be added to the config file:
```json
{
  "name": "<dataset name>",
  "data_representation": "powed",
  "default_null_value": <default null value>,
  "train_dataset": "<Training set name>",
  "test_dataset": "<Test set name>",
  "validation_dataset": "<Validation set name>"
},
```
For instance:
```json
{
  "name": "UJIIndoorLoc",
  "data_representation": "powed",
  "default_null_value": 100,
  "train_dataset": "Train.csv",
  "test_dataset": "Test.csv",
  "validation_dataset": "Validation.csv"
},
```

## Datasets structure
The structure of the datasets is:
* AP1, AP2, AP3, ..., APn, LONGITUDE, LATITUDE, ALTITUDE, FLOOR, BUILDINGID

Training and test sets have the same structure.

## Usage
General parameters:
  * --config-file : Datasets and model's configuration (see config.json)
  * -- dataset : Dataset or datasets to be tested (i.e., UJI1 or UJI1,UJI2,TUT1)
  * -- algorithm : {CNN-LSTM|CGAN}
  * -- method : methods availables (e.g., FLOOR, BUILDING, FULL-DB)

1. **Training the CNN-LSTM model**
  * Run the following command:
```sh
  python main.py --config-file config.json --dataset LIB1 --algorithm CNN-LSTM
```

2. **Training the cGAN model**
  * Run the following command to train the cGAN model:

```sh
  python main.py --config-file config.json --dataset LIB1 --algorithm CGAN --method FLOOR
```

Once the models are trained you can change the "train" parameter to "False" in the config file (config.json) in order to use the model saved. If you want to train the models with different hyperparameters you can change them in the config file.
<!-- LICENSE -->
## License

CC By 4.0

<!-- CONTACT -->
## Contact

Darwin Quezada - quezada@uji.es

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

The authors gratefully acknowledge funding from the European Union’s Horizon 2020 Research and Innovation programme under the Marie Sk\l{}odowska Curie grant agreement No. $813278$, A-WEAR.
