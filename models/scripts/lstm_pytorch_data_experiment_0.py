from data_handling.preparation.from_numpy_to_torch import DataConverterToTorchTensor
from utils.utils import get_project_root_directory, get_test_data_path
from models.LSTM_Bayesian.lstm_parameters import LSTM_parameters
from models.LSTM_Bayesian.training_parameters import TrainingParameters
from models.LSTM_Bayesian.meta_objects.experiment import Experiment_LSTM


from data_access.PostgresClient import PostgresClient
from data_handling.preparation.from_database_to_numpy import DataConverterDatabaseToNumpy

from configurations.data_info import DataInfo
data_info = DataInfo()

PROJECT_PATH = get_project_root_directory()
PATH_DATA = get_test_data_path()

FOLDER_DATA = PROJECT_PATH + "experiments/11_april_2019/"

SAVING_NAME_NUMPY = "first_databse_connection_data"
SAVING_PARAMS_FOLDER = PROJECT_PATH + "/models/LSTM_Bayesian/.params/"
DO_TRAIN_MODEL = True

postgres_client = PostgresClient()
postgres_client.host = "172.18.31.10"
postgres_client.table_name = "TEST_merged_dat"
postgres_client.database_name = "postgres"
postgres_client.createsession()

data_converter = DataConverterDatabaseToNumpy(postgres_client)

def return_effort_changed_sign(row):

    if row[data_info.CURRENT_STATE_COLUMN] == data_info.BREAKING_STATE:
        return -1*row[data_info.EFFORT_COLUMN]
    else:
        return row[data_info.EFFORT_COLUMN]


data_converter.dataframe[data_info.EFFORT_COLUMN] = data_converter.dataframe.apply(return_effort_changed_sign, axis=1)
data_converter.dataframe.drop(data_info.CURRENT_STATE_COLUMN, axis=1, inplace=True)
data_converter.save_as_numpy(FOLDER_DATA, SAVING_NAME_NUMPY)

lstm_params = LSTM_parameters()
lstm_params.n_features = 3
lstm_params.batch_size = 1000
training_params = TrainingParameters()
training_params.batch_size = lstm_params.batch_size
training_params.num_epochs = 10000
training_params.checkpoints_intervals = 1000

dataset = DataConverterToTorchTensor(FOLDER_DATA + SAVING_NAME_NUMPY + ".npy")
dataset.index_feature_to_predict = 2
dataset.indexes_to_keep_for_input = [0, 1, 3]

possible_length_sequence = [11, 31, 51]
possible_dropout = [0.1, 0.5]
possible_n_layers = [1,2]
possible_hidden_units = [5, 10]

experiment_number = 1
best_test_loss = 1e10

best_dropout = None
best_length_sequence = None
best_n_layers = None
best_hidden_units = None

def params_to_string(dropout,
                     length_sequence,
                     n_layers,
                     hidden_units, mse_test):
    string = "dropout: {} \n length sequence: {} \n " \
             "n_layers: {} \n hidden_units; {}\n mean_square_error: {} \n".format(
        dropout, length_sequence, n_layers, hidden_units, mse_test)
    return string


for length_sequence in possible_length_sequence:
    for dropout in possible_dropout:
        for n_layer in possible_n_layers:
            for n_hidden_units in possible_hidden_units:

                print("\n Experiment %d \n" % experiment_number)
                dataset.length_of_sequences = length_sequence
                lstm_params.num_layers = n_layer
                lstm_params.h1 = n_hidden_units
                lstm_params.dropout = dropout
                lstm_params.bidirectional = True

                experiment_0 = Experiment_LSTM(name="experiment_%d" % experiment_number)
                experiment_0.lstm_params = lstm_params
                experiment_0.training_params = training_params
                experiment_0.data_converter_to_numpy = data_converter
                experiment_0.data_converter_to_torch = dataset

                experiment_0.create_directory_for_experiment(PROJECT_PATH + "experiments/11_april_2019")
                experiment_0.set_saving_name()
                experiment_0.set_experiment_folder_as_saving_path()
                experiment_0.set_logger()
                experiment_0.log_parameters()

                metrics = experiment_0.run(show_predictions=False,
                                                    return_regression_metrics=True)
                experiment_0.save_experiment_config()

                experiment_number += 1

                if mse_test < best_test_loss:
                    best_dropout = dropout
                    best_length_sequence = length_sequence
                    best_n_layers = n_layer
                    best_hidden_units = n_hidden_units
                    best_test_loss = mse_test

                print(params_to_string(dropout, length_sequence, n_layer, n_hidden_units, mse_test))

print("best params\n")
print("***************")
print(params_to_string(best_dropout, best_length_sequence, best_n_layers, best_hidden_units, best_test_loss))
