import json

from utils.utils import get_test_data_path, get_project_root_directory

from models.LSTM_Bayesian.lstm_parameters import LSTM_parameters
from models.LSTM_Bayesian.training_parameters import TrainingParameters
from models.LSTM_Bayesian.meta_objects.experiment import Experiment_LSTM

from data_handling.preparation.from_numpy_to_torch import DataConverterToTorchTensor

from configurations.data_info import DataInfo
data_info = DataInfo()


PROJECT_PATH = get_project_root_directory()
PATH_DATA = get_test_data_path()

FOLDER_DATA_NUMPY = PROJECT_PATH + "experiments/11_april_2019/"
FOLDER_DATA_ORIGINAL = PROJECT_PATH + "experiments/optimization_params_21_avril_2/"
#FOLDER_DATA_OPTIMIZATION = PROJECT_PATH + "experiments/optimization_params_21_avril_2/"

SAVING_NAME_NUMPY = "data_first_round_experience"
NAME_OF_EXPERIMENT = "experiment_number_3_of_generation_3"

NEW_EXPERIMENT_NAME = "experiment_25_avril_2019_two_features"

if __name__=="__main__":

    lstm_params = LSTM_parameters()
    lstm_params.load(filename=NAME_OF_EXPERIMENT + "_lstm_params.json",
                     path=FOLDER_DATA_ORIGINAL + NAME_OF_EXPERIMENT + "/")

    lstm_params.output_dim = 2

    training_params = TrainingParameters()
    training_params.load(filename= NAME_OF_EXPERIMENT + "_training_params.json",
                         path=FOLDER_DATA_ORIGINAL + NAME_OF_EXPERIMENT + "/")

    training_params.checkpoints_saving_path = PROJECT_PATH + "experiments/" + NEW_EXPERIMENT_NAME
    training_params.checkpoints_saving_name = NEW_EXPERIMENT_NAME
    training_params.checkpoints_intervals = 100

    dataset = DataConverterToTorchTensor(FOLDER_DATA_NUMPY + SAVING_NAME_NUMPY + ".npy")
    dataset.load_config(filename=NAME_OF_EXPERIMENT + "_config_numpy_to_torch.json",
                        path=FOLDER_DATA_ORIGINAL + NAME_OF_EXPERIMENT + "/")

    dataset.index_feature_to_predict = [2, 3]
    dataset.prepare_data()

    training_params.num_epochs = 5000

    experiment_to_train = Experiment_LSTM(NEW_EXPERIMENT_NAME)
    experiment_to_train.lstm_params = lstm_params
    experiment_to_train.training_params = training_params
    experiment_to_train.data_converter_to_torch = dataset

    experiment_to_train.create_directory_for_experiment(PROJECT_PATH + "experiments/")
    experiment_to_train.set_saving_name()
    experiment_to_train.set_experiment_folder_as_saving_path()
    experiment_to_train.set_logger()
    experiment_to_train.log_parameters()

    metrics_dict = experiment_to_train.run(return_regression_metrics=True)

    print(metrics_dict)
    experiment_to_train.save_experiment_config()
    print(json.dumps(metrics_dict, indent=2))




