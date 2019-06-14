from utils.utils import get_test_data_path, get_project_root_directory

from models.LSTM_Bayesian.lstm_parameters import LSTM_parameters
from models.LSTM_Bayesian.training_parameters import TrainingParameters
from data_handling.preparation.from_numpy_to_torch import DataConverterToTorchTensor
from models.LSTM_Bayesian.meta_objects.optimization.evolutionary_approch.hyperparameters_exploration import HyperparametersExplorator
from configurations.data_info import DataInfo
data_info = DataInfo()

PROJECT_PATH = get_project_root_directory()
PATH_DATA = get_test_data_path()

FOLDER_DATA_NUMPY = "/home/lbourassa/DATA/AI_vehicle_model/"#PROJECT_PATH + "experiments/11_april_2019/"
FOLDER_DATA_ORIGINAL = "/home/lbourassa/DATA/AI_vehicle_model/21_avril_2_generation_member_3/"# #PROJECT_PATH + "experiments/optimization_params_21_avril/"
FOLDER_DATA_OPTIMIZATION = "/home/lbourassa/DATA/AI_vehicle_model/21_avril_2_generation_member_3/"# #PROJECT_PATH + "experiments/optimization_params_25_avril_two_features/"

SAVING_NAME_NUMPY = "data_first_round_experience"
NAME_OF_EXPERIMENT = "experiment_number_3_of_generation_3"

if __name__=="__main__":

    lstm_params = LSTM_parameters()
    lstm_params.load(filename=NAME_OF_EXPERIMENT + "_lstm_params.json",
                     path=FOLDER_DATA_ORIGINAL)

    lstm_params.output_dim = 2

    training_params = TrainingParameters()
    training_params.load(filename= NAME_OF_EXPERIMENT + "_training_params.json",
                         path=FOLDER_DATA_ORIGINAL)

    dataset = DataConverterToTorchTensor(FOLDER_DATA_NUMPY + SAVING_NAME_NUMPY + ".npy")
    dataset.load_config(filename=NAME_OF_EXPERIMENT + "_config_numpy_to_torch.json",
                        path=FOLDER_DATA_ORIGINAL)

    dataset.index_feature_to_predict = [2, 3]
    dataset.prepare_data()

    training_params.num_epochs = 100

    dataset.prepare_data()

    genes_names = ["dropout",
                   "bidirectional",
                   "n_hidden_units",
                   "number_of_layers",
                   "LearningRate",
                   "num_epochs"]

    explorator = HyperparametersExplorator(genes_names)
    explorator.optimization_folder = FOLDER_DATA_OPTIMIZATION
    explorator.set_original_experiment( lstm_params, training_params, dataset)
    explorator.genesis_start_evolution_optimization()







