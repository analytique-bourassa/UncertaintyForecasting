import torch

from utils.utils import get_test_data_path, get_project_root_directory

from models.LSTM_Bayesian.lstm_parameters import LSTM_parameters
from models.LSTM_Bayesian.training_parameters import TrainingParameters
from models.LSTM_Bayesian.evaluation.evaluation import LSTMModelEvaluator
from models.LSTM_Bayesian.LSTM import LSTM
from models.LSTM_Bayesian.utils import load_checkpoint
from data_handling.preparation.from_numpy_to_torch import DataConverterToTorchTensor

from configurations.data_info import DataInfo
data_info = DataInfo()

PROJECT_PATH = get_project_root_directory()
PATH_DATA = get_test_data_path()

FOLDER_DATA_NUMPY = PROJECT_PATH + "experiments/11_april_2019/"
FOLDER_DATA_ORIGINAL = PROJECT_PATH + "experiments/optimization_params_21_avril_2/"

SAVING_NAME_NUMPY = "data_first_round_experience"
SAVING_PARAMS_FOLDER = FOLDER_DATA_NUMPY
DO_TRAIN_MODEL = False
EXPERIMENT_NAME = "experiment_number_3_of_generation_3"

if __name__=="__main__":

    lstm_params = LSTM_parameters()
    lstm_params.load(filename=EXPERIMENT_NAME + "_lstm_params.json",
                     path=FOLDER_DATA_ORIGINAL + EXPERIMENT_NAME + "/")

    training_params = TrainingParameters()
    training_params.load(filename=EXPERIMENT_NAME + "_training_params.json",
                         path=FOLDER_DATA_ORIGINAL + EXPERIMENT_NAME + "/")

    dataset = DataConverterToTorchTensor(PROJECT_PATH + "experiments/11_april_2019/" + SAVING_NAME_NUMPY + ".npy")
    dataset.load_config(filename=EXPERIMENT_NAME + "_config_numpy_to_torch.json",
                        path=FOLDER_DATA_ORIGINAL + EXPERIMENT_NAME + "/")

    dataset.prepare_data()

    model = LSTM(**lstm_params.as_dict)

    loss_fn = torch.nn.MSELoss(size_average=False)
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=training_params.learning_rate)

    load_checkpoint(model=model,
                    optimizer=optimiser,
                    path=FOLDER_DATA_ORIGINAL + EXPERIMENT_NAME + "/.checkpoints/",
                    name=EXPERIMENT_NAME + "__end")

    model_evaluator = LSTMModelEvaluator()
    if training_params.use_gpu:
        model = model.cuda()

    model_evaluator.data_converter_from_numpy_to_torch = dataset
    model_evaluator.model = model
    model_evaluator.training_params = training_params
    model_evaluator.saving_evaluation_folder = PROJECT_PATH + \
                                               "experiments/" + \
                                               "experiment_23_avril_2019" + \
                                               "/evaluation_ASC02_2/"

    model_evaluator.make_analysis()
    model_evaluator.save_analysis()





