import numpy as np
from utils.validator import Validator

def calculate_Kullback_Leibler_divergence_covariance_matrix(covariance_matrix_1, covariance_matrix_2):

    Validator.check_type(covariance_matrix_1, np.ndarray)
    Validator.check_type(covariance_matrix_2, np.ndarray)

    assert len(covariance_matrix_1.shape) == 2, "Convariance matrix must have two axis"
    assert len(covariance_matrix_2.shape) == 2, "Convariance matrix must have two axis"

    Validator.check_matching_dimensions(covariance_matrix_1.shape[0],covariance_matrix_1.shape[1])
    Validator.check_matching_dimensions(covariance_matrix_2.shape[0], covariance_matrix_2.shape[1])
    Validator.check_matching_dimensions(covariance_matrix_1.shape[0], covariance_matrix_2.shape[0])

    vector_space_dimension = covariance_matrix_1.shape[0]
    KL_divergence_term_1 = np.trace(np.dot(np.linalg.inv(covariance_matrix_2), covariance_matrix_1))
    KL_divergence_term_2 = np.log(np.linalg.det(covariance_matrix_2)/np.linalg.det(covariance_matrix_1))

    KL_divergence = 0.5*(KL_divergence_term_1 - vector_space_dimension + KL_divergence_term_2)

    return KL_divergence