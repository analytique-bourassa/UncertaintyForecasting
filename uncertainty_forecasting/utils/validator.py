import torch

NumberTypes = (int, float, complex)


class Validator():

    @staticmethod
    def check_type(value, type_):

        if not isinstance(value, type_):
            raise TypeError("The value {} must be of type {}".format(value, type_))

    @staticmethod
    def check_value_strictly_positive(value):

        if not value > 0:
            raise ValueError("The value {} must be of stictly positive".format(value))

    @staticmethod
    def check_value_is_a_number(value):

        if not isinstance(value, NumberTypes):
            raise TypeError("The value {} must be a number".format(value))

    @staticmethod
    def check_value_is_between_zero_and_one_inclusive(value):

        if not 0 <= value <= 1:
            raise ValueError("The value {} must be between zero and one (inclusive)".format(value))

    @staticmethod
    def check_all_elements_type(value, type):

        if not isinstance(value, list):
            raise TypeError("The value {} must be a list".format(value))

        if not all([isinstance(element, type) for element in value]):
            raise TypeError("All the element of {} must be of type {}".format(value, type))

    @staticmethod
    def check_matching_dimensions(first_dimension, second_dimension):

        if not first_dimension == second_dimension:
            raise ValueError("Mismatch between dimensions provided. {} != {}".format(first_dimension,
                                                                                     second_dimension))

    @staticmethod
    def check_torch_matrix_is_positive(matrix):

        if not isinstance(matrix, torch.Tensor):
            raise TypeError("Matrix {} should be a torch tensor".format(matrix))

        if not len(matrix.shape) == 2:
            raise TypeError("Matrix {} should have two axis".format(matrix))

        if not matrix.shape[0] == matrix.shape[1]:
            raise TypeError("Matrix {} should be two axis of same lenght".format(matrix))

        if not all([element >= 0 for element in matrix.view(-1)]):
            raise ValueError("Matrix {} should be positive".format(matrix))

    @staticmethod
    def check_is_torch_tensor_single_value(value):

        if not isinstance(value, torch.Tensor):
            raise TypeError("Value {} should be a torch tensor".format(value))

        if not (value.shape[0] == 1 and len(value.shape) == 1):
            raise ValueError("Value {} should be a tensor with a single element".format(value))

    @staticmethod
    def check_torch_vector_is_positive(vector):

        if not isinstance(vector, torch.Tensor):
            raise TypeError("Vector {} should be a torch tensor".format(vector))

        if not len(vector.shape) == 1:
            raise TypeError("Vector {} should have one axis".format(vector))

        if not all([element >= 0 for element in vector]):
            raise ValueError("Vector {} should be positive".format(vector))

