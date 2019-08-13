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
            raise ValueError("Mismatch between dimensions provided. {} != {}".format(first_dimension_one,
                                                                                     second_dimension))



