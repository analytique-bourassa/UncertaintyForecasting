import Numbers

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

        if not isinstance(value, Numbers.number):
            raise TypeError("The value {} must be a number".format(value))

    @staticmethod
    def check_value_is_between_zero_and_one_inclusive(value):

        if not 0 <= value <= 1:
            raise ValueError("The value {} must be between zero and one (inclusive)".format(value))


