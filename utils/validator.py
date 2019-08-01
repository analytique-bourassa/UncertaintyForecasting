class Validator():

    @staticmethod
    def check_type(value, type_):

        if not isinstance(value, type_):
            raise TypeError("The value {} must be of type {}".format(value, type_))

    @staticmethod
    def check_value_strictly_positive(value):

        if not value > 0:
            raise ValueError("The value {} must be of stictly positivee".format(value))

