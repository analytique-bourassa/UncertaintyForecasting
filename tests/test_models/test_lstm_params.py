import pytest
import random

from models.lstm_params import LSTM_parameters

INTEGER_GENERATION_UPPER_LIMIT = 100

class TestParams(object):
    """
        self.input_dim = 1
        self.hidden_dim = 5
        self.batch_size = 20
        self.num_layers = 1
        self.bidirectional = False
        self.dropout = 0.4
        self.output_dim = 1

    """
    def test_if_input_dim_not_positive_should_raise_value_error(self):

        # Prepare
        lstm_params = LSTM_parameters()
        invalid_input_dimension = -1*random.randint(0,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        with pytest.raises(ValueError):
            lstm_params.input_dim = invalid_input_dimension

        # Assert

    def test_if_input_dim_positive_integer_should_set_the_value(self):
        # Prepare
        lstm_params = LSTM_parameters()
        valid_input_dimension = random.randint(1,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        lstm_params.input_dim = valid_input_dimension

        # Assert
        assert lstm_params.input_dim == valid_input_dimension

    def test_if_hidden_dim_not_positive_should_raise_value_error(self):

        # Prepare
        lstm_params = LSTM_parameters()
        invalid_hidden_dim = -1*random.randint(0,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        with pytest.raises(ValueError):
            lstm_params.hidden_dim = invalid_hidden_dim

        # Assert

    def test_if_hidden_dim_positive_integer_should_set_the_value(self):
        # Prepare
        lstm_params = LSTM_parameters()
        valid_hidden_dim = random.randint(1,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        lstm_params.hidden_dim = valid_hidden_dim

        # Assert
        assert lstm_params.hidden_dim == valid_hidden_dim

    def test_if_batch_size_not_positive_should_raise_value_error(self):

        # Prepare
        lstm_params = LSTM_parameters()
        invalid_batch_size = -1*random.randint(0,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        with pytest.raises(ValueError):
            lstm_params.batch_size = invalid_batch_size

        # Assert

    def test_if_batch_size_positive_integer_should_set_the_value(self):
        # Prepare
        lstm_params = LSTM_parameters()
        valid_batch_size = random.randint(1,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        lstm_params.batch_size = valid_batch_size

        # Assert
        assert lstm_params.batch_size == valid_batch_size

    def test_if_num_layers_not_positive_should_raise_value_error(self):

        # Prepare
        lstm_params = LSTM_parameters()
        invalid_num_layers = -1*random.randint(0,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        with pytest.raises(ValueError):
            lstm_params.num_layers = invalid_num_layers

        # Assert

    def test_if_num_layers_positive_integer_should_set_the_value(self):
        # Prepare
        lstm_params = LSTM_parameters()
        valid_num_layers = random.randint(1,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        lstm_params.num_layers = valid_num_layers

        # Assert
        assert lstm_params.num_layers == valid_num_layers

    def test_if_bidirectional_not_boolean_should_raise_type_error(self):
        # Prepare
        lstm_params = LSTM_parameters()
        invalid_bidirectional = 0.1

        # Action
        with pytest.raises(TypeError):
            lstm_params.bidirectional = invalid_bidirectional

        # Assert

    def test_if_bidirectional_is_True_should_set_the_value(self):
        # Prepare
        lstm_params = LSTM_parameters()
        valid_bidirectional = True

        # Action
        lstm_params.bidirectional = valid_bidirectional

        # Assert
        assert lstm_params.bidirectional == valid_bidirectional

    def test_if_bidirectional_is_False_should_set_the_value(self):

        # Prepare
        lstm_params = LSTM_parameters()
        valid_bidirectional = False

        # Action
        lstm_params.bidirectional = valid_bidirectional

        # Assert
        assert lstm_params.bidirectional == valid_bidirectional


    def test_if_output_dim_not_positive_should_raise_value_error(self):

        # Prepare
        lstm_params = LSTM_parameters()
        invalid_output_dim = -1*random.randint(0,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        with pytest.raises(ValueError):
            lstm_params.output_dim = invalid_output_dim

        # Assert

    def test_if_output_dim_positive_integer_should_set_the_value(self):
        # Prepare
        lstm_params = LSTM_parameters()
        valid_output_dim = random.randint(1,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        lstm_params.output_dim = valid_output_dim

        # Assert
        assert lstm_params.output_dim == valid_output_dim

    def test_if_dropout_is_smaller_than_zero__should_raise_value_error(self):

        # Prepare
        lstm_params = LSTM_parameters()
        invalid_dropout = -1*random.randint(1,INTEGER_GENERATION_UPPER_LIMIT)

        # Action
        with pytest.raises(ValueError):
            lstm_params.dropout = invalid_dropout

        # Assert

    def test_if_between_0_and_1_should_set_the_value(self):
        # Prepare
        lstm_params = LSTM_parameters()
        valid_dropout = 0.3

        # Action
        lstm_params.dropout = valid_dropout

        # Assert
        assert lstm_params.dropout == valid_dropout