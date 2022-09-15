import numpy as np

KEY_FOR_NUMBER_OF_DATA = 0

def reshape_data_for_LSTM(sequences):
    all_data = sequences[:, np.newaxis, :]
    all_data = np.swapaxes(all_data, axis1=2, axis2=0)
    all_data = np.swapaxes(all_data, axis1=2, axis2=1)

    return all_data

def reshape_into_sequences(data, length_of_sequences):

    assert isinstance(data, np.ndarray)

    n_data = data.shape[KEY_FOR_NUMBER_OF_DATA]

    number_of_sequences = n_data - length_of_sequences + 1

    sequences = list()
    for sequence_index in range(number_of_sequences):
        sequence = data[sequence_index:sequence_index + length_of_sequences]
        sequences.append(sequence)

    return np.array(sequences)

