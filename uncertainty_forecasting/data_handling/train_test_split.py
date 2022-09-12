import numpy as np

from math import floor


def return_train_test_split_indexes(number_of_data,
                                    test_size=0.3,
                                    random_state=0):

    np.random.seed(random_state)

    last_train_index = floor((1.0 - test_size)*number_of_data)

    all_indexes = np.arange(number_of_data)
    np.random.shuffle(all_indexes)

    train_indexes = all_indexes[:last_train_index]
    test_indexes = all_indexes[last_train_index:]

    return train_indexes, test_indexes


