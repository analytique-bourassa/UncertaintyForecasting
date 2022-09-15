import time

from functools import wraps

from uncertainty_forecasting.utils.validator import Validator
from uncertainty_forecasting.utils.time_profiler_logging import TimeProfilerLogger

class TimerContext:

    def __init__(self, name, show_time_when_exit=True, logger=None):

        Validator.check_type(name, str)
        Validator.check_type(show_time_when_exit, bool)
        if logger is not None:
            Validator.check_type(logger, TimeProfilerLogger)

        self.name = name
        self.show_time_when_exit = show_time_when_exit
        self.logger = logger

    def __enter__(self):

        self.t0 = time.time()
        return self

    def __exit__(self, *args):

        elapsed_time = time.time() - self.t0
        if self.logger is not None:
            self.logger.add_time(self.name, elapsed_time)

        if self.show_time_when_exit:
            print('Method: %s | Elapsed time: %0.2fs' % (self.name, time.time() - self.t0))

    def elapsed_time(self):

        elapsed_time = time.time() - self.t0

        return elapsed_time


def timer_decorator(logger=None):

    if logger is not None:
        Validator.check_type(logger, TimeProfilerLogger)

    def inner_function(function):
        @wraps(function)
        def wrapper(*args, **kwargs):

            start_time = time.time()
            function(*args, **kwargs)
            end_time = time.time()

            elapsed_time = end_time - start_time
            if logger is not None:
                logger.add_time(function.__name__, elapsed_time)

        return wrapper

    return inner_function

