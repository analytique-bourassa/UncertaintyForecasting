import time

class Timer:

    def __init__(self,name,show_time_when_exit=True):

        self.name = name
        self.show_time_when_exit =show_time_when_exit

    def __enter__(self):

        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        if self.show_time_when_exit:
            print('Method: %s | Elapsed time: %0.2fs' % (self.name, time.time() - self.t0))

    def elapsed_time(self):
        return time.time() - self.t0

    