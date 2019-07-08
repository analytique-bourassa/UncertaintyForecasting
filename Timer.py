import time

class Timer:

    def __init__(self,name):
        self.name = name

    def __enter__(self):

        self.t0 = time.time()

    def __exit__(self, *args):
        print('Method: %s | Elapsed time: %0.2fs' % (self.name, time.time() - self.t0))

    def elapsed_time(self):
        return time.time() - self.t0

    