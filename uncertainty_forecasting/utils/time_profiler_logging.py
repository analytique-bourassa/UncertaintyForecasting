class TimeProfilerLogger(object):

    __instance = None

    @staticmethod
    def getInstance():
        if TimeProfilerLogger.__instance == None:
            TimeProfilerLogger()
        return TimeProfilerLogger.__instance

    def __init__(self):
        """ Virtually private constructor. """

        if TimeProfilerLogger.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            TimeProfilerLogger.__instance = self
            self.times = {}

    def add_time(self, name, time):

        if name in self.times.keys():
            self.times[name] += time
        else:
            self.times[name] = time



