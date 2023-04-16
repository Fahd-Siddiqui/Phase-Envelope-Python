import numpy


class Utils:
    @classmethod
    def normalize(cls, array: numpy.ndarray) -> numpy.ndarray:
        return array / sum(array)
