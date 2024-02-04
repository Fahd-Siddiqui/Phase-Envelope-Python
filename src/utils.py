import numpy as np
import numpy.typing as npt


class Utils:
    @classmethod
    def normalize(cls, array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return array / sum(array)
