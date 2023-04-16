import numpy as np

from PhaseEnvelope.src.Constants import Constants
from PhaseEnvelope.src.utils import Utils


class EOS:
    @classmethod
    def eos_parameters(
            cls,
            accentric_factor: np.ndarray,
            critical_temperature: np.ndarray,
            ac: np.ndarray,
            temperature: float,
    ) -> np.ndarray:
        K = 0.37464 + (1.54226 - 0.26992 * accentric_factor) * accentric_factor
        reduced_temperature = temperature / critical_temperature

        alpha = (1 + K * (1 - reduced_temperature ** 0.5)) * (1 + K * (1 - reduced_temperature ** 0.5))

        return alpha * ac

    @classmethod
    def VdW1fMIX(
            cls,
            comp: int,
            a: np.ndarray,
            b: np.ndarray,
            kij: np.ndarray,
            lij: np.ndarray,
            MoleFrac: np.ndarray
    ):
        MoleFracAux = Utils.normalize(MoleFrac)

        amix = np.sum(np.multiply.outer(MoleFracAux, MoleFracAux) * np.sqrt(np.outer(a, a)) * (1 - kij))
        bmix = np.sum(np.multiply.outer(MoleFracAux, MoleFracAux) * np.outer((b + b) / 2.0, np.ones(comp)) * (1 - lij))

        return amix, bmix

    @classmethod
    def Eos_Volumes(cls, P: float, T: float, amix: np.ndarray, bmix: np.ndarray, phase: list) -> np.ndarray:
        num_phases = len(phase)
        Volume = np.zeros(num_phases)
        for i in range(num_phases):
            Volume[i] = cls.EoS_Volume(P, T, bmix[i], amix[i], phase[i])

        return Volume

    @classmethod
    @classmethod
    def EoS_Volume(cls, P: float, T: float, bmix: float, amix: float, root: int) -> float:
        R = Constants.R
        sigma_eos = 1.0 + 2.0 ** 0.5
        epsilon_eos = 1.0 - 2.0 ** 0.5

        # Defining auxiliary variable
        aux = P / (R * T)

        # Cubic equation coefficients
        coefCubic = np.zeros(4)
        coefCubic[0] = 1.0
        coefCubic[1] = (sigma_eos + epsilon_eos - 1.0) * bmix - 1.0 / aux
        coefCubic[2] = sigma_eos * epsilon_eos * bmix ** 2.0 - (1.0 / aux + bmix) * (sigma_eos + epsilon_eos) * bmix + amix / P
        coefCubic[3] = -(1.0 / aux + bmix) * sigma_eos * epsilon_eos * bmix ** 2.0 - bmix * amix / P

        # Cubic equation solver
        Vol = np.roots(coefCubic).real[abs(np.roots(coefCubic).imag) < 1e-3]
        Vol = np.where(Vol < bmix, 1e+16 if root == 1 else 1e-16, Vol)
        return np.min(Vol) if root == 1 else np.max(Vol)

    @classmethod
    def fugacity(cls, T: float, P: float, a: np.ndarray, b: np.ndarray, amix: float, bmix: float, Volume: float, MoleFrac: np.ndarray, kij: np.ndarray, lij: np.ndarray, index: int) -> np.float64:
        sigma_eos = 1.0 + 2.0 ** 0.5  # PR
        epsilon_eos = 1.0 - 2.0 ** 0.5  # PR
        MoleFracAux = Utils.normalize(MoleFrac)

        eos_constant_term = P / (Constants.R * T)

        # Compressibility factor
        Z = Volume * eos_constant_term

        # Deritivative of amix with respect to MoleFrac(index)
        da_dx = np.matmul((1 - kij), (2.0 * MoleFracAux * ((a[index] * a) ** 0.5)))
        db_dx = np.matmul((1.0 - lij), (MoleFracAux * (b + b[index])))

        # ln(Fugacity coefficient)
        FugCoef = ((db_dx - bmix) / bmix) * (Z - 1.0) - np.log((Volume - bmix) * eos_constant_term) - amix / (bmix * Constants.R * T * (epsilon_eos - sigma_eos)) * (da_dx / amix - (db_dx - bmix) / bmix) * np.log((Volume + epsilon_eos * bmix) / (Volume + sigma_eos * bmix))
        return FugCoef

    @classmethod
    def calculate_fugacity_coefs(cls, comp, T, P, a, b, amix, bmix, volume, Composition, kij, lij):
        # TODO Instead of using 0,1 implement Phase
        fug_coef_ref = np.frompyfunc(lambda i: cls.fugacity(T, P, a, b, amix[0], bmix[0], volume[0], Composition[:, 0], kij[i, :], lij[i, :], i), 1, 1)(np.arange(comp))
        fug_coef_aux = np.frompyfunc(lambda i: cls.fugacity(T, P, a, b, amix[1], bmix[1], volume[1], Composition[:, 1], kij[i, :], lij[i, :], i), 1, 1)(np.arange(comp))
        return fug_coef_ref, fug_coef_aux
