import numpy as np

from src.constants import Constants
from src.utils import Utils


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
    def van_der_waals_mixing_rule(
            cls,
            a: np.ndarray,
            b: np.ndarray,
            kij: np.ndarray,
            lij: np.ndarray,
            mole_fractions: np.ndarray
    ):
        mole_fractions = Utils.normalize(mole_fractions)

        mole_frac_outer = np.multiply.outer(mole_fractions, mole_fractions)

        # Calculate the outer product of a using broadcasting
        a_outer = a[:, np.newaxis] * a[np.newaxis, :]

        # Combine the multiplication and square root using the distributive property
        mix_factor = np.sqrt(a_outer * (1 - kij) ** 2)
        lij_factor = (b + b)[:, np.newaxis] / 2.0 * (1 - lij)

        amix = np.sum(mole_frac_outer * mix_factor)
        bmix = np.sum(mole_frac_outer * lij_factor)

        return amix, bmix

    @classmethod
    def calculate_mixing_rules(cls, amix, bmix, a, b, kij, lij, mole_fractions, pressure, temperature, phases):
        volumes = np.zeros(len(phases))
        for ph in phases:
            amix[ph], bmix[ph] = EOS.van_der_waals_mixing_rule(a, b, kij, lij, mole_fractions[:, ph])
            volumes[ph] = cls.calculate_eos_volume(pressure, temperature, bmix[ph], amix[ph], phases[ph])

        return volumes

    @classmethod
    def calculate_eos_volume(cls, P: float, T: float, bmix: float, amix: float, root: int) -> float:
        sigma_eos = 1.0 + 2.0 ** 0.5
        epsilon_eos = 1.0 - 2.0 ** 0.5

        # Defining auxiliary variable
        aux = P / (Constants.R * T)

        cubic_equation_coefficients = np.zeros(4)
        cubic_equation_coefficients[0] = 1.0
        cubic_equation_coefficients[1] = (sigma_eos + epsilon_eos - 1.0) * bmix - 1.0 / aux
        cubic_equation_coefficients[2] = sigma_eos * epsilon_eos * bmix ** 2.0 - (1.0 / aux + bmix) * (sigma_eos + epsilon_eos) * bmix + amix / P
        cubic_equation_coefficients[3] = -(1.0 / aux + bmix) * sigma_eos * epsilon_eos * bmix ** 2.0 - bmix * amix / P

        # Cubic equation solver
        volumes = np.roots(cubic_equation_coefficients).real[abs(np.roots(cubic_equation_coefficients).imag) < 1e-3]
        volumes = np.where(volumes < bmix, 1e+16 if root == 1 else 1e-16, volumes)
        return np.min(volumes) if root == 1 else np.max(volumes)

    @classmethod
    def fugacity(cls, T: float, P: float, a: np.ndarray, b: np.ndarray, amix: float, bmix: float, Volume: float, MoleFrac: np.ndarray, kij: np.ndarray, lij: np.ndarray, index: int) -> np.float64:
        sigma_eos = 1.0 + 2.0 ** 0.5  # PR
        epsilon_eos = 1.0 - 2.0 ** 0.5  # PR
        MoleFracAux = Utils.normalize(MoleFrac)

        eos_constant_term = P / (Constants.R * T)

        # Compressibility factor
        z_factor = Volume * eos_constant_term

        # Derivative of amix with respect to MoleFrac(index)
        da_dx = np.matmul((1 - kij), (2.0 * MoleFracAux * ((a[index] * a) ** 0.5)))
        db_dx = np.matmul((1.0 - lij), (MoleFracAux * (b + b[index])))

        # ln(Fugacity coefficient)
        fugacity_coefficient = ((db_dx - bmix) / bmix) * (z_factor - 1.0) - np.log((Volume - bmix) * eos_constant_term) - amix / (bmix * Constants.R * T * (epsilon_eos - sigma_eos)) * (da_dx / amix - (db_dx - bmix) / bmix) * np.log((Volume + epsilon_eos * bmix) / (Volume + sigma_eos * bmix))
        return fugacity_coefficient

    @classmethod
    def fugacity_vec(cls, temperature, pressure, a, b, amix, bmix, volume, mole_fractions, kij, lij):
        return np.frompyfunc(lambda i, phase_number: EOS.fugacity(temperature, pressure, a, b, amix[phase_number], bmix[phase_number], volume[phase_number], mole_fractions[:, phase_number], kij[i, :], lij[i, :], i), 2, 1)
