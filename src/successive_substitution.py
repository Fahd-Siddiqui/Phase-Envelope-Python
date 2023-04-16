import numpy as np

from src.calculator import Calculator
from src.eos import EOS


class SuccessiveSubstitution:
    def __init__(self, iterations_max, tol, diff):
        self.iterations_max = iterations_max
        self.tol = tol
        self.diff = diff

    def run(self, step, temperature, acentric, Tc, ac, comp, z, b, kij, lij, composition, pressure, amix, bmix, phase, k_factors, F, temperature_diff):
        it = 0
        temperature = self.get_initial_temperature_guess(temperature, pressure, comp, z, phase, b, amix, bmix, acentric, Tc, ac, kij, lij)

        while np.abs(step[0]) > self.tol and it < self.iterations_max:
            it = it + 1
            temperatre_old = temperature

            composition[:, 1] = composition[:, 0] * k_factors
            F[0] = sum(composition[:, 1]) - 1.0
            temperature_diff[0] = 0.0

            # Numerical Derivative With Respect to Temperature
            a = EOS.eos_parameters(acentric, Tc, ac, temperature)
            temperature_diff = self.calculate_temperature_diff(temperatre_old, self.diff, acentric, Tc, ac, comp, a, b, kij, lij, composition, pressure, amix, bmix, phase, k_factors, temperature_diff)

            temperature_diff[0] = temperature_diff[0] / (2.0 * self.diff)

            # Temperature Step Calculation
            step[0] = F[0] / temperature_diff[0]

            # Step Brake
            if np.abs(step[0]) > 0.25 * temperatre_old:
                step[0] = 0.25 * temperatre_old * step[0] / np.abs(step[0])

            # Updating Temperature
            temperature = temperatre_old - step[0]

            # Updating K-factors
            a = EOS.eos_parameters(acentric, Tc, ac, temperature)
            k_factors = self.update_k_factors(temperature, pressure, comp, a, b, kij, lij, composition, amix, bmix, phase)

        if it == self.iterations_max and np.abs(step[0]) > self.tol:
            raise Exception("In Successive Substitution Method - Maximum Number of Iterations Reached")

        return temperature, k_factors

    @staticmethod
    def get_initial_temperature_guess(temperature: float, P: float, comp, z, phase, b: np.ndarray, amix, bmix, acentric, Tc, ac, kij: np.ndarray, lij: np.ndarray):
        temperature_old = temperature - 1.0
        while temperature_old != temperature:
            temperature_old = temperature

            a = EOS.eos_parameters(acentric, Tc, ac, temperature)  # Updating Attractive Parameter
            amix[0], bmix[0] = EOS.VdW1fMIX(comp, a, b, kij, lij, z)  # Mixing Rule
            amix[1], bmix[1] = amix[0], bmix[0]

            volume = EOS.Eos_Volumes(P, temperature, amix, bmix, phase)
            fug_coef_ref = np.frompyfunc(lambda i: EOS.fugacity(temperature, P, a, b, amix[0], bmix[0], volume[0], z, kij[i, :], lij[i, :], i), 1, 1)(np.arange(comp))
            fug_coef_aux = np.frompyfunc(lambda i: EOS.fugacity(temperature, P, a, b, amix[0], bmix[0], volume[1], z, kij[i, :], lij[i, :], i), 1, 1)(np.arange(comp))

            gibbs_vap = np.sum(z * fug_coef_ref)
            gibbs_liq = np.sum(z * fug_coef_aux)

            # factor 1.75 was proposed by Pedersen and Christensen (Phase Behavior Of Petroleum Reservoir Fluids, 2007 - Chapter 6.5 Phase Identification)
            if (gibbs_liq < gibbs_vap) or (gibbs_liq == gibbs_vap and (volume[0] / bmix[0]) < 1.75):
                temperature = temperature + 10.0

        return temperature

    @classmethod
    def calculate_temperature_diff(cls, T_old, diff, acentric, Tc, ac, comp, a, b, kij, lij, Composition, P, amix, bmix, phase, K, dF):
        for ph in phase:
            amix[ph], bmix[ph] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, ph])

        for sign in [-1, 1]:
            T = T_old + sign * diff
            a = EOS.eos_parameters(acentric, Tc, ac, T)
            fugacity_coef_difference = Calculator.calculate_fugacity_coef_difference(comp, T, P, a, b, amix, bmix, Composition, kij, lij, phase)
            dF[0] += sign * np.sum(Composition[:, 0] * K * (-fugacity_coef_difference))

        return dF

    @classmethod
    def update_k_factors(cls, T, P, comp, a, b, kij, lij, Composition, amix, bmix, phase):
        for ph in phase:
            amix[ph], bmix[ph] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, ph])

        fugacity_coef_difference = Calculator.calculate_fugacity_coef_difference(comp, T, P, a, b, amix, bmix, Composition, kij, lij, phase)
        K = np.exp(-fugacity_coef_difference)
        return K
