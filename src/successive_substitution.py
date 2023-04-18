import numpy as np

from src.calculator import Calculator
from src.eos import EOS


class SuccessiveSubstitution:
    def __init__(self, max_iterations, tolerance, diff):
        self.iterations_max = max_iterations
        self.tol = tolerance
        self.diff = diff

    def calculate(
            self, 
            step, 
            temperature,
            acentric_factors,
            critical_temperatures,
            ac,
            b,
            kij,
            lij,
            mole_fractions,
            pressure,
            amix,
            bmix,
            phases,
            k_factors,
            vapor_fraction,
            vapor_fraction_differential
    ):
        it = 0

        while np.abs(step[0]) > self.tol and it < self.iterations_max:
            it = it + 1
            temperature_old = temperature

            mole_fractions[:, 1] = mole_fractions[:, 0] * k_factors
            vapor_fraction[0] = sum(mole_fractions[:, 1]) - 1.0
            vapor_fraction_differential[0] = 0.0

            # Numerical Derivative With Respect to Temperature
            a = EOS.eos_parameters(acentric_factors, critical_temperatures, ac, temperature)
            vapor_fraction_differential = self.calculate_temperature_diff(temperature_old, self.diff, acentric_factors, critical_temperatures, ac, a, b, kij, lij, mole_fractions, pressure, amix, bmix, phases, k_factors, vapor_fraction_differential)

            vapor_fraction_differential[0] = vapor_fraction_differential[0] / (2.0 * self.diff)

            # Temperature Step Calculation
            step[0] = vapor_fraction[0] / vapor_fraction_differential[0]

            # Step Brake
            if np.abs(step[0]) > 0.25 * temperature_old:
                step[0] = 0.25 * temperature_old * step[0] / np.abs(step[0])

            # Updating Temperature
            temperature = temperature_old - step[0]

            # Updating K-factors
            a = EOS.eos_parameters(acentric_factors, critical_temperatures, ac, temperature)
            k_factors = self.update_k_factors(temperature, pressure, a, b, kij, lij, mole_fractions, amix, bmix, phases)

        if it == self.iterations_max and np.abs(step[0]) > self.tol:
            raise Exception("In Successive Substitution Method - Maximum Number of Iterations Reached")

        return temperature, k_factors

    @classmethod
    def calculate_temperature_diff(cls, temperature_previous, diff, acentric_factors, critical_temperatures, ac, a, b, kij, lij, mole_fractions, pressure, amix, bmix, phase, k_factors, temperature_diff):
        for ph in phase:
            amix[ph], bmix[ph] = EOS.van_der_waals_mixing_rule(a, b, kij, lij, mole_fractions[:, ph])

        for sign in [-1, 1]:
            temperature = temperature_previous + sign * diff
            a = EOS.eos_parameters(acentric_factors, critical_temperatures, ac, temperature)
            fugacity_coef_difference = Calculator.calculate_fugacity_coefficients_difference(temperature, pressure, a, b, amix, bmix, mole_fractions, kij, lij, phase)
            temperature_diff[0] += sign * np.sum(mole_fractions[:, 0] * k_factors * (-fugacity_coef_difference))

        return temperature_diff

    @classmethod
    def update_k_factors(cls, temperature, pressure, a, b, kij, lij, mole_fractions, amix, bmix, phases):
        for ph in phases:
            amix[ph], bmix[ph] = EOS.van_der_waals_mixing_rule(a, b, kij, lij, mole_fractions[:, ph])

        fugacity_coefficients_difference = Calculator.calculate_fugacity_coefficients_difference(temperature, pressure, a, b, amix, bmix, mole_fractions, kij, lij, phases)
        K = np.exp(-fugacity_coefficients_difference)
        return K
