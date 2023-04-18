import numpy as np

from src.eos import EOS


class Calculator:
    @classmethod
    def calculate_fugacity_coefficients_difference(cls, temperature, pressure, a, b, amix, bmix, mole_fractions, kij, lij, phase):
        volume, fugacity_coefficient_reference, fugacity_coefficient_auxiliary = cls. calculate_fugacity_coefficients(temperature, pressure, a, b, amix, bmix, mole_fractions, kij, lij, phase)
        return fugacity_coefficient_reference - fugacity_coefficient_auxiliary

    @classmethod
    def calculate_fugacity_coefficients(cls, temperature, pressure, a, b, amix, bmix, mole_fractions, kij, lij, phase):
        number_of_components = mole_fractions.shape[0]
        volume = EOS.calculate_mixing_rules(amix, bmix, a, b, kij, lij, mole_fractions, pressure, temperature, phase)
        fugacity_vec = EOS.fugacity_vec(temperature, pressure, a, b, amix, bmix, volume, mole_fractions, kij, lij)

        fugacity_coefficients = [
            fugacity_vec(np.arange(number_of_components), ph).astype(np.float64)
            for ph in phase
        ]

        return volume, fugacity_coefficients[phase[1]],  fugacity_coefficients[phase[0]]

    @staticmethod
    def get_initial_temperature_guess(
            temperature: float,
            pressure: float,
            vapor_mole_fractions,
            phase,
            b: np.ndarray,
            amix,
            bmix,
            acentric_factors,
            critical_temperatures,
            ac,
            kij: np.ndarray,
            lij: np.ndarray
    ):
        temperature_old = temperature - 1.0
        while temperature_old != temperature:
            temperature_old = temperature

            a = EOS.eos_parameters(acentric_factors, critical_temperatures, ac, temperature)  # Updating Attractive Parameter
            amix[0], bmix[0] = EOS.van_der_waals_mixing_rule(a, b, kij, lij, vapor_mole_fractions)  # Mixing Rule
            amix[1], bmix[1] = amix[0], bmix[0]

            volume, fugacity_coefficient_liquid, fugacity_coefficient_vapor = Calculator.calculate_fugacity_coefficients(temperature, pressure, a, b, amix, bmix, np.array([vapor_mole_fractions, vapor_mole_fractions]).T, kij, lij, phase)

            gibbs_vap = np.sum(vapor_mole_fractions * fugacity_coefficient_vapor)
            gibbs_liq = np.sum(vapor_mole_fractions * fugacity_coefficient_liquid)

            # factor 1.75 was proposed by Pedersen and Christensen (Phase Behavior Of Petroleum Reservoir Fluids, 2007 - Chapter 6.5 Phase Identification)
            if (gibbs_liq < gibbs_vap) or (gibbs_liq == gibbs_vap and (volume[0] / bmix[0]) < 1.75):
                temperature = temperature + 10.0

        return temperature
