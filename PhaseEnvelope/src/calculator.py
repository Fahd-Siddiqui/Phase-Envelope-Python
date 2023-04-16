import numpy as np

from PhaseEnvelope.src.eos import EOS


class Calculator:
    @classmethod
    def calculate_fugacity_coef_difference(cls, comp, T, P, a, b, amix, bmix, Composition, kij, lij, phase):
        volume = EOS.calculate_mixing_rules(amix, bmix, comp, a, b, kij, lij, Composition, P, T, phase)
        fugacity_vec = EOS.fugacity_vec(T, P, a, b, amix, bmix, volume, Composition, kij, lij)

        fugacity_coefficients = [
            fugacity_vec(np.arange(comp), ph).astype('float64')
            for ph in phase
        ]

        return fugacity_coefficients[phase[1]] - fugacity_coefficients[phase[0]]
