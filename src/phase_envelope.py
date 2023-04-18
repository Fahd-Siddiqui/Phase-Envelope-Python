import logging
import time

import numpy as np

from src.constants import Constants, PR
from src.calculator import Calculator
from src.continuation import Continuation
from src.successive_substitution import SuccessiveSubstitution
from src.utils import Utils


class PhaseEnvelope:

    def __init__(self, logging_level: str = ""):
        if not logging_level:
            logging_level = "ERROR"

        self.logger = logging.getLogger(name="Phase Envelope")
        self.logger.setLevel(logging_level.upper())

        console_handler = logging.StreamHandler()

        log_format = '%(asctime)s | %(levelname)s: %(message)s'
        console_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(console_handler)

        self.MAX_ITERATIONS = 100  # Maximum Number Of Iteration In Newton's Method
        self.MAXIMUM_TEMPERATURE_STEP = 5.0  # Maximum Temperature Step In Continuation Method
        self.CRITICAL_K_FACTOR = 0.04  # k_factors-factor Reference Value Used To Detect Critical Points
        self.FLAG_CRITICAL = 0  # if calculating the critical point, flag_crit = 1
        self.FLAG_ERROR = 0
        self.MAXIMUM_STEPS = 1.0e6
        self.TOLERANCE = 1.0e-8
        self.DIFF = 1.0e-8

    def calculate(self, temperature, pressure, vapor_mole_fractions, critical_temperatures, critical_pressures, acentric_factors):
        vapor_mole_fractions = np.array(vapor_mole_fractions)
        critical_temperatures = np.array(critical_temperatures)
        critical_pressures = np.array(critical_pressures)
        acentric_factors = np.array(acentric_factors)

        start_time = time.perf_counter()
        res = self._calculate(temperature, pressure, vapor_mole_fractions, critical_temperatures, critical_pressures, acentric_factors)
        process_time = round(time.perf_counter() - start_time, 2)

        self.logger.info(f"Time taken {process_time} s")

        return res

    def _calculate(self, temperature, pressure, vapor_mole_fractions, critical_temperatures, critical_pressures, acentric_factors):
        number_of_components = len(vapor_mole_fractions)
        amix = np.zeros(2)
        bmix = np.zeros(2)

        mole_fractions = np.zeros((number_of_components, 2))
        kij = np.zeros((number_of_components, number_of_components))
        lij = np.zeros((number_of_components, number_of_components))

        # Independent Variables
        # (F-1)*C   k_factors
        # (F-1)     beta
        # 1         temperature
        # 1         pressure

        step = np.zeros(number_of_components + 2)
        sensitivity = np.zeros(number_of_components + 2)
        vapor_fraction = np.zeros(number_of_components + 2)
        vapor_fraction_differential = np.zeros((number_of_components + 2) ** 2)

        # EoS Parameters Calculation
        b = PR.b_constant * Constants.R * critical_temperatures / critical_pressures
        ac = PR.a_constant * (Constants.R * critical_temperatures) ** 2 / critical_pressures

        k_factors = np.exp(5.373 * (1.0 + acentric_factors) * (1.0 - critical_temperatures / temperature)) * (critical_pressures / pressure)
        vapor_mole_fractions = Utils.normalize(vapor_mole_fractions)
        mole_fractions[:, 0] = vapor_mole_fractions

        phases = np.array([0, 1])
        # phases[0] = 0 #Reference Phase Index (Vapor)
        # phases[1] = 1 #Incipient Phase Index (Liquid)

        step[0] = self.MAXIMUM_STEPS

        temperature = Calculator.get_initial_temperature_guess(temperature, pressure, mole_fractions[:, 0], phases, b, amix, bmix, acentric_factors, critical_temperatures, ac, kij, lij)

        ss = SuccessiveSubstitution(
            max_iterations=self.MAX_ITERATIONS,
            tolerance=self.TOLERANCE,
            diff=self.DIFF
        )

        temperature, k_factors = ss.calculate(
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
        )

        specified_variable = np.log(pressure)  # Specified Variable Value
        specified_variable_differential = 0.1  # Specified Variable Variation
        variation = np.log(np.hstack((k_factors, temperature, pressure)))

        specified_variable_index = number_of_components + 1  # Specified Variable Index
        variation[specified_variable_index] = specified_variable  # Specified Independent Variable
        variation_differential = np.zeros_like(variation)
        variation_differential[number_of_components + 1] = 1.0

        # Initializing Continuation Method
        cm = Continuation(
            self.TOLERANCE,
            self.MAX_ITERATIONS,
            self.MAXIMUM_TEMPERATURE_STEP,
            self.MAXIMUM_STEPS,
            self.DIFF,
            self.CRITICAL_K_FACTOR,
            self.FLAG_ERROR,
            self.FLAG_CRITICAL
        )

        phase_envelope_results = cm.calculate(
            temperature,
            pressure,
            number_of_components,

            acentric_factors,
            critical_temperatures,
            ac,
            amix,
            bmix,
            b,
            phases,
            kij,
            lij,

            mole_fractions,
            specified_variable_index,
            specified_variable,
            specified_variable_differential,
            vapor_fraction,
            variation,
            vapor_fraction_differential,
            variation_differential,
            sensitivity
        )

        return phase_envelope_results
