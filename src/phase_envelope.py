import logging
import time
from typing import Dict, List

import numpy as np
import numpy.typing as npt

from src.calculator import Calculator
from src.Constants import Constants
from src.eos import EOS
from src.successive_substitution import SuccessiveSubstitution
from src.utils import Utils


class PhaseEnvelope:
    TOLERANCE = 1.0e-8
    diff = 1.0e-8
    MAXIMUM_ITERATIONS = 100  # Maximum Number Of Iteration In Newton's Method
    MAXIMUM_TEMPERATURE_STEP = 5.0  # Maximum Temperature Step In Continuation Method

    def __init__(self, logging_level: str = ""):
        if not logging_level:
            logging_level = "ERROR"

        self.logger = logging.getLogger(name="Phase Envelope")
        self.logger.setLevel(logging_level.upper())

        console_handler = logging.StreamHandler()

        log_format = "%(asctime)s | %(levelname)s: %(message)s"
        console_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(console_handler)

    def calculate(
        self,
        temperature: float,
        pressure: float,
        phase_compositions: npt.NDArray[np.float64],
        critical_temperatures: npt.NDArray[np.float64],
        critical_pressures: npt.NDArray[np.float64],
        acentric_factors: npt.NDArray[np.float64],
    ):
        phase_compositions = np.array(phase_compositions)
        critical_temperatures = np.array(critical_temperatures)
        critical_pressures = np.array(critical_pressures)
        acentric_factors = np.array(acentric_factors)

        start_time = time.perf_counter()
        results: List[Dict] = self._calculate(
            temperature,
            pressure,
            phase_compositions,
            critical_temperatures,
            critical_pressures,
            acentric_factors,
        )
        process_time = round(time.perf_counter() - start_time, 2)

        self.logger.info(f"Time taken {process_time} s")

        return results

    def _calculate(
        self,
        temperature: float,
        pressure: float,
        phase_compositions: npt.NDArray[np.float64],
        critical_temperatures: npt.NDArray[np.float64],
        critical_pressures: npt.NDArray[np.float64],
        acentric_factors: npt.NDArray[np.float64],
    ):
        component_index = len(phase_compositions)
        a_mix: npt.NDArray[np.float64] = np.zeros(2)
        b_mix: npt.NDArray[np.float64] = np.zeros(2)

        phase_envelope_results = []

        composition: npt.NDArray[np.float64] = np.zeros((component_index, component_index))
        kij: npt.NDArray[np.float64] = np.zeros((component_index, component_index))
        lij: npt.NDArray[np.float64] = np.zeros((component_index, component_index))

        dF = np.zeros((component_index + 2) ** 2)
        sensitivity = np.zeros(component_index + 2)
        F = np.zeros(component_index + 2)
        #     Indep# endent Variables     #
        # (F-1)*C          k_factors        #
        #  (F-1)         beta         #
        #   1         temperature     #
        #   1          pressure       #

        # EoS Parameters Calculation
        b = Constants.PR_b0 * Constants.R * critical_temperatures / critical_pressures
        ac = Constants.PR_a0 * Constants.R**2 * critical_temperatures * critical_temperatures / critical_pressures

        # Whitson's Approach for Vapor-Liquid Equilibria
        k_factors = np.exp(5.373 * (1.0 + acentric_factors) * (1.0 - critical_temperatures / temperature)) * (
            critical_pressures / pressure
        )

        phase_compositions = Utils.normalize(phase_compositions)
        composition[:, 0] = phase_compositions  # Reference Phase composition

        phase = np.array([0, 1])
        # phase[0] = 0 #Reference Phase Index (Vapor)
        # phase[1] = 1 #Incipient Phase Index (Liquid)

        ss = SuccessiveSubstitution(iterations_max=100, tol=self.TOLERANCE, diff=self.diff)
        temperature, k_factors = ss.run(
            1.0e6,
            temperature,
            acentric_factors,
            critical_temperatures,
            ac,
            component_index,
            phase_compositions,
            b,
            kij,
            lij,
            composition,
            pressure,
            a_mix,
            b_mix,
            phase,
            k_factors,
            F,
            dF,
        )

        # Continuation Method And Newton's Method Settings
        S = np.log(pressure)  # Specified Variable Value
        dS = 0.1  # Specified Variable Variation
        Var = np.log(np.hstack((k_factors, temperature, pressure)))

        self.specified_variable_index = component_index + 1  # Specified Variable Index
        Var[self.specified_variable_index] = S  # Specified Independent Variable
        diff_f_diff_s = np.zeros_like(Var)
        diff_f_diff_s[component_index + 1] = 1.0
        K_CritPoint = 0.04  # k_factors-factor Reference Value Used To Detect Critical Points
        flag_crit = 0  # if calculating the critical point, flag_crit = 1
        flag_error = 0

        # Initializing Continuation Method
        point = 0
        while 0.5 <= pressure < 1000.0 and flag_error == 0:  # Main Loop
            point = point + 1
            it = 0
            max_step = 1e6
            (
                F,
                Var,
                S,
                dF,
                a_mix,
                b_mix,
                component_index,
                b,
                kij,
                lij,
                composition,
                pressure,
                temperature,
                phase,
                acentric_factors,
                critical_temperatures,
                ac,
                it,
                max_step,
            ) = self.newtons_method_loop(
                F,
                Var,
                S,
                dF,
                a_mix,
                b_mix,
                component_index,
                b,
                kij,
                lij,
                composition,
                pressure,
                temperature,
                phase,
                acentric_factors,
                critical_temperatures,
                ac,
                it,
                max_step,
            )

            # self.logger.info(
            #     "Incipient Phase = ",
            #     phase[1] + 1,
            #     "    P = ",
            #     P,
            #     "T = ",
            #     T,
            #     "self.specified_variable =",
            #     self.specified_variable,
            # )

            if max_step > self.TOLERANCE or any(np.isnan(Var)):
                flag_error = 1
                # raise Exception("Something went wrong. Unable to converge")

            if flag_error == 0:
                if flag_crit == 2:
                    flag_crit = 0

                # self.logger.info(f"{phase[0] + 1}, {P}, {T}, {composition[1, 1]} {component_index}")
                results = {
                    "phase": "vapor" if phase[0] == 0 else "liquid",
                    "pressure": pressure,
                    "temperature": temperature,
                    "composition": list(composition.flatten()),
                }

                self.logger.debug(f"{results}")

                phase_envelope_results.append(results)

                # Analyzing Sensitivity Of The Independent Variables
                self.specified_variable_index_old = self.specified_variable_index
                Var_old = Var

                # Sensitivity Vector Calculation
                sensitivity_matrix = dF.reshape(((component_index + 2), (component_index + 2)), order="C")
                sensitivity = self.solve(sensitivity_matrix, diff_f_diff_s)

                if flag_crit == 0:
                    (
                        dS,
                        K_CritPoint,
                        S,
                        Var,
                        sensitivity,
                        it,
                        temperature,
                        component_index,
                        flag_crit,
                        maxK_i,
                        dSmax,
                        T_old,
                    ) = self.handle_non_critical_point(
                        dS,
                        K_CritPoint,
                        S,
                        Var,
                        sensitivity,
                        it,
                        temperature,
                        component_index,
                        flag_crit,
                    )
                else:
                    dS, K_CritPoint, S, Var, sensitivity, phase, flag_crit = self.handle_critical_point(
                        dS, K_CritPoint, S, Var, sensitivity, phase
                    )

            elif flag_error == 1 and flag_crit == 2 and K_CritPoint > 0.009:
                # TODO Porbably unused?
                K_CritPoint = K_CritPoint - 0.005
                Var = Var_old
                S = Var[self.specified_variable_index]
                phase_aux = phase[0]
                phase[0] = phase[1]
                phase[1] = phase_aux
                dS = (K_CritPoint - Var[maxK_i]) / sensitivity[maxK_i]

                S = S + dS
                Var += dS * sensitivity
                flag_crit = 1
                flag_error = 0

            elif flag_error == 1 and flag_crit == 0 and abs(dS) > self.TOLERANCE:
                # TODO Porbably unused?
                Var = Var_old
                S = Var[self.specified_variable_index]
                dS = dS / 4.0
                S = S + dS
                Var += dS * sensitivity
                flag_error = 0

            k_factors = np.exp(Var[0:component_index])
            composition[:, 1] = composition[:, 0] * k_factors

            temperature = np.exp(Var[component_index + 0])
            pressure = np.exp(Var[component_index + 1])

        return phase_envelope_results

    @staticmethod
    def solve(matrix: npt.NDArray[np.float64], b: npt.NDArray[np.float64]):
        return np.linalg.solve(matrix, b)

    def differentiate_first_c_residuals_lnK(
        self,
        comp,
        Composition,
        amix,
        bmix,
        a,
        b,
        kij,
        lij,
        Volume,
        P,
        T,
        phase,
        dF,
    ):
        for i in range(comp):
            for j in range(comp):
                diffFrac = self.diff * Composition[j, 1]
                aux = Composition[j, 1]

                # Numerically Differentiating the Fugacity Coefficient of the Incipient Phase
                for sign in [1, -1]:
                    Composition[j, 1] = aux + sign * diffFrac
                    amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])
                    Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], phase[1])
                    FugCoef_aux = EOS.fugacity(
                        T,
                        P,
                        a,
                        b,
                        amix[1],
                        bmix[1],
                        Volume[1],
                        Composition[:, 1],
                        kij[i, :],
                        lij[i, :],
                        i,
                    )

                    dF[i * (comp + 2) + j] = FugCoef_aux if sign == 1 else dF[i * (comp + 2) + j] - FugCoef_aux

                # Derivative of ln[FugacityCoefficient(IncipientPhase,Component i)] With Respect to ln[K(j)]
                dF[i * (comp + 2) + j] *= Composition[j, 1] / (2.0 * diffFrac)

                # Derivative of ln[K[i]] With Respect to ln[K(j)] = Kronecker Delta
                if i == j:
                    dF[i * (comp + 2) + j] += 1.0

                Composition[j, 1] = aux  # reset Composition[j, 1] to its original value

    def newtons_method_loop(
        self,
        F,
        Var,
        S,
        dF,
        amix,
        bmix,
        comp,
        b,
        kij,
        lij,
        Composition,
        P,
        T,
        phase,
        acentric,
        Tc,
        ac,
        it,
        maxstep,
    ):
        while maxstep > self.TOLERANCE and it < self.MAXIMUM_ITERATIONS:  # Newton's Method Loop
            it = it + 1

            # Calculating Residuals
            a = EOS.eos_parameters(acentric, Tc, ac, T)
            Volume = EOS.calculate_mixing_rules(
                amix,
                bmix,
                comp,
                a,
                b,
                kij,
                lij,
                Composition,
                P,
                T,
                phase,
            )
            fugacity_difference = Calculator.calculate_fugacity_coef_difference(
                comp,
                T,
                P,
                a,
                b,
                amix,
                bmix,
                Composition,
                kij,
                lij,
                phase,
            )
            F[0:comp] = Var[0:comp] + fugacity_difference
            F[comp] = np.sum(Composition[:, 1] - Composition[:, 0])
            F[comp + 1] = Var[self.specified_variable_index] - S

            # Differentiating The First "C" Residuals With Respect to ln[K(j)]
            self.differentiate_first_c_residuals_lnK(
                comp,
                Composition,
                amix,
                bmix,
                a,
                b,
                kij,
                lij,
                Volume,
                P,
                T,
                phase,
                dF,
            )

            # Differentiating "C+1" Residual With Respect to ln[K[i]]
            dF[comp * (comp + 2) : comp * (comp + 2) + comp] = Composition[:, 1]

            # Differentiating The First "C" Residuals With Respect to ln[T]
            diffT = self.diff * Var[comp]

            # Numerically Differentiating The ln(Fugacity Coefficient) With Respect to ln(T)
            index = np.arange(comp) * (comp + 2) + comp
            for sign in [1, -1]:
                T = np.exp(Var[comp] + sign * diffT)
                a = EOS.eos_parameters(acentric, Tc, ac, T)
                # Volume = EOS.calculate_mixing_rules(amix, bmix, comp, a, b, kij, lij, Composition, P, T, phase)
                fugacity_difference = Calculator.calculate_fugacity_coef_difference(
                    comp, T, P, a, b, amix, bmix, Composition, kij, lij, phase
                )

                if sign == -1:
                    dF[index] -= fugacity_difference
                else:
                    dF[index] = fugacity_difference

            dF[index] /= 2.0 * diffT
            T = np.exp(Var[comp])
            a = EOS.eos_parameters(acentric, Tc, ac, T)

            # Differentiating The First "C" Residuals With Respect to ln[P]
            diffP = self.diff * Var[comp + 1]
            for ph in phase:
                amix[ph], bmix[ph] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, ph])

            # # Numerically Differentiating The ln(FugacityCoefficient) With Respect to ln(T)
            for sign in [1, -1]:
                P = np.exp(Var[comp + 1] + sign * diffP)
                fugacity_difference = Calculator.calculate_fugacity_coef_difference(
                    comp, T, P, a, b, amix, bmix, Composition, kij, lij, phase
                )

                if sign == -1:
                    dF[index + 1] -= fugacity_difference
                else:
                    dF[index + 1] = fugacity_difference

            dF[index + 1] /= 2.0 * diffP

            # Derivative of the "C+1" Residual With Respect to ln(T)
            index = comp * (comp + 2) + comp
            dF[index] = 0.0

            # Derivative of the "C+1" Residual With Respect to ln(P)
            dF[index + 1] = 0.0

            # Derivative of the "C+2" Residual
            dF[(comp + 1) * (comp + 2) :] = 0.0
            dF[(comp + 1) * (comp + 2) + self.specified_variable_index] = 1.0

            # Solving The System of Equations
            step = self.solve(dF.reshape(((comp + 2), (comp + 2)), order="C"), F)
            # self.logger.debug( "VAR", Var, "K", K, "P", P, "T", T, "step", step)

            # Updating The Independent Variables
            Var = Var - step
            maxstep = max(abs(step / Var))

            # Calculating The Natural Form Of Indep# endent Variables And Updating Compositions Of The Incipient Phase
            K = np.exp(Var[0:comp])
            Composition[:, 1] = Composition[:, 0] * K
            T = np.exp(Var[comp + 0])
            P = np.exp(Var[comp + 1])

        return (
            F,
            Var,
            S,
            dF,
            amix,
            bmix,
            comp,
            b,
            kij,
            lij,
            Composition,
            P,
            T,
            phase,
            acentric,
            Tc,
            ac,
            it,
            maxstep,
        )

    def handle_critical_point(self, dS, K_CritPoint, S, Var, sensitivity, phase):
        # Passing Through The Critical Point
        dS = K_CritPoint * dS / abs(dS)
        S += 2.0 * dS
        Var += 2.0 * dS * sensitivity

        # Defining Incipient Phase As Vapor - Initializing Bubble Curve
        # flip the K-factors
        phase_aux = phase[0]
        phase[0] = phase[1]
        phase[1] = phase_aux
        flag_crit = 2

        return dS, K_CritPoint, S, Var, sensitivity, phase, flag_crit

    def handle_non_critical_point(
        self,
        dS,
        K_CritPoint,
        S,
        Var,
        sensitivity,
        it,
        T,
        comp,
        flag_crit,
    ):
        # Find the greatest sensitivity
        self.specified_variable_index = np.argmax(np.abs(sensitivity))

        # Updating Specified Variable
        if self.specified_variable_index != self.specified_variable_index_old:
            s = sensitivity[self.specified_variable_index]
            dS *= s
            sensitivity /= s
            sensitivity[self.specified_variable_index] = 1.0
            S = Var[self.specified_variable_index]

        # Adjusting Stepsize
        dSmax = max(abs(Var[self.specified_variable_index]) ** 0.5 / 10.0, 0.1) * abs(dS) / dS

        dS *= 4.0 / it
        if np.abs(dSmax) < np.abs(dS):
            dS = dSmax

        # Defining Specified Variable Value In The Next Point
        S += dS

        # Indep# endent Variables Initial Guess For The Next Point
        Var += dS * sensitivity

        # **************************************************************************************************************************

        # Analyzing Temperature Stepsize
        T_old = T
        T = np.exp(Var[comp + 0])
        # Large Temperature Steps Are Not Advisable
        while abs(T - T_old) > self.MAXIMUM_TEMPERATURE_STEP:
            dS *= 0.5
            S -= dS
            Var -= dS * sensitivity
            T = np.exp(Var[comp])

        # Analyzing Proximity to Critical Point
        # Seeking The Greatest K-factor
        maxK_i = np.argmax(abs(Var[0:comp]))
        # If The ln[K[i]] Stepsize Is Too Big, It Should Be Decreased
        if np.abs(Var[maxK_i]) < 0.1:
            # Analyzing maxK stepsize
            if np.abs(dS * sensitivity[maxK_i]) > K_CritPoint:  # then
                S -= dS
                Var -= dS * sensitivity
                dS *= K_CritPoint / abs(sensitivity[maxK_i]) / abs(dS)
                S += dS
                Var += dS * sensitivity

            # The current point must be near enough to the critical point
            # so that the algorithm can pass through it without diverging.

            if abs(Var[maxK_i]) < K_CritPoint:
                # last point before the critical point
                S -= dS
                Var -= dS * sensitivity
                dS = (K_CritPoint - Var[maxK_i]) / sensitivity[maxK_i]

                S += dS
                Var -= dS * sensitivity
                flag_crit = 1

        return (
            dS,
            K_CritPoint,
            S,
            Var,
            sensitivity,
            it,
            T,
            comp,
            flag_crit,
            maxK_i,
            dSmax,
            T_old,
        )
