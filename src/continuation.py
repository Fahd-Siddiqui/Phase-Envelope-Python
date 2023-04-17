import logging

import numpy as np

from src.calculator import Calculator
from src.eos import EOS


class Continuation:
    def __init__(
            self,
            tolerance,
            maximum_iterations,
            maximum_temperature_step,
            maximum_step,
            diff,
            critical_k_factor,
            flag_error,
            flag_crit,
    ):
        self.logger = logging.getLogger(name="Phase Envelope")

        self.tolerance = tolerance
        self.maximum_iterations = maximum_iterations
        self.maximum_temperature_step = maximum_temperature_step
        self.maximum_steps = maximum_step
        self.flag_error = flag_error
        self.flag_crit = flag_crit
        self.critical_k_factor = critical_k_factor
        self.diff = diff

    @staticmethod
    def solve(matrix: np.ndarray, b: np.ndarray):
        return np.linalg.solve(matrix, b)

    def differentiate_first_c_residuals_ln_k_factors(self, number_of_components, mole_fractions, amix, bmix, a, b, kij, lij, volume, pressure, T, phase, dF):
        for i in range(number_of_components):
            for j in range(number_of_components):
                diffFrac = self.diff * mole_fractions[j, 1]
                aux = mole_fractions[j, 1]

                # Numerically Differentiating the Fugacity Coefficient of the Incipient Phase
                for sign in [1, -1]:
                    mole_fractions[j, 1] = aux + sign * diffFrac
                    amix[1], bmix[1] = EOS.VdW1fMIX(number_of_components, a, b, kij, lij, mole_fractions[:, 1])
                    volume[1] = EOS.EoS_Volume(pressure, T, bmix[1], amix[1], phase[1])
                    FugCoef_aux = EOS.fugacity(T, pressure, a, b, amix[1], bmix[1], volume[1], mole_fractions[:, 1], kij[i, :], lij[i, :], i)

                    dF[i * (number_of_components + 2) + j] = FugCoef_aux if sign == 1 else dF[i * (number_of_components + 2) + j] - FugCoef_aux

                # Derivative of ln[FugacityCoefficient(IncipientPhase,Component i)] With Respect to ln[K(j)]
                dF[i * (number_of_components + 2) + j] *= mole_fractions[j, 1] / (2.0 * diffFrac)

                # Derivative of ln[K[i]] With Respect to ln[K(j)] = Kronecker Delta
                if i == j:
                    dF[i * (number_of_components + 2) + j] += 1.0

                mole_fractions[j, 1] = aux  # reset Composition[j, 1] to its original value

    def calculate(
            self,
            temperature,
            pressure,
            number_of_components,

            acentric_factors,
            critical_temperature,
            ac,
            amix,
            bmix,
            b,
            phases,
            kij,
            lij,

            mole_fractions,
            specified_variabel_index,
            specified_variable,
            specified_variable_differential,
            vapor_fraction,
            variation,
            vapor_fraction_differential,
            variation_differential,
            sensitivity

    ):
        phase_envelope_results = []

        point = 0
        while 0.5 <= pressure < 1000.0 and self.flag_error == 0:  # Main Loop
            point = point + 1
            iteration = 0
            maxstep = self.maximum_steps
            while maxstep > self.tolerance and iteration < self.maximum_iterations:  # Newton's Method Loop
                iteration += 1

                # Calculating Residuals
                a = EOS.eos_parameters(acentric_factors, critical_temperature, ac, temperature)
                volume = EOS.calculate_mixing_rules(amix, bmix, number_of_components, a, b, kij, lij, mole_fractions, pressure, temperature, phases)
                fugacity_difference = Calculator.calculate_fugacity_coef_difference(number_of_components, temperature, pressure, a, b, amix, bmix, mole_fractions, kij, lij, phases)
                vapor_fraction[0:number_of_components] = variation[0:number_of_components] + fugacity_difference
                vapor_fraction[number_of_components] = np.sum(mole_fractions[:, 1] - mole_fractions[:, 0])
                vapor_fraction[number_of_components + 1] = variation[specified_variabel_index] - specified_variable

                # Differentiating The First "C" Residuals With Respect to ln[k_factors(j)]
                self.differentiate_first_c_residuals_ln_k_factors(number_of_components, mole_fractions, amix, bmix, a, b, kij, lij, volume, pressure, temperature, phases, vapor_fraction_differential)

                # Differentiating "C+1" Residual With Respect to ln[k_factors[i]]///////////////////////////////////////////////////////////////
                vapor_fraction_differential[number_of_components * (number_of_components + 2):number_of_components * (number_of_components + 2) + number_of_components] = mole_fractions[:, 1]

                # Differentiating The First "C" Residuals With Respect to ln[T]*********************************************************
                diffT = self.diff * variation[number_of_components]

                # Numerically Differentiating The ln(FugacityCoefficient) With Respect to ln(T)
                i_arr = np.arange(number_of_components)
                for sign in [1, -1]:
                    temperature = np.exp(variation[number_of_components] + sign * diffT)
                    a = EOS.eos_parameters(acentric_factors, critical_temperature, ac, temperature)
                    fugacity_difference = Calculator.calculate_fugacity_coef_difference(number_of_components, temperature, pressure, a, b, amix, bmix, mole_fractions, kij, lij, phases)

                    if sign == -1:
                        vapor_fraction_differential[i_arr * (number_of_components + 2) + number_of_components] -= fugacity_difference
                    else:
                        vapor_fraction_differential[i_arr * (number_of_components + 2) + number_of_components] = fugacity_difference

                vapor_fraction_differential[(np.arange(number_of_components) * (number_of_components + 2) + number_of_components)] /= (2.0 * diffT)
                temperature = np.exp(variation[number_of_components])
                a = EOS.eos_parameters(acentric_factors, critical_temperature, ac, temperature)

                # Differentiating The First "C" Residuals With Respect to ln[P]/////////////////////////////////////////////////////////
                diffP = self.diff * variation[number_of_components + 1]
                for ph in phases:
                    amix[ph], bmix[ph] = EOS.VdW1fMIX(number_of_components, a, b, kij, lij, mole_fractions[:, ph])

                # # Numerically Differentiating The ln(FugacityCoefficient) With Respect to ln(T)
                for sign in [1, -1]:
                    pressure = np.exp(variation[number_of_components + 1] + sign * diffP)
                    fugacity_difference = Calculator.calculate_fugacity_coef_difference(number_of_components, temperature, pressure, a, b, amix, bmix, mole_fractions, kij, lij, phases)

                    if sign == -1:
                        vapor_fraction_differential[i_arr * (number_of_components + 2) + number_of_components + 1] -= fugacity_difference
                    else:
                        vapor_fraction_differential[i_arr * (number_of_components + 2) + number_of_components + 1] = fugacity_difference

                vapor_fraction_differential[(i_arr * (number_of_components + 2)) + number_of_components + 1] /= 2.0 * diffP

                # Derivative of the "C+1" Residual With Respect to ln(T)
                vapor_fraction_differential[number_of_components * (number_of_components + 2) + number_of_components + 0] = 0.0

                # Derivative of the "C+1" Residual With Respect to ln(P)
                vapor_fraction_differential[number_of_components * (number_of_components + 2) + number_of_components + 1] = 0.0

                # Derivative of the "C+2" Residual
                vapor_fraction_differential[(number_of_components + 1) * (number_of_components + 2):] = 0.0
                vapor_fraction_differential[(number_of_components + 1) * (number_of_components + 2) + specified_variabel_index] = 1.0

                # Solving The System of Equations
                A = vapor_fraction_differential.reshape(((number_of_components + 2), (number_of_components + 2)), order="C")
                step = self.solve(A, vapor_fraction)
                # step = np.linalg.solve(A, F)
                # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                # self.logger.debug( "VAR", Var, "k_factors", k_factors, "P", P, "T", T, "step", step)

                # Updating The Independent Variables************************************************************************************
                variation = variation - step
                maxstep = max(abs(step / variation))
                # **********************************************************************************************************************

                # Calculating The Natural Form Of Indep# endent Variables And Updating Compositions Of The Incipient Phase////////////////
                k_factors = np.exp(variation[0:number_of_components])
                mole_fractions[:, 1] = mole_fractions[:, 0] * k_factors
                temperature = np.exp(variation[number_of_components + 0])
                pressure = np.exp(variation[number_of_components + 1])

                # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            # self.logger.info("Incipient Phase = ", phases[1] + 1, "    P = ", P, "T = ", T, "specified_variabel_index =", specified_variabel_index)

            if maxstep > self.tolerance or any(np.isnan(variation)):
                flag_error = 1
                # raise Exception("Something went wrong. Unable t oconverge")

            if self.flag_error == 0:
                if self.flag_crit == 2:
                    self.flag_crit = 0

                # self.logger.info(f"{phases[0] + 1}, {P}, {T}, {mole_fractions[1, 1]} {number_of_components}")
                current_phase = "vapor" if phases[0] == 0 else "liquid"
                results = {
                    # "phases": current_phase,
                    "pressure": pressure,
                    "temperature": temperature,
                    "composition": list(mole_fractions.flatten()),
                }

                self.logger.debug(f"{results}")

                phase_envelope_results.append(results)

                # Analyzing Sensitivity Of The Indep# endent Variables************************************************************************
                SpecVar_old = specified_variabel_index
                Var_old = variation

                # Sensitivity Vector Calculation
                A = vapor_fraction_differential.reshape(((number_of_components + 2), (number_of_components + 2)), order="C")
                sensitivity = self.solve(A, variation_differential)

                if self.flag_crit == 0:
                    # Find the greatest sensitivity
                    specified_variabel_index = np.argmax(np.abs(sensitivity))

                    # Updating Specified Variable
                    if specified_variabel_index != SpecVar_old:
                        s = sensitivity[specified_variabel_index]
                        specified_variable_differential *= s
                        sensitivity /= s
                        sensitivity[specified_variabel_index] = 1.0
                        specified_variable = variation[specified_variabel_index]

                    # Adjusting Stepsize////////////////////////////////////////////////////////////////////////////////////////////////////////
                    dSmax = max(abs(variation[specified_variabel_index]) ** 0.5 / 10.0, 0.1) * abs(specified_variable_differential) / specified_variable_differential

                    specified_variable_differential *= 4.0 / iteration
                    if np.abs(dSmax) < np.abs(specified_variable_differential):
                        specified_variable_differential = dSmax

                    # Defining Specified Variable Value In The Next Point
                    specified_variable += specified_variable_differential
                    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    # Indep# endent Variables Initial Guess For The Next Point********************************************************************
                    variation += specified_variable_differential * sensitivity

                    # **************************************************************************************************************************

                    # Analyzing Temperature Stepsize////////////////////////////////////////////////////////////////////////////////////////////
                    T_old = temperature
                    temperature = np.exp(variation[number_of_components + 0])
                    # Large Temperature Steps Are Not Advisable
                    while abs(temperature - T_old) > self.maximum_temperature_step:
                        specified_variable_differential *= 0.5
                        specified_variable -= specified_variable_differential
                        variation -= specified_variable_differential * sensitivity
                        temperature = np.exp(variation[number_of_components])
                    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    # Analyzing Proximity to Critical Point*************************************************************************************
                    # Seeking The Greatest k_factors-factor
                    maxK_i = np.argmax(abs(variation[0:number_of_components]))
                    # If The ln[k_factors[i]] Stepsize Is Too Big, It Should Be Decreased
                    if np.abs(variation[maxK_i]) < 0.1:
                        # Analyzing maxK stepsize
                        if np.abs(specified_variable_differential * sensitivity[maxK_i]) > self.critical_k_factor:  # then
                            specified_variable -= specified_variable_differential
                            variation -= specified_variable_differential * sensitivity
                            specified_variable_differential *= self.critical_k_factor / abs(sensitivity[maxK_i]) / abs(specified_variable_differential)
                            specified_variable += specified_variable_differential
                            variation += specified_variable_differential * sensitivity

                        # The current point must be near enough to the critical point
                        # so that the algorithm can pass through iteration without diverging.

                        if abs(variation[maxK_i]) < self.critical_k_factor:
                            # last point before the critical point
                            specified_variable -= specified_variable_differential
                            variation -= specified_variable_differential * sensitivity
                            specified_variable_differential = (self.critical_k_factor - variation[maxK_i]) / sensitivity[maxK_i]

                            specified_variable += specified_variable_differential
                            variation -= specified_variable_differential * sensitivity
                            self.flag_crit = 1

                else:
                    # Passing Through The Critical Point
                    specified_variable_differential = self.critical_k_factor * specified_variable_differential / abs(specified_variable_differential)
                    specified_variable += 2.0 * specified_variable_differential
                    variation += 2.0 * specified_variable_differential * sensitivity

                    # Defining Incipient Phase As Vapor - Initializing Bubble Curve
                    # flip the k_factors-factors
                    phase_aux = phases[0]
                    phases[0] = phases[1]
                    phases[1] = phase_aux
                    self.flag_crit = 2

            elif self.flag_error == 1 and self.flag_crit == 2 and self.critical_k_factor > 0.009:
                # TODO Porbably unused?
                K_CritPoint = self.critical_k_factor - 0.005
                variation = Var_old
                specified_variable = variation[specified_variabel_index]
                phase_aux = phases[0]
                phases[0] = phases[1]
                phases[1] = phase_aux
                specified_variable_differential = (K_CritPoint - variation[maxK_i]) / sensitivity[maxK_i]

                specified_variable = specified_variable + specified_variable_differential
                variation += specified_variable_differential * sensitivity
                self.flag_crit = 1
                self.flag_error = 0

            elif self.flag_error == 1 and self.flag_crit == 0 and abs(specified_variable_differential) > 1e-6:
                # TODO Porbably unused?
                variation = Var_old
                specified_variable = variation[specified_variabel_index]
                specified_variable_differential = specified_variable_differential / 4.0
                specified_variable = specified_variable + specified_variable_differential
                variation += specified_variable_differential * sensitivity
                self.flag_error = 0

            k_factors = np.exp(variation[0:number_of_components])
            mole_fractions[:, 1] = mole_fractions[:, 0] * k_factors

            temperature = np.exp(variation[number_of_components + 0])
            pressure = np.exp(variation[number_of_components + 1])

        return phase_envelope_results
