import logging
import time

import numpy as np

from src.Constants import Constants
from src.eos import EOS
from src.successive_substitution import SuccessiveSubstitution
from src.utils import Utils
from src.calculator import Calculator


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

    def calculate(self, T, P, z, Tc, Pc, acentric):
        z = np.array(z)
        Tc = np.array(Tc)
        Pc = np.array(Pc)
        acentric = np.array(acentric)

        start_time = time.perf_counter()
        res = self._calculate(T, P, z, Tc, Pc, acentric)
        process_time = round(time.perf_counter() - start_time, 2)

        self.logger.info(f"Time taken {process_time} s")

        return res

    def _calculate(self, T, P, z, Tc, Pc, acentric):
        comp = len(z)
        amix = np.zeros(2)
        bmix = np.zeros(2)

        phase_envelope_results = []

        Composition = np.zeros((comp, comp))
        kij = np.zeros((comp, comp))
        lij = np.zeros((comp, comp))

        dF = np.zeros((comp + 2) ** 2)
        step = np.zeros(comp + 2)
        sensitivity = np.zeros(comp + 2)
        F = np.zeros(comp + 2)
        #     Indep# endent Variables     #
        # (F-1)*C          K        #
        #  (F-1)         beta         #
        #   1         temperature     #
        #   1          pressure       #

        # EoS Parameters Calculation
        b = 0.07780 * Constants.R * Tc / Pc  # covolume
        ac = 0.45724 * Constants.R ** 2 * Tc * Tc / Pc

        K= np.exp(5.373 * (1.0 + acentric) * (1.0 - Tc / T)) * (Pc / P)  # Whitson's Approach for Vapor-Liquid Equilibria
        z = Utils.normalize(z)
        Composition[:, 0] = z # Reference Phase Composition

        phase = np.array([0, 1])
        # phase[0] = 0 #Reference Phase Index (Vapor)
        # phase[1] = 1 #Incipient Phase Index (Liquid)

        tol = 1.0e-8
        diff = 1.0e-8
        step[0] = 1.0e+6
        ss = SuccessiveSubstitution(iterations_max=100, tol=tol, diff=diff)
        T, K = ss.run(step, T, acentric, Tc, ac, comp, z, b, kij, lij, Composition, P, amix, bmix, phase, K, F, dF)

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        # Continuation Method And Newton's Method Settings******************************************************************************
        S = np.log(P)  # Specified Variable Value
        dS = 0.1  # Specified Variable Variation
        Var = np.log(np.hstack((K, T, P)))

        SpecVar = comp + 1  # Specified Variable Index
        Var[SpecVar] = S  # Specified Indep# endent Variable
        dfdS = np.zeros_like(Var)
        dfdS[comp + 1] = 1.0
        maxit = 100  # Maximum Number Of Iteration In Newton's Method
        maxTstep = 5.0  # Maximum Temperature Step In Continuation Method
        K_CritPoint = 0.04  # K-factor Reference Value Used To Detect Critical Points
        flag_crit = 0  # if calculating the critical point, flag_crit = 1
        flag_error = 0
        # ******************************************************************************************************************************

        # Initializing Continuation Method
        point = 0
        while 0.5 <= P < 1000.0 and flag_error == 0:  # Main Loop
            point = point + 1
            it = 0
            maxstep = 1e+6
            while maxstep > tol and it < maxit:  # Newton's Method Loop
                it = it + 1

                # Calculating Residuals
                a = EOS.eos_parameters(acentric, Tc, ac, T)
                Volume = EOS.calculate_mixing_rules(amix, bmix, comp, a,b, kij,lij, Composition, P,T, phase)
                fugacity_difference = Calculator.calculate_fugacity_coef_difference(comp, T, P, a, b, amix, bmix, Composition, kij, lij, phase)
                F[0:comp] = Var[0:comp] + fugacity_difference
                F[comp] = np.sum(Composition[:, 1] - Composition[:, 0])
                F[comp + 1] = Var[SpecVar] - S

                # Differentiating The First "C" Residuals With Respect to ln[K(j)]******************************************************
                self.differentiate_first_c_residuals_lnK(comp, diff, Composition, amix, bmix, a, b, kij, lij, Volume, P, T, phase, dF)

                # **********************************************************************************************************************

                # Differentiating "C+1" Residual With Respect to ln[K[i]]///////////////////////////////////////////////////////////////
                dF[comp * (comp + 2):comp * (comp + 2) + comp] = Composition[:, 1]

                # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                # Differentiating The First "C" Residuals With Respect to ln[T]*********************************************************
                diffT = diff * Var[comp]

                # Numerically Differentiating The ln(FugacityCoefficient) With Respect to ln(T)
                i_arr = np.arange(comp)
                for sign in [1, -1]:
                    T = np.exp(Var[comp] + sign * diffT)
                    a = EOS.eos_parameters(acentric, Tc, ac, T)
                    # Volume = EOS.calculate_mixing_rules(amix, bmix, comp, a, b, kij, lij, Composition, P, T, phase)
                    fugacity_difference = Calculator.calculate_fugacity_coef_difference(comp, T, P, a, b, amix, bmix, Composition, kij, lij, phase)

                    if sign == -1:
                        dF[i_arr * (comp + 2) + comp] -= fugacity_difference
                    else:
                        dF[i_arr * (comp + 2) + comp] = fugacity_difference

                dF[(np.arange(comp) * (comp + 2) + comp)] /= (2.0 * diffT)
                T = np.exp(Var[comp])
                a = EOS.eos_parameters(acentric, Tc, ac, T) 

                # Differentiating The First "C" Residuals With Respect to ln[P]/////////////////////////////////////////////////////////
                diffP = diff * Var[comp + 1]
                for ph in phase:
                    amix[ph], bmix[ph] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, ph])

                # # Numerically Differentiating The ln(FugacityCoefficient) With Respect to ln(T)
                for sign in [1, -1]:
                    P = np.exp(Var[comp + 1] + sign * diffP)
                    fugacity_difference = Calculator.calculate_fugacity_coef_difference(comp, T, P, a, b, amix, bmix, Composition, kij, lij, phase)

                    if sign == -1:
                        dF[i_arr * (comp + 2) + comp + 1] -= fugacity_difference
                    else:
                        dF[i_arr * (comp + 2) + comp + 1] = fugacity_difference

                dF[(i_arr * (comp + 2)) + comp + 1] /= 2.0 * diffP

                # Derivative of the "C+1" Residual With Respect to ln(T)
                dF[comp * (comp + 2) + comp + 0] = 0.0

                # Derivative of the "C+1" Residual With Respect to ln(P)
                dF[comp * (comp + 2) + comp + 1] = 0.0

                # Derivative of the "C+2" Residual
                dF[(comp + 1) * (comp + 2):] = 0.0
                dF[(comp + 1) * (comp + 2) + SpecVar] = 1.0

                # Solving The System of Equations
                A = dF.reshape(((comp + 2), (comp + 2)), order="C")
                step = self.solve(A, F)
                # step = np.linalg.solve(A, F)
                # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                # self.logger.debug( "VAR", Var, "K", K, "P", P, "T", T, "step", step)

                # Updating The Independent Variables************************************************************************************
                Var = Var - step
                maxstep = max(abs(step/Var))
                # **********************************************************************************************************************

                # Calculating The Natural Form Of Indep# endent Variables And Updating Compositions Of The Incipient Phase////////////////
                K = np.exp(Var[0:comp])
                Composition[:, 1] = Composition[:, 0] * K
                T = np.exp(Var[comp + 0])
                P = np.exp(Var[comp + 1])

                # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            # self.logger.info("Incipient Phase = ", phase[1] + 1, "    P = ", P, "T = ", T, "SpecVar =", SpecVar)

            if maxstep > tol or any(np.isnan(Var)):
                flag_error = 1
                # raise Exception("Something went wrong. Unable t oconverge")

            if flag_error == 0:
                if flag_crit == 2:
                    flag_crit = 0

                # self.logger.info(f"{phase[0] + 1}, {P}, {T}, {Composition[1, 1]} {comp}")
                current_phase = "vapor" if phase[0] == 0 else "liquid"
                results = {
                    # "phase": current_phase,
                    "pressure": P,
                    "temperature": T,
                    "composition": list(Composition.flatten()),
                }

                self.logger.debug(f"{results}")

                phase_envelope_results.append(results)

                # Analyzing Sensitivity Of The Indep# endent Variables************************************************************************
                SpecVar_old = SpecVar
                Var_old = Var

                # Sensitivity Vector Calculation
                A = dF.reshape(((comp + 2), (comp + 2)), order="C")
                sensitivity = self.solve(A, dfdS)

                if flag_crit == 0:
                    # Find the greatest sensitivity
                    SpecVar = np.argmax(np.abs(sensitivity))

                    # Updating Specified Variable
                    if SpecVar != SpecVar_old:
                        s = sensitivity[SpecVar]
                        dS *= s
                        sensitivity /= s
                        sensitivity[SpecVar] = 1.0
                        S = Var[SpecVar]

                    # Adjusting Stepsize////////////////////////////////////////////////////////////////////////////////////////////////////////
                    dSmax = max(abs(Var[SpecVar]) ** 0.5 / 10.0, 0.1) * abs(dS) / dS

                    dS *= 4.0 / it
                    if np.abs(dSmax) < np.abs(dS):
                        dS = dSmax

                    # Defining Specified Variable Value In The Next Point
                    S += dS
                    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    # Indep# endent Variables Initial Guess For The Next Point********************************************************************
                    Var += dS * sensitivity

                    # **************************************************************************************************************************

                    # Analyzing Temperature Stepsize////////////////////////////////////////////////////////////////////////////////////////////
                    T_old = T
                    T = np.exp(Var[comp + 0])
                    # Large Temperature Steps Are Not Advisable
                    while abs(T - T_old) > maxTstep:
                        dS *= 0.5
                        S -= dS
                        Var -= dS * sensitivity
                        T = np.exp(Var[comp])
                    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    # Analyzing Proximity to Critical Point*************************************************************************************
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

                else:
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

            elif flag_error == 1 and flag_crit == 2 and K_CritPoint > 0.009:
                # TODO Porbably unused?
                K_CritPoint = K_CritPoint - 0.005
                Var = Var_old
                S = Var[SpecVar]
                phase_aux = phase[0]
                phase[0] = phase[1]
                phase[1] = phase_aux
                dS = (K_CritPoint - Var[maxK_i]) / sensitivity[maxK_i]

                S = S + dS
                Var += dS * sensitivity
                flag_crit = 1
                flag_error = 0

            elif flag_error == 1 and flag_crit == 0 and abs(dS) > 1e-6:
                # TODO Porbably unused?
                Var = Var_old
                S = Var[SpecVar]
                dS = dS / 4.0
                S = S + dS
                Var += dS * sensitivity
                flag_error = 0

            K = np.exp(Var[0:comp])
            Composition[:, 1] = Composition[:, 0] * K

            T = np.exp(Var[comp + 0])
            P = np.exp(Var[comp + 1])

        return phase_envelope_results

    @staticmethod
    def solve(matrix: np.ndarray, b: np.ndarray):
        return np.linalg.solve(matrix, b)
    def differentiate_first_c_residuals_lnK(self, comp, diff, Composition, amix, bmix, a, b, kij, lij, Volume, P, T, phase, dF):
        for i in range(comp):
            for j in range(comp):
                diffFrac = diff * Composition[j, 1]
                aux = Composition[j, 1]

                # Numerically Differentiating the Fugacity Coefficient of the Incipient Phase
                for sign in [1, -1]:
                    Composition[j, 1] = aux + sign * diffFrac
                    amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])
                    Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], phase[1])
                    FugCoef_aux = EOS.fugacity(T, P, a, b, amix[1], bmix[1], Volume[1], Composition[:, 1], kij[i, :], lij[i, :], i)

                    dF[i * (comp + 2) + j] = FugCoef_aux if sign == 1 else dF[i * (comp + 2) + j] - FugCoef_aux

                # Derivative of ln[FugacityCoefficient(IncipientPhase,Component i)] With Respect to ln[K(j)]
                dF[i * (comp + 2) + j] *= Composition[j, 1] / (2.0 * diffFrac)

                # Derivative of ln[K[i]] With Respect to ln[K(j)] = Kronecker Delta
                if i == j:
                    dF[i * (comp + 2) + j] += 1.0

                Composition[j, 1] = aux  # reset Composition[j, 1] to its original value
