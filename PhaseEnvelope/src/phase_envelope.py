import numpy as np

from PhaseEnvelope.src.Constants import Constants
from PhaseEnvelope.src.eos import EOS
from PhaseEnvelope.src.successive_substitution import SuccessiveSubstitution


class PhaseEnvelope:

    def calculate(self, T, P, comp, z, Tc, Pc, acentric):
        # TODO Delete
        # b = np.zeros(comp)
        # ac = np.zeros(comp)
        # K = np.zeros(comp)

        amix = np.zeros(2)
        bmix = np.zeros(2)

        phase_envelope_results = []

        Composition = np.zeros((comp, comp))
        kij = np.zeros((comp, comp))
        lij = np.zeros((comp, comp))

        dF = np.zeros((comp + 2) ** 2, dtype=np.float64)
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
        z = z / sum(z)  # Normalizing Global Composition
        Composition[:, 0] = z # Reference Phase Composition

        phase = [0, 1]
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
        while (P >= 0.5 and P < 1000. and flag_error == 0):  # Main Loop
            point = point + 1
            it = 0
            maxstep = 1e+6
            while (maxstep > tol and it < maxit):  # Newton's Method Loop
                it = it + 1

                # Calculating Residuals/////////////////////////////////////////////////////////////////////////////////////////////////
                a = EOS.eos_parameters(acentric, Tc, ac, T)
                amix[0], bmix[0] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 0])  # Mixing Rule - Reference Phase
                amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
                Volume = EOS.Eos_Volumes(P,T,amix, bmix, phase)
                FugCoef_ref, FugCoef_aux = EOS.calculate_fugacity_coefs(comp, T, P, a, b, amix, bmix, Volume, Composition, kij, lij)
                F[0:comp] = Var[0:comp] + FugCoef_aux - FugCoef_ref
                F[comp] = np.sum(Composition[:, 1] - Composition[:, 0])

                # Residual Responsible For Determining The Specified Indep# endent Variable
                F[comp + 1] = Var[SpecVar] - S
                # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                # Differentiating The First "C" Residuals With Respect to ln[K(j)]******************************************************
                for i in range(comp):
                    for j in range(comp):
                        diffFrac = diff * Composition[j, 1]

                        aux = Composition[j, 1]
                        # Numerically Differentiating the Fugacity Coefficient of the Incipient Phase
                        Composition[j, 1] = aux + diffFrac
                        amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
                        Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], phase[1])
                        FugCoef_aux = EOS.fugacity(T, P, a, b, amix[1], bmix[1], Volume[1], Composition[:, 1], kij[i, :], lij[i, :], i)
                        dF[(i) * (comp + 2) + j] = FugCoef_aux

                        Composition[j, 1] = aux - diffFrac
                        amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
                        Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], phase[1])
                        FugCoef_aux = EOS.fugacity(T, P, a, b, amix[1], bmix[1], Volume[1], Composition[:, 1], kij[i, :], lij[i, :], i)
                        dF[(i) * (comp + 2) + j] -= FugCoef_aux

                        Composition[j, 1] = aux
                        # Derivative of ln[FugacityCoefficient(IncipientPhase,Component i)] With Respect to ln[K(j)]
                        dF[(i) * (comp + 2) + j] *= Composition[j, 1] / (2.0 * diffFrac)

                        # Derivative of ln[K[i]] With Respect to ln[K(j)] = Kronecker Delta
                        if i == j:
                            dF[i * (comp + 2) + j] += 1.0

                # **********************************************************************************************************************

                # Differentiating "C+1" Residual With Respect to ln[K[i]]///////////////////////////////////////////////////////////////
                dF[comp * (comp + 2):comp * (comp + 2) + comp] = Composition[:, 1]

                # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                # Differentiating The First "C" Residuals With Respect to ln[T]*********************************************************
                diffT = diff * Var[comp]

                # Numerically Differentiating The ln(FugacityCoefficient) With Respect to ln(T)
                T = np.exp(Var[comp] + diffT)
                a = EOS.eos_parameters(acentric, Tc, ac, T) 
                amix[0], bmix[0] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 0])  # Mixing Rule - Reference Phase
                amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
                Volume = EOS.Eos_Volumes(P,T,amix, bmix, phase)

                FugCoef_ref, FugCoef_aux = EOS.calculate_fugacity_coefs(comp, T, P, a, b, amix, bmix, Volume, Composition, kij, lij)
                i_arr = np.arange(comp)
                dF[i_arr * (comp + 2) + comp] = FugCoef_aux - FugCoef_ref

                T = np.exp(Var[comp + 0] - diffT)
                a = EOS.eos_parameters(acentric, Tc, ac, T) 
                amix[0], bmix[0] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 0])  # Mixing Rule - Reference Phase
                amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
                Volume = EOS.Eos_Volumes(P,T,amix, bmix, phase)
                FugCoef_ref, FugCoef_aux = EOS.calculate_fugacity_coefs(comp, T, P, a, b, amix, bmix, Volume, Composition, kij, lij)
                dF[i_arr * (comp + 2) + comp] -= (FugCoef_aux - FugCoef_ref)

                dF[(np.arange(comp) * (comp + 2) + comp)] /= (2.0 * diffT)

                T = np.exp(Var[comp])
                a = EOS.eos_parameters(acentric, Tc, ac, T) 
                # OBS: The derivative ok ln(K) with respect to ln(T) is null.
                # **********************************************************************************************************************

                # Differentiating The First "C" Residuals With Respect to ln[P]/////////////////////////////////////////////////////////
                diffP = diff * Var[comp + 1]

                amix[0], bmix[0] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 0])  # Mixing Rule - Reference Phase
                amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase

                # Numerically Differentiating The ln(FugacityCoefficient) With Respect to ln(T)
                P = np.exp(Var[comp + 1] + diffP)
                Volume = EOS.Eos_Volumes(P,T,amix, bmix, phase)
                FugCoef_ref, FugCoef_aux = EOS.calculate_fugacity_coefs(comp, T, P, a, b, amix, bmix, Volume, Composition, kij, lij)
                dF[i_arr * (comp + 2) + comp + 1] = FugCoef_aux - FugCoef_ref

                P = np.exp(Var[comp + 1] - diffP)
                Volume = EOS.Eos_Volumes(P,T,amix, bmix, phase)
                FugCoef_ref, FugCoef_aux = EOS.calculate_fugacity_coefs(comp, T, P, a, b, amix, bmix, Volume, Composition, kij, lij)
                dF[i_arr * (comp + 2) + comp+1] -= (FugCoef_aux - FugCoef_ref)

                dF[(i_arr * (comp + 2)) + comp + 1] /= 2.0 * diffP

                # OBS: The derivative ok ln(K) with respect to ln(P) is null.
                # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                # Derivative of the "C+1" Residual With Respect to ln(T)****************************************************************
                dF[comp * (comp + 2) + comp + 0] = 0.0
                # **********************************************************************************************************************

                # Derivative of the "C+1" Residual With Respect to ln(P)////////////////////////////////////////////////////////////////
                dF[comp * (comp + 2) + comp + 1] = 0.0
                # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                # Derivative of the "C+2" Residual**************************************************************************************
                dF[(comp + 1) * (comp + 2):] = 0.0
                dF[(comp + 1) * (comp + 2) + SpecVar] = 1.0

                # besides the specified variable are null. Its derivative with respect to the specified variable is 1.
                # **********************************************************************************************************************

                # Solving The System of Equations///////////////////////////////////////////////////////////////////////////////////////
                # call GaussElimination(dF,F,step,comp+1)
                A = dF.reshape(((comp + 2), (comp + 2)), order="C")
                step = self.solve(A, F)
                # step = np.linalg.solve(A, F)
                # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                # print( "VAR", Var, "K", K, "P", P, "T", T, "step", step, "Var_old", Var_old

                # Updating The Independent Variables************************************************************************************
                Var = Var - step
                maxstep = max(abs(step/Var))
                # **********************************************************************************************************************

                # Calculating The Natural Form Of Indep# endent Variables And Updating Compositions Of The Incipient Phase////////////////
                K = np.exp(Var[0:comp])
                Composition[:,1] = Composition[:, 0] * K
                T = np.exp(Var[comp + 0])
                P = np.exp(Var[comp + 1])

                # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            print("Incipient Phase = ", phase[1] + 1, "    P = ", P, "T = ", T, "SpecVar =", SpecVar)

            if maxstep > tol:
                flag_error = 1
            else:
                for i in range(comp + 2):
                    if (Var[i] != Var[i]):  # then
                        flag_error = 1
                        exit()

            if flag_error == 0:
                if flag_crit == 2:
                    flag_crit = 0

                print(phase[0] + 1, ",", P, ",", T, ",", (Composition[1, 1], ",", 1, comp))
                phase_envelope_results.append({
                    "pressure": P,
                    "temperature": T,
                    "composition": list(Composition.flatten()),
                })
                # print(T, P)

                # Analyzing Sensitivity Of The Indep# endent Variables************************************************************************
                SpecVar_old = SpecVar
                Var_old = Var

                # Sensitivity Vector Calculation
                # call GaussElimination(dF,dfdS,sensitivity,comp+1)
                A = dF.reshape(((comp + 2), (comp + 2)), order="C")
                sensitivity = self.solve(A, dfdS)
                # sensitivity = np.linalg.solve(A, dfdS)

                if (flag_crit == 0):  # then
                    # Choosing The New Specified Indep# endent Variable
                    for i in range(comp + 2):
                        if (np.abs(sensitivity[i]) > np.abs(sensitivity[SpecVar])):
                            SpecVar = i
                    # enddo
                    # OBS: The specified variable is the one with the greatest sensitivity,
                    # i.e. the one which its variation makes the system varies more intensely.

                    # Updating Specified Variable
                    if (SpecVar != SpecVar_old):  # then
                        dS = dS * sensitivity[SpecVar]
                        for i in range(comp + 2):
                            if (i != SpecVar):
                                sensitivity[i] = sensitivity[i] / sensitivity[SpecVar]
                        # enddo
                        sensitivity[SpecVar] = 1.0
                        S = Var[SpecVar]
                    # endif
                    # **************************************************************************************************************************

                    # Adjusting Stepsize////////////////////////////////////////////////////////////////////////////////////////////////////////
                    dSmax = (np.abs(Var[SpecVar]) ** 0.5) / 10.0
                    if (dSmax < 0.1):
                        dSmax = 0.1

                    dSmax = np.abs(dSmax) * (np.abs(dS) / dS)
                    dS = dS * 4.0 / it
                    if (np.abs(dSmax) < np.abs(dS)):
                        dS = dSmax
                    # Defining Specified Variable Value In The Next Point
                    S = S + dS
                    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    # Indep# endent Variables Initial Guess For The Next Point********************************************************************
                    for i in range(comp + 2):
                        Var[i] = Var[i] + dS * sensitivity[i]
                    # enddo
                    # **************************************************************************************************************************

                    # Analyzing Temperature Stepsize////////////////////////////////////////////////////////////////////////////////////////////
                    T_old = T
                    T = np.exp(Var[comp + 0])
                    # Large Temperature Steps Are Not Advisable
                    while (np.abs(T - T_old) > maxTstep):
                        dS = dS / 2.0
                        S = S - dS
                        for i in range(comp + 2):
                            Var[i] = Var[i] - dS * sensitivity[i]
                        # enddo
                        T = np.exp(Var[comp + 0])
                    # enddo
                    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                    # Analyzing Proximity to Critical Point*************************************************************************************
                    # Seeking The Greatest K-factor
                    maxK_i = np.argmax(abs(Var[0:comp]))
                    # If The ln[K[i]] Stepsize Is Too Big, It Should Be Decreased
                    if (np.abs(Var[maxK_i]) < 0.1):  # then
                        # Analyzing maxK stepsize
                        if (np.abs(dS * sensitivity[maxK_i]) > K_CritPoint):  # then
                            S = S - dS
                            for i in range(comp + 2):
                                Var[i] = Var[i] - dS * sensitivity[i]
                            # enddo

                            # Shortening ln(K) Stepsize
                            dS = (dS / abs(dS)) * K_CritPoint / abs(sensitivity[maxK_i])

                            S = S + dS
                            for i in range(comp + 2):
                                Var[i] = Var[i] + dS * sensitivity[i]
                            # enddo
                        # endif
                        # OBS: The current point must be near enough to the critical point
                        # so that the algorithm can pass through it without diverging.

                        if (abs(Var[maxK_i]) < K_CritPoint):  # then
                            # This is gonna be the last point before the critical point
                            S = S - dS
                            for i in range(comp + 2):
                                Var[i] = Var[i] - dS * sensitivity[i]
                            # enddo

                            dS = (K_CritPoint - Var[maxK_i]) / sensitivity[maxK_i]

                            S = S + dS
                            for i in range(comp + 2):
                                Var[i] = Var[i] + dS * sensitivity[i]
                            # enddo
                            flag_crit = 1
                        # endif
                    # endif
                else:
                    # Passing Through The Critical Point
                    dS = K_CritPoint * dS / abs(dS)
                    S = S + 2. * dS
                    for i in range(comp + 2):
                        Var[i] = Var[i] + 2. * dS * sensitivity[i]
                    # enddo

                    # Defining Incipient Phase As Vapor - Initializing Bubble Curve
                    phase_aux = phase[0]
                    phase[0] = phase[1]
                    phase[1] = phase_aux
                    # OBS: This will cause the definition of the K-factors to change from
                    # FugCoef(Vap phase)/FugCoef(Liq phase) to FugCoef(Liq phase)/FugCoef(Vap phase).

                    flag_crit = 2
                # endif

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
                Var[:comp + 2] += dS * sensitivity[:comp + 2]
                flag_crit = 1
                flag_error = 0

            elif flag_error == 1 and flag_crit == 0 and abs(dS) > 1e-6:
                Var = Var_old
                S = Var[SpecVar]
                dS = dS / 4.0
                S = S + dS
                Var[:comp + 2] += dS * sensitivity[:comp + 2]
                flag_error = 0

            K = np.exp(Var[0:comp])
            Composition[:, 1] = Composition[:, 0] * K

            T = np.exp(Var[comp + 0])
            P = np.exp(Var[comp + 1])

        return phase_envelope_results


    @staticmethod
    def solve(A:np.ndarray, b: np.ndarray):
        return np.linalg.solve(A, b)
    # return np.linalg.lstsq(A, b)
    #return np.dot(np.linalg.inv(A), b)
