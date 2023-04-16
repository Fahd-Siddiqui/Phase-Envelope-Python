# import numpy
#
# from PhaseEnvelope.src.eos import EOS
#
#
# class PhaseEnvelope:
#
#     #@numba.jit(nopython=True)
#     def calculate(self, comp, z, Tc, Pc, acentric):
#         R = 83.14462175  # cm³.bar/(mol.K)
#         Volume = numpy.zeros(2)
#         amix = numpy.zeros(2)
#         bmix = numpy.zeros(2)
#
#         phase_envelope_results = []
#
#         b = numpy.zeros(comp)
#         ac = numpy.zeros(comp)
#
#         K = numpy.zeros(comp)
#         Composition = numpy.zeros((comp, comp))
#         kij = numpy.zeros((comp, comp))
#         lij = numpy.zeros((comp, comp))
#
#         dF = numpy.zeros((comp + 2) ** 2)
#         step = numpy.zeros(comp + 2)
#         dfdS = numpy.zeros(comp + 2)
#         Var = numpy.zeros(comp + 2)
#         sensitivity = numpy.zeros(comp + 2)
#         F = numpy.zeros(comp + 2)
#         #     Indep# endent Variables     #
#         # (F-1)*C          K        #
#         #  (F-1)         beta         #
#         #   1         temperature     #
#         #   1          pressure       #
#
#         # Reading And Calculating Properties********************************************************************************************
#         aux = 0.0
#         for i in range(comp):
#             # Reading Global Composition, Critical Temperature, Critical Pressure and Acentric Factor
#             # read(input_num,*) z[i], Tc[i], Pc[i], acentric[i]
#
#             # EoS Parameters Calculation
#             b[i] = 0.07780 * R * Tc[i] / Pc[i]  # covolume
#             ac[i] = 0.45724 * R * R * Tc[i] * Tc[i] / Pc[i]
#             aux = aux + z[i]
#         # enddo
#
#         # ******************************************************************************************************************************
#
#         # Initial Settings - Dew point ///////////////////////////////////////////////////////////////////////////////////////////////////
#         T = 80.0  # Initial Temperature Guess (K)
#         P = 0.5  # Initial Pressure (bar)
#
#         for i in range(comp):
#             # K[i] = 1.0/numpy.exp(numpy.log(Pc[i]/P) + 5.373*(1.0 + acentric[i])*(1.0 - Tc[i]/T)) #Whitson's Approach for Vapor-Liquid Equilibria
#             K[i] = numpy.exp(5.373 * (1.0 + acentric[i]) * (1.0 - Tc[i] / T)) * (Pc[i] / P)  # Whitson's Approach for Vapor-Liquid Equilibria
#             z[i] = z[i] / aux  # Normalizing Global Composition
#             Composition[i, 0] = z[i]  # Reference Phase Composition
#         # enddo
#         phase = [0, 1]
#         # phase[0] = 0 #Reference Phase Index (Vapor)
#         # phase[1] = 1 #Incipient Phase Index (Liquid)
#         # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
#         # Determining Temperature Initial Guess (Point Near To The Dew Curve)***********************************************************
#         T_old = T - 1.0
#         while (T_old != T):
#             T_old = T
#
#             a = EOS.eos_parameters(acentric, Tc, ac, T)  # Updating Attractive Parameter
#             amix[0], bmix[0] = EOS.VdW1fMIX(comp, a, b, kij, lij, z)  # Mixing Rule
#             Volume[0] = EOS.EoS_Volume(P, T, bmix[0], amix[0], 0)
#             Volume[1] = EOS.EoS_Volume(P, T, bmix[0], amix[0], 1)
#             # Calculating Gibbs Energy Of Vapor And Liquid Phases
#             Gibbs_vap = 0.0
#             Gibbs_liq = 0.0
#             for i in range(comp):
#                 FugCoef_ref = EOS.fugacity(T, P, a, b, amix[0], bmix[0], Volume[0], z, kij[i, :], lij[i, :], i)
#                 Gibbs_vap = Gibbs_vap + z[i] * FugCoef_ref
#                 FugCoef_aux = EOS.fugacity(T, P, a, b, amix[0], bmix[0], Volume[1], z, kij[i, :], lij[i, :], i)
#                 Gibbs_liq = Gibbs_liq + z[i] * FugCoef_aux
#             # enddo
#
#             # If Exists A Pure Liquid Or A Liquid-Vapor Equilibrium Nearer To The Bubble Point
#             # Than To The Dew Point, The Temperature Is Increased.
#             if ((Gibbs_liq < Gibbs_vap) or (Gibbs_liq == Gibbs_vap and (Volume[0] / bmix[0]) < 1.75)):  # then
#                 T = T + 10.0
#             # endif
#             # OBS: The test comparing the ratio between the volume and the covolume to the constant factor 1.75 was proposed by
#             # Pedersen and Christensen (Phase Behavior Of Petroleum Reservoir Fluids, 2007 - Chapter 6.5 Phase Identification)
#         # enddo
#         # ******************************************************************************************************************************
#
#         # Successive Substitution///////////////////////////////////////////////////////////////////////////////////////////////////////
#         tol = 1.0e-8
#         diff = 1.0e-8
#         step[0] = 1.0e+6
#         maxit = 100
#         it = 0
#         while (numpy.abs(step[0]) > tol and it < maxit):
#             it = it + 1
#             T_old = T
#
#             aux = 0.0
#             for i in range(comp):
#                 Composition[i, 1] = Composition[i, 0] * K[i]  # Incipient Phase Composition
#                 aux = aux + Composition[i, 1]
#             # enddo
#
#             F[0] = -1.0
#             dF[0] = 0.0
#
#             for i in range(comp):
#                 Composition[i, 1] = Composition[i, 1] / aux  # Normalizing Composition
#                 F[0] = F[0] + Composition[i, 0] * K[i]  # Residual
#             # enddo
#
#             # Numerical Derivative With Respect to Temperature
#             T = T_old + diff
#             a = EOS.eos_parameters(acentric, Tc, ac, T)  # Updating Attractive Parameter
#             amix[0], bmix[0] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 0])  # Mixing Rule - Reference Phase
#             amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
#             Volume[0] = EOS.EoS_Volume(P, T, bmix[0], amix[0], 0)
#             Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], 1)
#             for i in range(comp):
#                 FugCoef_ref = EOS.fugacity(T, P, a, b, amix[0], bmix[0], Volume[0], Composition[:, 0], kij[i, :], lij[i, :], i)
#                 FugCoef_aux = EOS.fugacity(T, P, a, b, amix[1], bmix[1], Volume[1], Composition[:, 1], kij[i, :], lij[i, :], i)
#                 dF[0] = dF[0] + Composition[i, 0] * K[i] * (FugCoef_ref - FugCoef_aux)
#             # enddo
#
#             T = T_old - diff
#             a = EOS.eos_parameters(acentric, Tc, ac, T)   # Updating Attractive Parameter
#             amix[0], bmix[0] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 0])  # Mixing Rule - Reference Phase
#             amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
#             Volume[0] = EOS.EoS_Volume(P, T, bmix[0], amix[0], 0)
#             Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], 1)
#             for i in range(comp):
#                 FugCoef_ref = EOS.fugacity(T, P, a, b, amix[0], bmix[0], Volume[0], Composition[:, 0], kij[i, :], lij[i, :], i)
#                 FugCoef_aux = EOS.fugacity(T, P, a, b, amix[1], bmix[1], Volume[1], Composition[:, 1], kij[i, :], lij[i, :], i)
#                 dF[0] = dF[0] - Composition[i, 0] * K[i] * (FugCoef_ref - FugCoef_aux)
#             # enddo
#
#             dF[0] = dF[0] / (2.0 * diff)
#
#             # Temperature Step Calculation
#             step[0] = F[0] / dF[0]
#
#             # Step Brake
#             if (numpy.abs(step[0]) > 0.25 * T_old):
#                 step[0] = 0.25 * T_old * step[0] / numpy.abs(step[0])
#
#                 # Updating Temperature
#             T = T_old - step[0]
#
#             # Updating K-factors
#             a = EOS.eos_parameters(acentric, Tc, ac, T)
#             amix[0], bmix[0] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 0])  # Mixing Rule - Reference Phase
#             amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
#             Volume[0] = EOS.EoS_Volume(P, T, bmix[0], amix[0], 0)
#             Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], 1)
#             for i in range(comp):
#                 FugCoef_ref = EOS.fugacity(T, P, a, b, amix[0], bmix[0], Volume[0], Composition[:, 0], kij[i, :], lij[i, :], i)
#                 FugCoef_aux = EOS.fugacity(T, P, a, b, amix[1], bmix[1], Volume[1], Composition[:, 1], kij[i, :], lij[i, :], i)
#                 K[i] = numpy.exp(FugCoef_ref - FugCoef_aux)
#             # enddo
#         # enddo
#
#         if (it == maxit and numpy.abs(step[0]) > tol):  # then
#             print("WARNING: In Successive Substitution Method - Maximum Number of Iterations Reached#")
#             print("Exiting Program...")
#             exit()
#         # endif
#         # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
#         # Continuation Method And Newton's Method Settings******************************************************************************
#         S = numpy.log(P)  # Specified Variable Value
#         dS = 0.1  # Specified Variable Variation
#         for i in range(comp):
#             Var[i] = numpy.log(K[i])
#         # enddo
#         Var[comp + 0] = numpy.log(T)
#         SpecVar = comp + 1  # Specified Variable Index
#         Var[SpecVar] = S  # Specified Indep# endent Variable
#         # Var_old = Var
#         for i in range(comp + 1):
#             dfdS[i] = 0.0
#         # enddo
#         dfdS[comp + 1] = 1.0
#         # tol = 1.0d-6
#         # diff = 1.0d-6
#         maxit = 100  # Maximum Number Of Iteration In Newton's Method
#         maxTstep = 5.0  # Maximum Temperature Step In Continuation Method
#         K_CritPoint = 0.04  # K-factor Reference Value Used To Detect Critical Points
#         flag_crit = 0  # if calculating the critical point, flag_crit = 1
#         flag_error = 0
#         # ******************************************************************************************************************************
#
#         # Initializing Continuation Method
#         point = 0
#         while (P >= 0.5 and P < 1000. and flag_error == 0):  # Main Loop
#             point = point + 1
#             it = 0
#             maxstep = 1e+6
#             while (maxstep > tol and it < maxit):  # Newton's Method Loop
#                 it = it + 1
#
#                 # Calculating Residuals/////////////////////////////////////////////////////////////////////////////////////////////////
#                 a = EOS.eos_parameters(acentric, Tc, ac, T)
#                 amix[0], bmix[0] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 0])  # Mixing Rule - Reference Phase
#                 amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
#                 Volume[0] = EOS.EoS_Volume(P, T, bmix[0], amix[0], phase[0])
#                 Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], phase[1])
#                 F[comp + 0] = 0.0
#                 for i in range(comp):
#                     FugCoef_ref = EOS.fugacity(T, P, a, b, amix[0], bmix[0], Volume[0], Composition[:, 0], kij[i, :], lij[i, :], i)
#                     FugCoef_aux = EOS.fugacity(T, P, a, b, amix[1], bmix[1], Volume[1], Composition[:, 1], kij[i, :], lij[i, :], i)
#                     # Residual Responsible For The Chemical Equilibrium
#                     F[i] = Var[i] + (FugCoef_aux - FugCoef_ref)
#                     # Residual Responsible For Assuring That The Summation Of The Incipient Phase Equals 1
#                     F[comp + 0] = F[comp + 0] + Composition[i, 1] - Composition[i, 0]
#                     # enddo
#                 # Residual Responsible For Determining The Specified Indep# endent Variable
#                 F[comp + 1] = Var[SpecVar] - S
#                 # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
#                 # Differentiating The First "C" Residuals With Respect to ln[K(j)]******************************************************
#                 for i in range(comp):
#                     for j in range(comp):
#                         diffFrac = diff * Composition[j, 1]
#
#                         aux = Composition[j, 1]
#                         # Numerically Differentiating the Fugacity Coefficient of the Incipient Phase
#                         Composition[j, 1] = aux + diffFrac
#                         amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
#                         Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], phase[1])
#                         FugCoef_aux = EOS.fugacity(T, P, a, b, amix[1], bmix[1], Volume[1], Composition[:, 1], kij[i, :], lij[i, :], i)
#                         dF[(i) * (comp + 2) + j] = FugCoef_aux
#                         Composition[j, 1] = aux - diffFrac
#                         amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
#                         Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], phase[1])
#                         FugCoef_aux = EOS.fugacity(T, P, a, b, amix[1], bmix[1], Volume[1], Composition[:, 1], kij[i, :], lij[i, :], i)
#                         dF[(i) * (comp + 2) + j] = dF[(i) * (comp + 2) + j] - FugCoef_aux
#
#                         Composition[j, 1] = aux
#                         # Derivative of ln[FugacityCoefficient(IncipientPhase,Component i)] With Respect to ln[K(j)]
#                         dF[(i) * (comp + 2) + j] = dF[(i) * (comp + 2) + j] * Composition[j, 1] / (2.0 * diffFrac)
#
#                         # Derivative of ln[K[i]] With Respect to ln[K(j)] = Kronecker Delta
#                         if (i == j):
#                             dF[(i) * (comp + 2) + j] = dF[(i) * (comp + 2) + j] + 1.0
#                     # enddo
#                 # enddo
#                 # **********************************************************************************************************************
#
#                 # Differentiating "C+1" Residual With Respect to ln[K[i]]///////////////////////////////////////////////////////////////
#                 for i in range(comp):
#                     dF[comp * (comp + 2) + i] = Composition[i, 1]
#                 # enddo
#                 # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
#                 # Differentiating The First "C" Residuals With Respect to ln[T]*********************************************************
#                 diffT = diff * Var[comp + 0]
#
#                 # Numerically Differentiating The ln(FugacityCoefficient) With Respect to ln(T)
#                 T = numpy.exp(Var[comp + 0] + diffT)
#                 a = EOS.eos_parameters(acentric, Tc, ac, T)
#                 amix[0], bmix[0] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 0])  # Mixing Rule - Reference Phase
#                 amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
#                 Volume[0] = EOS.EoS_Volume(P, T, bmix[0], amix[0], phase[0])
#                 Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], phase[1])
#                 for i in range(comp):
#                     FugCoef_ref = EOS.fugacity(T, P, a, b, amix[0], bmix[0], Volume[0], Composition[:, 0], kij[i, :], lij[i, :], i)
#                     FugCoef_aux = EOS.fugacity(T, P, a, b, amix[1], bmix[1], Volume[1], Composition[:, 1], kij[i, :], lij[i, :], i)
#                     dF[(i) * (comp + 2) + comp + 0] = FugCoef_aux - FugCoef_ref
#                 # enddo
#
#                 T = numpy.exp(Var[comp + 0] - diffT)
#                 a = EOS.eos_parameters(acentric, Tc, ac, T)
#                 amix[0], bmix[0] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 0])  # Mixing Rule - Reference Phase
#                 amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
#                 Volume[0] = EOS.EoS_Volume(P, T, bmix[0], amix[0], phase[0])
#                 Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], phase[1])
#                 for i in range(comp):
#                     FugCoef_ref = EOS.fugacity(T, P, a, b, amix[0], bmix[0], Volume[0], Composition[:, 0], kij[i, :], lij[i, :], i)
#                     FugCoef_aux = EOS.fugacity(T, P, a, b, amix[1], bmix[1], Volume[1], Composition[:, 1], kij[i, :], lij[i, :], i)
#                     dF[(i) * (comp + 2) + comp + 0] = dF[(i) * (comp + 2) + comp + 0] - (FugCoef_aux - FugCoef_ref)
#                 # enddo
#
#                 for i in range(comp):
#                     dF[(i) * (comp + 2) + comp + 0] = dF[(i) * (comp + 2) + comp + 0] / (2.0 * diffT)
#                 # enddo
#
#                 T = numpy.exp(Var[comp + 0])
#                 a = EOS.eos_parameters(acentric, Tc, ac, T)
#                 # OBS: The derivative ok ln(K) with respect to ln(T) is null.
#                 # **********************************************************************************************************************
#
#                 # Differentiating The First "C" Residuals With Respect to ln[P]/////////////////////////////////////////////////////////
#                 diffP = diff * Var[comp + 1]
#
#                 amix[0], bmix[0] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 0])  # Mixing Rule - Reference Phase
#                 amix[1], bmix[1] = EOS.VdW1fMIX(comp, a, b, kij, lij, Composition[:, 1])  # Mixing Rule - Incipient Phase
#
#                 # Numerically Differentiating The ln(FugacityCoefficient) With Respect to ln(T)
#                 P = numpy.exp(Var[comp + 1] + diffP)
#                 Volume[0] = EOS.EoS_Volume(P, T, bmix[0], amix[0], phase[0])
#                 Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], phase[1])
#                 for i in range(comp):
#                     FugCoef_ref = EOS.fugacity(T, P, a, b, amix[0], bmix[0], Volume[0], Composition[:, 0], kij[i, :], lij[i, :], i)
#                     FugCoef_aux = EOS.fugacity(T, P, a, b, amix[1], bmix[1], Volume[1], Composition[:, 1], kij[i, :], lij[i, :], i)
#                     dF[(i) * (comp + 2) + comp + 1] = FugCoef_aux - FugCoef_ref
#                 # enddo
#
#                 P = numpy.exp(Var[comp + 1] - diffP)
#                 Volume[0] = EOS.EoS_Volume(P, T, bmix[0], amix[0], phase[0])
#                 Volume[1] = EOS.EoS_Volume(P, T, bmix[1], amix[1], phase[1])
#                 for i in range(comp):
#                     FugCoef_ref = EOS.fugacity(T, P, a, b, amix[0], bmix[0], Volume[0], Composition[:, 0], kij[i, :], lij[i, :], i)
#                     FugCoef_aux = EOS.fugacity(T, P, a, b, amix[1], bmix[1], Volume[1], Composition[:, 1], kij[i, :], lij[i, :], i)
#                     dF[(i) * (comp + 2) + comp + 1] = dF[(i) * (comp + 2) + comp + 1] - (FugCoef_aux - FugCoef_ref)
#                 # enddo
#                 for i in range(comp):
#                     dF[(i) * (comp + 2) + comp + 1] = dF[(i) * (comp + 2) + comp + 1] / (2.0 * diffP)
#                 # enddo
#
#                 P = numpy.exp(Var[comp + 1])
#                 # OBS: The derivative ok ln(K) with respect to ln(P) is null.
#                 # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
#                 # Derivative of the "C+1" Residual With Respect to ln(T)****************************************************************
#                 dF[comp * (comp + 2) + comp + 0] = 0.0
#                 # **********************************************************************************************************************
#
#                 # Derivative of the "C+1" Residual With Respect to ln(P)////////////////////////////////////////////////////////////////
#                 dF[comp * (comp + 2) + comp + 1] = 0.0
#                 # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
#                 # Derivative of the "C+2" Residual**************************************************************************************
#                 for i in range(comp + 2):
#                     dF[(comp + 1) * (comp + 2) + i] = 0.0
#                 # enddo
#                 dF[(comp + 1) * (comp + 2) + SpecVar] = 1.0
#                 # OBS: The derivative of the specification equation with respect to all indep# endent variables
#                 # besides the specified variable are null. Its derivative with respect to the specified variable is 1.
#                 # **********************************************************************************************************************
#
#                 # Solving The System of Equations///////////////////////////////////////////////////////////////////////////////////////
#                 # call GaussElimination(dF,F,step,comp+1)
#                 A = dF.reshape(((comp + 2), (comp + 2)), order="C")
#                 step = numpy.linalg.solve(A, F)
#                 # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                 # print( "VAR", Var, "K", K, "P", P, "T", T, "step", step, "Var_old", Var_old
#
#                 # Updating The Indep# endent Variables************************************************************************************
#                 maxstep = 0.0
#                 for i in range(comp + 2):
#                     Var[i] = Var[i] - step[i]
#                     # Calculating Variable Used As Convergence Criteria
#                     if (maxstep < numpy.abs(step[i] / Var[i])):
#                         maxstep = numpy.abs(step[i] / Var[i])
#                 # enddo
#                 # **********************************************************************************************************************
#
#                 # Calculating The Natural Form Of Indep# endent Variables And Updating Compositions Of The Incipient Phase////////////////
#                 for i in range(comp):
#                     K[i] = numpy.exp(Var[i])
#                     Composition[i, 1] = Composition[i, 0] * K[i]
#                 # enddo
#                 T = numpy.exp(Var[comp + 0])
#                 P = numpy.exp(Var[comp + 1])
#
#                 # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
#             # enddo #End of Newton's Method Loop
#
#             # write(*,*) "Incipient Phase = ",leg# end_ELV(phase[1]+1),"    P = ",P,"T = ",T, "SpecVar =",SpecVar
#             print("Incipient Phase = ", phase[1] + 1, "    P = ", P, "T = ", T, "SpecVar =", SpecVar)
#
#             if (maxstep > tol):  # then
#                 flag_error = 1
#             else:
#                 for i in range(comp + 2):
#                     if (Var[i] != Var[i]):  # then
#                         flag_error = 1
#                         exit()
#                     # endif
#                 # enddo
#             # endif
#
#             if (flag_error == 0):  # then
#                 if (flag_crit == 2):
#                     flag_crit = 0
#
#                 print(phase[0] + 1, ",", P, ",", T, ",", (Composition[1, 1], ",", 1, comp))
#                 phase_envelope_results.append({
#                     "pressure": P,
#                     "temperature": T,
#                     "composition": list(Composition.flatten()),
#                 })
#                 # print(T, P)
#
#                 # Analyzing Sensitivity Of The Indep# endent Variables************************************************************************
#                 SpecVar_old = SpecVar
#                 Var_old = Var
#
#                 # Sensitivity Vector Calculation
#                 # call GaussElimination(dF,dfdS,sensitivity,comp+1)
#                 A = dF.reshape(((comp + 2), (comp + 2)), order="C")
#                 sensitivity = numpy.linalg.solve(A, dfdS)
#
#                 if (flag_crit == 0):  # then
#                     # Choosing The New Specified Indep# endent Variable
#                     for i in range(comp + 2):
#                         if (numpy.abs(sensitivity[i]) > numpy.abs(sensitivity[SpecVar])):
#                             SpecVar = i
#                     # enddo
#                     # OBS: The specified variable is the one with the greatest sensitivity,
#                     # i.e. the one which its variation makes the system varies more intensely.
#
#                     # Updating Specified Variable
#                     if (SpecVar != SpecVar_old):  # then
#                         dS = dS * sensitivity[SpecVar]
#                         for i in range(comp + 2):
#                             if (i != SpecVar):
#                                 sensitivity[i] = sensitivity[i] / sensitivity[SpecVar]
#                         # enddo
#                         sensitivity[SpecVar] = 1.0
#                         S = Var[SpecVar]
#                     # endif
#                     # **************************************************************************************************************************
#
#                     # Adjusting Stepsize////////////////////////////////////////////////////////////////////////////////////////////////////////
#                     dSmax = (numpy.abs(Var[SpecVar]) ** 0.5) / 10.0
#                     if (dSmax < 0.1):
#                         dSmax = 0.1
#
#                     dSmax = numpy.abs(dSmax) * (numpy.abs(dS) / dS)
#                     dS = dS * 4.0 / it
#                     if (numpy.abs(dSmax) < numpy.abs(dS)):
#                         dS = dSmax
#                     # Defining Specified Variable Value In The Next Point
#                     S = S + dS
#                     # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
#                     # Indep# endent Variables Initial Guess For The Next Point********************************************************************
#                     for i in range(comp + 2):
#                         Var[i] = Var[i] + dS * sensitivity[i]
#                     # enddo
#                     # **************************************************************************************************************************
#
#                     # Analyzing Temperature Stepsize////////////////////////////////////////////////////////////////////////////////////////////
#                     T_old = T
#                     T = numpy.exp(Var[comp + 0])
#                     # Large Temperature Steps Are Not Advisable
#                     while (numpy.abs(T - T_old) > maxTstep):
#                         dS = dS / 2.0
#                         S = S - dS
#                         for i in range(comp + 2):
#                             Var[i] = Var[i] - dS * sensitivity[i]
#                         # enddo
#                         T = numpy.exp(Var[comp + 0])
#                     # enddo
#                     # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
#                     # Analyzing Proximity to Critical Point*************************************************************************************
#                     # Seeking The Greatest K-factor
#                     maxK_i = numpy.argmax(abs(Var[0:comp]))
#                     # If The ln[K[i]] Stepsize Is Too Big, It Should Be Decreased
#                     if (numpy.abs(Var[maxK_i]) < 0.1):  # then
#                         # Analyzing maxK stepsize
#                         if (numpy.abs(dS * sensitivity[maxK_i]) > K_CritPoint):  # then
#                             S = S - dS
#                             for i in range(comp + 2):
#                                 Var[i] = Var[i] - dS * sensitivity[i]
#                             # enddo
#
#                             # Shortening ln(K) Stepsize
#                             dS = (dS / abs(dS)) * K_CritPoint / abs(sensitivity[maxK_i])
#
#                             S = S + dS
#                             for i in range(comp + 2):
#                                 Var[i] = Var[i] + dS * sensitivity[i]
#                             # enddo
#                         # endif
#                         # OBS: The current point must be near enough to the critical point
#                         # so that the algorithm can pass through it without diverging.
#
#                         if (abs(Var[maxK_i]) < K_CritPoint):  # then
#                             # This is gonna be the last point before the critical point
#                             S = S - dS
#                             for i in range(comp + 2):
#                                 Var[i] = Var[i] - dS * sensitivity[i]
#                             # enddo
#
#                             dS = (K_CritPoint - Var[maxK_i]) / sensitivity[maxK_i]
#
#                             S = S + dS
#                             for i in range(comp + 2):
#                                 Var[i] = Var[i] + dS * sensitivity[i]
#                             # enddo
#                             flag_crit = 1
#                         # endif
#                     # endif
#                 else:
#                     # Passing Through The Critical Point
#                     dS = K_CritPoint * dS / abs(dS)
#                     S = S + 2. * dS
#                     for i in range(comp + 2):
#                         Var[i] = Var[i] + 2. * dS * sensitivity[i]
#                     # enddo
#
#                     # Defining Incipient Phase As Vapor - Initializing Bubble Curve
#                     phase_aux = phase[0]
#                     phase[0] = phase[1]
#                     phase[1] = phase_aux
#                     # OBS: This will cause the definition of the K-factors to change from
#                     # FugCoef(Vap phase)/FugCoef(Liq phase) to FugCoef(Liq phase)/FugCoef(Vap phase).
#
#                     flag_crit = 2
#                 # endif
#
#             elif (flag_error == 1 and flag_crit == 2 and K_CritPoint > 0.009):  # then
#                 K_CritPoint = K_CritPoint - 0.005
#                 Var = Var_old
#                 S = Var[SpecVar]
#                 phase_aux = phase[0]
#                 phase[0] = phase[1]
#                 phase[1] = phase_aux
#                 dS = (K_CritPoint - Var[maxK_i]) / sensitivity[maxK_i]
#
#                 S = S + dS
#                 for i in range(comp + 2):
#                     Var[i] = Var[i] + dS * sensitivity[i]
#                 # enddo
#
#                 flag_crit = 1
#                 flag_error = 0
#             elif (flag_error == 1 and flag_crit == 0 and abs(dS) > 1e-6):  # then
#                 Var = Var_old
#                 S = Var[SpecVar]
#                 dS = dS / 4.
#
#                 S = S + dS
#                 for i in range(comp + 2):
#                     Var[i] = Var[i] + dS * sensitivity[i]
#                 # enddo
#
#                 flag_error = 0
#             # endif
#
#             for i in range(comp):
#                 K[i] = numpy.exp(Var[i])
#                 Composition[i, 1] = Composition[i, 0] * K[i]
#             # enddo
#             T = numpy.exp(Var[comp + 0])
#             P = numpy.exp(Var[comp + 1])
#         # enddo #End of  the main loop
#
#         return phase_envelope_results
