import unittest

import numpy

from src.phase_envelope import PhaseEnvelope
from src.successive_substitution import SuccessiveSubstitution


class TestPhaseEnvelope(unittest.TestCase):
    def test_phase_envelope_simple(self):
        self.maxDiff=None
        expected_res = [
            {'pressure': 0.5, 'temperature': 451.31807134411906, 'mole_fractions': [0.9, 0.0030316630751809886, 0.1, 0.99696833692482]}, {'pressure': 0.5525854590378239, 'temperature': 454.0221093121738, 'mole_fractions': [0.9, 0.003319307908447115, 0.1, 0.9966806920915531]},
            {'pressure': 0.6107013790800849, 'temperature': 456.7641623942874, 'mole_fractions': [0.9, 0.003634921798497187, 0.1, 0.9963650782015027]}, {'pressure': 0.6749294037880015, 'temperature': 459.5448350699871, 'mole_fractions': [0.9, 0.003981313103529249, 0.1, 0.9960186868964709]},
            {'pressure': 0.7459123488206351, 'temperature': 462.36471736910266, 'mole_fractions': [0.9, 0.004361583913199878, 0.1, 0.9956384160868001]}, {'pressure': 0.824360635350064, 'temperature': 465.22438088550575, 'mole_fractions': [0.9, 0.0047791623009073485, 0.1, 0.9952208376990925]},
            {'pressure': 0.9110594001952544, 'temperature': 468.12437427364546, 'mole_fractions': [0.9, 0.005237838305004741, 0.1, 0.9947621616949953]}, {'pressure': 1.0068763537352383, 'temperature': 471.06521816798124, 'mole_fractions': [0.9, 0.005741804089297174, 0.1, 0.994258195910703]},
            {'pressure': 1.1127704642462337, 'temperature': 474.0473994584889, 'mole_fractions': [0.9, 0.006295698788373571, 0.1, 0.9937043012116261]}, {'pressure': 1.2298015555784747, 'temperature': 477.0713648476963, 'mole_fractions': [0.9, 0.006904658604876922, 0.1, 0.9930953413951233]},
            {'pressure': 1.3591409142295225, 'temperature': 480.1375136059902, 'mole_fractions': [0.9, 0.0075743727942478144, 0.1, 0.9924256272057521]}, {'pressure': 1.5020830119732165, 'temperature': 483.24618943219775, 'mole_fractions': [0.9, 0.00831114624825828, 0.1, 0.991688853751742]},
            {'pressure': 1.6600584613682736, 'temperature': 486.3976713154605, 'mole_fractions': [0.9, 0.009121969472170388, 0.1, 0.9908780305278295]}, {'pressure': 1.834648333809622, 'temperature': 489.59216328204093, 'mole_fractions': [0.9, 0.010014596841845812, 0.1, 0.9899854031581539]},
            {'pressure': 2.0275999834223373, 'temperature': 492.82978289675407, 'mole_fractions': [0.9, 0.010997634126592515, 0.1, 0.9890023658734073]}, {'pressure': 2.2408445351690323, 'temperature': 496.11054837295796, 'mole_fractions': [0.9, 0.0120806363706129, 0.1, 0.9879193636293873]},
            {'pressure': 2.476516212197557, 'temperature': 499.4343641271837, 'mole_fractions': [0.9, 0.013274217339746559, 0.1, 0.9867257826602535]}, {'pressure': 2.7369736958635995, 'temperature': 502.8010045942671, 'mole_fractions': [0.9, 0.014590171859187075, 0.1, 0.9854098281408131]},
            {'pressure': 3.025858568169659, 'temperature': 506.22182937858616, 'mole_fractions': [0.9, 0.0160468260050117, 0.1, 0.9839531739949883]}, {'pressure': 3.3616038735250244, 'temperature': 509.85437527285745, 'mole_fractions': [0.9, 0.017736978860518293, 0.1, 0.9822630211394818]},
            {'pressure': 3.7528969692308207, 'temperature': 513.7043542154368, 'mole_fractions': [0.9, 0.01970425756484555, 0.1, 0.9802957424351542]}, {'pressure': 4.21028056149315, 'temperature': 517.7771759945728, 'mole_fractions': [0.9, 0.02200190934413561, 0.1, 0.9779980906558645]},
            {'pressure': 4.746589198274063, 'temperature': 522.0777086225577, 'mole_fractions': [0.9, 0.02469517870602692, 0.1, 0.9753048212939733]}, {'pressure': 5.377497855875088, 'temperature': 526.6099571877459, 'mole_fractions': [0.9, 0.027864328197785507, 0.1, 0.9721356718022147]},
            {'pressure': 6.122213598423263, 'temperature': 531.3766351250075, 'mole_fractions': [0.9, 0.03160848510613454, 0.1, 0.9683915148938654]}, {'pressure': 7.004350433920412, 'temperature': 536.3785931087117, 'mole_fractions': [0.9, 0.03605054602415764, 0.1, 0.9639494539758424]},
            {'pressure': 7.5104134842481285, 'temperature': 538.9904183438309, 'mole_fractions': [0.9, 0.038602966503314524, 0.1, 0.9613970334966857]}, {'pressure': 8.053039498311465, 'temperature': 541.6140588672158, 'mole_fractions': [0.9, 0.041343430064165944, 0.1, 0.9586565699358343]},
            {'pressure': 8.634870143618581, 'temperature': 544.2481477394105, 'mole_fractions': [0.9, 0.04428624193970775, 0.1, 0.9557137580602924]}, {'pressure': 9.258737947676686, 'temperature': 546.891169491357, 'mole_fractions': [0.9, 0.04744679197921882, 0.1, 0.952553208020781]},
            {'pressure': 9.927680087592398, 'temperature': 549.541447189215, 'mole_fractions': [0.9, 0.05084162763126554, 0.1, 0.9491583723687347]}, {'pressure': 10.644953175968238, 'temperature': 552.1971283515306, 'mole_fractions': [0.9, 0.05448852914791397, 0.1, 0.9455114708520862]},
            {'pressure': 11.414049115077475, 'temperature': 554.8561696043307, 'mole_fractions': [0.9, 0.0584065863595341, 0.1, 0.9415934136404662]}, {'pressure': 12.23871209650022, 'temperature': 557.5163199458482, 'mole_fractions': [0.9, 0.0626162761528895, 0.1, 0.9373837238471107]},
            {'pressure': 13.12295682898015, 'temperature': 560.1751024765687, 'mole_fractions': [0.9, 0.06713953952408681, 0.1, 0.9328604604759132]}, {'pressure': 14.07010505422318, 'temperature': 562.8271387641806, 'mole_fractions': [0.9, 0.07199481229668654, 0.1, 0.9280051877033139]},
            {'pressure': 15.083614669507542, 'temperature': 565.4670905972858, 'mole_fractions': [0.9, 0.07720119968614912, 0.1, 0.9227988003138512]}, {'pressure': 16.168180744317116, 'temperature': 568.092205402211, 'mole_fractions': [0.9, 0.08278409294852725, 0.1, 0.9172159070514727]},
            {'pressure': 17.733629117845368, 'temperature': 571.5640975643905, 'mole_fractions': [0.9, 0.09086096714833876, 0.1, 0.9091390328516612]}, {'pressure': 20.054543368527575, 'temperature': 576.1304345760296, 'mole_fractions': [0.9, 0.10286902795182505, 0.1, 0.897130972048175]},
            {'pressure': 21.57066606885635, 'temperature': 578.7977227503367, 'mole_fractions': [0.9, 0.11072982023753035, 0.1, 0.8892701797624696]}, {'pressure': 23.770855742757046, 'temperature': 582.2955969460419, 'mole_fractions': [0.9, 0.1221531132193584, 0.1, 0.8778468867806417]},
            {'pressure': 27.058391568104838, 'temperature': 586.8338463436538, 'mole_fractions': [0.9, 0.13923800756860824, 0.1, 0.8607619924313918]}, {'pressure': 30.982522249763612, 'temperature': 591.3793909051027, 'mole_fractions': [0.9, 0.15961969038889876, 0.1, 0.8403803096111013]},
            {'pressure': 35.31487612797394, 'temperature': 595.5240657980244, 'mole_fractions': [0.9, 0.18205490832003932, 0.1, 0.8179450916799607]}, {'pressure': 40.08011459105592, 'temperature': 599.2394736443687, 'mole_fractions': [0.9, 0.20658744367704118, 0.1, 0.7934125563229588]},
            {'pressure': 47.479716096105136, 'temperature': 603.6385499634694, 'mole_fractions': [0.9, 0.24423974859551423, 0.1, 0.755760251404486]}, {'pressure': 57.78788305035784, 'temperature': 607.6627153786644, 'mole_fractions': [0.9, 0.29547881020981076, 0.1, 0.7045211897901893]},
            {'pressure': 70.68218733487244, 'temperature': 610.1624159771142, 'mole_fractions': [0.9, 0.35707472564818704, 0.1, 0.6429252743518131]}, {'pressure': 86.45361869526151, 'temperature': 610.3948088782214, 'mole_fractions': [0.9, 0.42805230475858097, 0.1, 0.571947695241419]},
            {'pressure': 105.74415517299806, 'temperature': 607.4512468455885, 'mole_fractions': [0.9, 0.5079307417992383, 0.1, 0.49206925820076175]}, {'pressure': 129.339020413543, 'temperature': 599.9011446077882, 'mole_fractions': [0.9, 0.5955214697109658, 0.1, 0.40447853028903435]},
            {'pressure': 136.10356488595968, 'temperature': 596.994792768476, 'mole_fractions': [0.9, 0.6187359352001502, 0.1, 0.38126406479984976]}, {'pressure': 144.8540272185635, 'temperature': 592.7422535708602, 'mole_fractions': [0.9, 0.6476293401043584, 0.1, 0.3523706598956417]},
            {'pressure': 150.50839628421622, 'temperature': 589.6891010468271, 'mole_fractions': [0.9, 0.6656646352211113, 0.1, 0.33433536477888876]}, {'pressure': 157.81162038561249, 'temperature': 585.3716853346003, 'mole_fractions': [0.9, 0.6882840055546668, 0.1, 0.3117159944453332]},
            {'pressure': 162.52314140366204, 'temperature': 582.3489041066117, 'mole_fractions': [0.9, 0.7025068751952628, 0.1, 0.2974931248047373]}, {'pressure': 168.5989904512539, 'temperature': 578.1534220010057, 'mole_fractions': [0.9, 0.7204663713538947, 0.1, 0.2795336286461055]},
            {'pressure': 172.51288669528185, 'temperature': 575.2590387470864, 'mole_fractions': [0.9, 0.7318329087665353, 0.1, 0.2681670912334648]}, {'pressure': 177.55363431285468, 'temperature': 571.2880633766006, 'mole_fractions': [0.9, 0.7462725974829075, 0.1, 0.2537274025170924]},
            {'pressure': 180.79713728230843, 'temperature': 568.5746390365938, 'mole_fractions': [0.9, 0.7554644705384199, 0.1, 0.2445355294615802]}, {'pressure': 184.9708492278261, 'temperature': 564.880757785716, 'mole_fractions': [0.9, 0.7672044033510299, 0.1, 0.23279559664897007]},
            {'pressure': 190.25746243179395, 'temperature': 559.832280225296, 'mole_fractions': [0.9, 0.7819856140179173, 0.1, 0.2180143859820826]}, {'pressure': 193.59879096431627, 'temperature': 556.3960615621182, 'mole_fractions': [0.9, 0.7913145936966299, 0.1, 0.20868540630337018]},
            {'pressure': 197.81819152733883, 'temperature': 551.7366778273317, 'mole_fractions': [0.9, 0.8031352398195709, 0.1, 0.19686476018042914]}, {'pressure': 200.477587029481, 'temperature': 548.5863781135826, 'mole_fractions': [0.9, 0.8106413055267412, 0.1, 0.18935869447325893]},
            {'pressure': 203.82772822720145, 'temperature': 544.3381330813057, 'mole_fractions': [0.9, 0.8202061044981348, 0.1, 0.17979389550186511]}, {'pressure': 205.93467406909477, 'temperature': 541.4792852640306, 'mole_fractions': [0.9, 0.826312697699432, 0.1, 0.17368730230056806]},
            {'pressure': 208.58398327181322, 'temperature': 537.6391089041348, 'mole_fractions': [0.9, 0.834133413663969, 0.1, 0.1658665863360311]}, {'pressure': 210.2474691186204, 'temperature': 535.0635302273878, 'mole_fractions': [0.9, 0.8391505928556668, 0.1, 0.16084940714433316]},
            {'pressure': 212.3363495572753, 'temperature': 531.6135223103543, 'mole_fractions': [0.9, 0.8456048898317248, 0.1, 0.15439511016827528]}, {'pressure': 214.8901790698365, 'temperature': 526.9915306379916, 'mole_fractions': [0.9, 0.8538095584008777, 0.1, 0.1461904415991224]},
            {'pressure': 216.44501709505957, 'temperature': 523.9003670641478, 'mole_fractions': [0.9, 0.8590356578837798, 0.1, 0.14096434211622014]}, {'pressure': 218.33380099026294, 'temperature': 519.7719059967658, 'mole_fractions': [0.9, 0.8657143342525401, 0.1, 0.13428566574746]},
            {'pressure': 219.47598268844666, 'temperature': 517.0180023473305, 'mole_fractions': [0.9, 0.8699900528343006, 0.1, 0.1300099471656993]}, {'pressure': 220.8536963228419, 'temperature': 513.347741089584, 'mole_fractions': [0.9, 0.8754799894748081, 0.1, 0.12452001052519193]},
            {'pressure': 222.43412297564478, 'temperature': 508.46359516820974, 'mole_fractions': [0.9, 0.8824409941891852, 0.1, 0.11755900581081483]}, {'pressure': 223.72469727140603, 'temperature': 503.5991209826523, 'mole_fractions': [0.9, 0.8890128599496768, 0.1, 0.1109871400503232]},
            {'pressure': 224.45382842530304, 'temperature': 500.23313893695854, 'mole_fractions': [0.9, 0.8933647279579296, 0.1, 0.10663527204207038]}, {'pressure': 224.82957749204152, 'temperature': 498.2010471587874, 'mole_fractions': [0.9, 0.8959189225807612, 0.1, 0.10408107741923883]},
            {'pressure': 225.72269494505386, 'temperature': 491.5475312248371, 'mole_fractions': [0.9, 0.9039210560847678, 0.1, 0.09607894391523232]}, {'pressure': 225.93441253883682, 'temperature': 488.9118482066479, 'mole_fractions': [0.9, 0.9069469104188794, 0.1, 0.09305308958112057]},
            {'pressure': 226.0653092373394, 'temperature': 486.29282151977895, 'mole_fractions': [0.9, 0.9098774702578796, 0.1, 0.09012252974212048]}, {'pressure': 226.11707998726936, 'temperature': 482.82881099792235, 'mole_fractions': [0.9, 0.9136418209958436, 0.1, 0.08635817900415642]},
            {'pressure': 225.9748314579137, 'temperature': 478.2643128896316, 'mole_fractions': [0.9, 0.9184175120540755, 0.1, 0.0815824879459247]}, {'pressure': 225.75076115082692, 'temperature': 475.25841429393637, 'mole_fractions': [0.9, 0.9214536650728117, 0.1, 0.07854633492718834]},
            {'pressure': 225.29799418361205, 'temperature': 471.29998152547427, 'mole_fractions': [0.9, 0.9253268340574672, 0.1, 0.07467316594253302]}, {'pressure': 224.90216833259524, 'temperature': 468.69400571590364, 'mole_fractions': [0.9, 0.9278022509286525, 0.1, 0.07219774907134749]},
            {'pressure': 224.26260927093534, 'temperature': 465.2623287455804, 'mole_fractions': [0.9, 0.9309756304882018, 0.1, 0.06902436951179836]}, {'pressure': 223.22124186482415, 'temperature': 460.7663941947387, 'mole_fractions': [0.9, 0.9349908976539416, 0.1, 0.0650091023460586]},
            {'pressure': 222.41407584785605, 'temperature': 457.8215937019191, 'mole_fractions': [0.9, 0.9375371230633907, 0.1, 0.062462876936609395]}, {'pressure': 221.20644725277666, 'temperature': 453.9627069434391, 'mole_fractions': [0.9, 0.940777636560588, 0.1, 0.059222363439412035]},
            {'pressure': 220.32282102639596, 'temperature': 451.4339833480792, 'mole_fractions': [0.9, 0.9428440339230565, 0.1, 0.05715596607694343]}, {'pressure': 219.05334723205738, 'temperature': 448.11805980702087, 'mole_fractions': [0.9, 0.9454875105083457, 0.1, 0.05451248949165445]},
            {'pressure': 217.21057393966694, 'temperature': 443.797578895822, 'mole_fractions': [0.9, 0.9488229465483582, 0.1, 0.05117705345164179]}, {'pressure': 215.89471221928054, 'temperature': 440.98211615282884, 'mole_fractions': [0.9, 0.9509324035049135, 0.1, 0.049067596495086664]},
            {'pressure': 214.04183152029154, 'temperature': 437.3097055209119, 'mole_fractions': [0.9, 0.9536103820859334, 0.1, 0.04638961791406659]}, {'pressure': 211.41545742262866, 'temperature': 432.5590081967015, 'mole_fractions': [0.9, 0.9569550564597935, 0.1, 0.04304494354020657]},
            {'pressure': 209.57779912082407, 'temperature': 429.4846744651189, 'mole_fractions': [0.9, 0.9590497673114605, 0.1, 0.04095023268853958]}, {'pressure': 207.03501291799245, 'temperature': 425.50077266281994, 'mole_fractions': [0.9, 0.9616850130184242, 0.1, 0.03831498698157593]},
            {'pressure': 205.288851371022, 'temperature': 422.91759995927885, 'mole_fractions': [0.9, 0.9633469413912721, 0.1, 0.036653058608727945]}, {'pressure': 202.90696050183828, 'temperature': 419.56314263617065, 'mole_fractions': [0.9, 0.9654512552193921, 0.1, 0.03454874478060786]},
            {'pressure': 199.65311381663446, 'temperature': 415.2481463910756, 'mole_fractions': [0.9, 0.9680702874163188, 0.1, 0.031929712583681255]}, {'pressure': 197.44518065084674, 'temperature': 412.4699634460068, 'mole_fractions': [0.9, 0.9697050551276083, 0.1, 0.030294944872391785]},
            {'pressure': 194.46623931906086, 'temperature': 408.8858359654431, 'mole_fractions': [0.9, 0.9717552950114567, 0.1, 0.028244704988543146]}, {'pressure': 190.45471739837748, 'temperature': 404.3152819897392, 'mole_fractions': [0.9, 0.9742747535021606, 0.1, 0.0257252464978395]},
            {'pressure': 187.9426170508094, 'temperature': 401.5824170709623, 'mole_fractions': [0.9, 0.9757306866084718, 0.1, 0.02426931339152819]}, {'pressure': 184.59374089876226, 'temperature': 398.07404352350346, 'mole_fractions': [0.9, 0.9775445679019101, 0.1, 0.022455432098089825]},
            {'pressure': 180.15124638228667, 'temperature': 393.62829505150944, 'mole_fractions': [0.9, 0.9797539649123556, 0.1, 0.02024603508764439]}, {'pressure': 177.4643999336587, 'temperature': 391.0410842423552, 'mole_fractions': [0.9, 0.9809937368298178, 0.1, 0.0190062631701822]},
            {'pressure': 173.9208077499966, 'temperature': 387.73269921501003, 'mole_fractions': [0.9, 0.9825294835775386, 0.1, 0.01747051642246153]}, {'pressure': 169.2820208795578, 'temperature': 383.56096266817406, 'mole_fractions': [0.9, 0.9843858715220213, 0.1, 0.015614128477978923]},
            {'pressure': 163.81330786998205, 'temperature': 378.84514981924656, 'mole_fractions': [0.9, 0.9863750235980266, 0.1, 0.013624976401973321]}, {'pressure': 158.35676681728512, 'temperature': 374.32808574447375, 'mole_fractions': [0.9, 0.9881690195727469, 0.1, 0.011830980427253012]},
            {'pressure': 152.95450469981742, 'temperature': 370.0156229577152, 'mole_fractions': [0.9, 0.9897771641421396, 0.1, 0.010222835857860466]}, {'pressure': 147.6435898952142, 'temperature': 365.91022770825316, 'mole_fractions': [0.9, 0.9912100479664634, 0.1, 0.008789952033536694]},
            {'pressure': 142.4555492399344, 'temperature': 362.01135638016177, 'mole_fractions': [0.9, 0.9924791841466785, 0.1, 0.007520815853321566]}, {'pressure': 137.416150080239, 'temperature': 358.31585443944397, 'mole_fractions': [0.9, 0.993596673110973, 0.1, 0.006403326889027113]},
            {'pressure': 132.5454284136517, 'temperature': 354.81835729454053, 'mole_fractions': [0.9, 0.9945749011334869, 0.1, 0.005425098866513176]}, {'pressure': 127.8579176400145, 'temperature': 351.5116775892977, 'mole_fractions': [0.9, 0.9954262759031482, 0.1, 0.004573724096851832]},
            {'pressure': 123.3630310075106, 'temperature': 348.3871683338939, 'mole_fractions': [0.9, 0.9961630008899924, 0.1, 0.003836999110007801]}, {'pressure': 119.06555307863805, 'temperature': 345.4350554631834, 'mole_fractions': [0.9, 0.9967968887679837, 0.1, 0.003203111232016178]},
            {'pressure': 114.96620006808907, 'temperature': 342.64473669667495, 'mole_fractions': [0.9, 0.9973392128804714, 0.1, 0.0026607871195286413]}, {'pressure': 111.06221459726486, 'temperature': 340.00504594867374, 'mole_fractions': [0.9, 0.9978005946955636, 0.1, 0.0021994053044364402]},
            {'pressure': 107.34796646359128, 'temperature': 337.5044841085172, 'mole_fractions': [0.9, 0.9981909243991476, 0.1, 0.001809075600852369]}, {'pressure': 103.81553692117569, 'temperature': 335.13141794445875, 'mole_fractions': [0.9, 0.9985193112066199, 0.1, 0.001480688793380107]},
            {'pressure': 100.45526942039983, 'temperature': 332.8742493571255, 'mole_fractions': [0.9, 0.9987940596249657, 0.1, 0.001205940375034318]}, {'pressure': 97.2562746179068, 'temperature': 330.7215573786426, 'mole_fractions': [0.9, 0.9990226677418205, 0.1, 0.0009773322581796686]},
            {'pressure': 94.20688169626781, 'temperature': 328.66221530781536, 'mole_fractions': [0.9, 0.9992118436300331, 0.1, 0.0007881563699667903]}, {'pressure': 91.29503162462903, 'temperature': 326.6854852810068, 'mole_fractions': [0.9, 0.9993675361049214, 0.1, 0.0006324638950788114]},
            {'pressure': 88.50861097130779, 'temperature': 324.78109246056044, 'mole_fractions': [0.9, 0.9994949763260585, 0.1, 0.0005050236739415777]}, {'pressure': 85.83572728044807, 'temperature': 322.93928090949044, 'mole_fractions': [0.9, 0.9995987270662882, 0.1, 0.0004012729337119059]},
            {'pressure': 83.2649288870058, 'temperature': 321.15085312547376, 'mole_fractions': [0.9, 0.9996827368500716, 0.1, 0.0003172631499285116]}, {'pressure': 80.78537341184283, 'temperature': 319.4071951295028, 'mole_fractions': [0.9, 0.9997503965668684, 0.1, 0.00024960343313148526]},
            {'pressure': 78.38695010080288, 'temperature': 317.70028893874616, 'mole_fractions': [0.9, 0.999804596572249, 0.1, 0.0001954034277511493]}, {'pressure': 76.06036170276094, 'temperature': 316.0227141907118, 'mole_fractions': [0.9, 0.9998477826831261, 0.1, 0.00015221731687382385]},
            {'pressure': 73.79717178000965, 'temperature': 314.3676406183606, 'mole_fractions': [0.9, 0.99988200984114, 0.1, 0.0001179901588601233]}, {'pressure': 71.5898232706397, 'temperature': 312.7288129969959, 'mole_fractions': [0.9, 0.9999089925508112, 0.1, 9.100744918878371e-05]},
            {'pressure': 69.43163383715843, 'temperature': 311.10053008982214, 'mole_fractions': [0.9, 0.9999301514911269, 0.1, 6.984850887325554e-05]}, {'pressure': 67.31677309628984, 'temperature': 309.47761900898547, 'mole_fractions': [0.9, 0.9999466559480595, 0.1, 5.334405194044854e-05]},
            {'pressure': 65.2402262846843, 'temperature': 307.8554062840186, 'mole_fractions': [0.9, 0.9999594619210762, 0.1, 4.0538078923686146e-05]}, {'pressure': 63.197748320458146, 'temperature': 306.2296867930393, 'mole_fractions': [0.9, 0.9999693459206493, 0.1, 3.06540793506663e-05]},
            {'pressure': 61.18581160976825, 'temperature': 304.5966915680354, 'mole_fractions': [0.9, 0.9999769345993477, 0.1, 2.306540065244492e-05]}, {'pressure': 59.201550351587805, 'temperature': 302.95305533885505, 'mole_fractions': [0.9, 0.9999827304502956, 0.1, 1.726954970456278e-05]},
            {'pressure': 57.24270353503327, 'temperature': 301.2957845357326, 'mole_fractions': [0.9, 0.9999871338682591, 0.1, 1.2866131740928356e-05]}, {'pressure': 55.30755831730724, 'temperature': 299.6222263314917, 'mole_fractions': [0.9, 0.9999904619051202, 0.1, 9.53809487983404e-06]},
            {'pressure': 53.39489502541686, 'temperature': 297.93003917539806, 'mole_fractions': [0.9, 0.9999929640677174, 0.1, 7.0359322825514505e-06]}, {'pressure': 51.50393464491381, 'temperature': 296.21716515343644, 'mole_fractions': [0.9, 0.9999948355063719, 0.1, 5.164493628035122e-06]},
            {'pressure': 49.6342893435074, 'temperature': 294.48180440617773, 'mole_fractions': [0.9, 0.9999962279308461, 0.1, 3.7720691538990824e-06]}, {'pressure': 47.78591632308305, 'temperature': 292.7223917460595, 'mole_fractions': [0.9, 0.9999972585704927, 0.1, 2.741429507328048e-06]},
            {'pressure': 45.959075094999804, 'temperature': 290.9375755407823, 'mole_fractions': [0.9, 0.999998017469871, 0.1, 1.9825301292253057e-06]}, {'pressure': 44.154288124053394, 'temperature': 289.12619886799956, 'mole_fractions': [0.9, 0.9999985733825459, 0.1, 1.426617454139879e-06]},
            {'pressure': 42.3723046792024, 'temperature': 287.2872828974432, 'mole_fractions': [0.9, 0.9999989784960532, 0.1, 1.021503946599771e-06]}, {'pressure': 40.614067657198945, 'temperature': 285.42001241874453, 'mole_fractions': [0.9, 0.99999927219153, 0.1, 7.278084699766911e-07]},
            {'pressure': 38.88068310217222, 'temperature': 283.523723404919, 'mole_fractions': [0.9, 0.9999994840133423, 0.1, 5.15986657623538e-07]}, {'pressure': 37.17339212413597, 'temperature': 281.5978924812663, 'mole_fractions': [0.9, 0.9999996359978712, 0.1, 3.640021286778605e-07]},
            {'pressure': 35.49354491725107, 'temperature': 279.64212815574916, 'mole_fractions': [0.9, 0.9999997444868582, 0.1, 2.555131417655257e-07]}, {'pressure': 33.84257659014894, 'temperature': 277.6561636583614, 'mole_fractions': [0.9, 0.9999998215295924, 0.1, 1.7847040759128985e-07]},
            {'pressure': 32.22198454210955, 'temperature': 275.63985123232374, 'mole_fractions': [0.9, 0.9999998759597488, 0.1, 1.2404025126226472e-07]}, {'pressure': 30.633307147545214, 'temperature': 273.5931577180604, 'mole_fractions': [0.9, 0.9999999142167874, 0.1, 8.578321258478115e-08]},
            {'pressure': 29.07810354470297, 'temperature': 271.5161612709092, 'mole_fractions': [0.9, 0.9999999409683357, 0.1, 5.903166430945006e-08]}, {'pressure': 27.557934361021935, 'temperature': 269.4090490546766, 'mole_fractions': [0.9, 0.9999999595786689, 0.1, 4.042133102409915e-08]},
            {'pressure': 26.07434324571413, 'temperature': 267.27211575490855, 'mole_fractions': [0.9, 0.9999999724590531, 0.1, 2.7540946941191936e-08]}, {'pressure': 24.628839118811275, 'temperature': 265.1057627577206, 'mole_fractions': [0.9, 0.9999999813280525, 0.1, 1.8671947684424075e-08]},
            {'pressure': 23.222879084240603, 'temperature': 262.91049784195917, 'mole_fractions': [0.9, 0.9999999874037087, 0.1, 1.2596291254744614e-08]}, {'pressure': 21.857851991737444, 'temperature': 260.68693523425685, 'mole_fractions': [0.9, 0.9999999915445269, 0.1, 8.455473111881264e-09]},
            {'pressure': 20.535062667920467, 'temperature': 258.4357958781957, 'mole_fractions': [0.9, 0.9999999943522546, 0.1, 5.6477454044296815e-09]}, {'pressure': 19.25571687003392, 'temperature': 256.15790777043634, 'mole_fractions': [0.9, 0.9999999962463471, 0.1, 3.753653016148666e-09]},
            {'pressure': 18.02090704608964, 'temperature': 253.8542062185244, 'mole_fractions': [0.9, 0.9999999975175826, 0.1, 2.4824173261975422e-09]}, {'pressure': 16.831599011779783, 'temperature': 251.52573387744758, 'mole_fractions': [0.9, 0.9999999983664329, 0.1, 1.6335670432378608e-09]},
            {'pressure': 15.688619676902592, 'temperature': 249.17364042528817, 'mole_fractions': [0.9, 0.9999999989303531, 0.1, 1.0696470062622035e-09]}, {'pressure': 14.592645971455985, 'temperature': 246.79918174287303, 'mole_fractions': [0.9, 0.9999999993030764, 0.1, 6.96923639339474e-10]},
            {'pressure': 13.54419513328613, 'temperature': 244.40371846868777, 'mole_fractions': [0.9, 0.9999999995481743, 0.1, 4.518257365772956e-10]}, {'pressure': 12.54361652455094, 'temperature': 241.98871380889818, 'mole_fractions': [0.9, 0.9999999997085276, 0.1, 2.9147254041303984e-10]},
            {'pressure': 11.591085142647042, 'temperature': 239.555730493582, 'mole_fractions': [0.9, 0.9999999998129038, 0.1, 1.8709625724919616e-10]}, {'pressure': 10.686596982153093, 'temperature': 237.1064267845684, 'mole_fractions': [0.9, 0.9999999998804985, 0.1, 1.1950145420087781e-10]},
            {'pressure': 9.829966387463731, 'temperature': 234.64255145791435, 'mole_fractions': [0.9, 0.9999999999240512, 0.1, 7.594895184369684e-11]}, {'pressure': 9.020825511051315, 'temperature': 232.16593770510477, 'mole_fractions': [0.9, 0.9999999999519703, 0.1, 4.802979722423504e-11]},
            {'pressure': 8.258625959967263, 'temperature': 229.67849592154357, 'mole_fractions': [0.9, 0.9999999999697768, 0.1, 3.0223166831770016e-11]}, {'pressure': 7.542642673892549, 'temperature': 227.18220537851414, 'mole_fractions': [0.9, 0.9999999999810762, 0.1, 1.8923841688328028e-11]},
            {'pressure': 6.871980032790137, 'temperature': 224.67910480504094, 'mole_fractions': [0.9, 0.9999999999882099, 0.1, 1.1790131377583414e-11]}, {'pressure': 6.24558014242936, 'temperature': 222.17128193821995, 'mole_fractions': [0.9, 0.9999999999926907, 0.1, 7.309166649132939e-12]},
            {'pressure': 5.662233193554981, 'temperature': 219.66086213362425, 'mole_fractions': [0.9, 0.9999999999954913, 0.1, 4.508757187616446e-12]}, {'pressure': 5.120589737395821, 'temperature': 217.14999616010272, 'mole_fractions': [0.9, 0.9999999999972327, 0.1, 2.767486309658223e-12]},
            {'pressure': 4.619174668917571, 'temperature': 214.6408473343279, 'mole_fractions': [0.9, 0.9999999999983098, 0.1, 1.690260537984418e-12]}, {'pressure': 4.1564026621828525, 'temperature': 212.1355781783378, 'mole_fractions': [0.9, 0.9999999999989727, 0.1, 1.0272147594695886e-12]},
            {'pressure': 3.730594761796904, 'temperature': 209.63633680662818, 'mole_fractions': [0.9, 0.9999999999993789, 0.1, 6.211667052537558e-13]}, {'pressure': 3.339995802868991, 'temperature': 207.14524326670224, 'mole_fractions': [0.9, 0.9999999999996264, 0.1, 3.737612743868456e-13]},
            {'pressure': 2.982792311003345, 'temperature': 204.66437606731785, 'mole_fractions': [0.9, 0.9999999999997761, 0.1, 2.237791113292395e-13]}, {'pressure': 2.657130524794037, 'temperature': 202.19575913110359, 'mole_fractions': [0.9, 0.9999999999998666, 0.1, 1.3331646718817187e-13]},
            {'pressure': 2.36113418674297, 'temperature': 199.74134940239577, 'mole_fractions': [0.9, 0.999999999999921, 0.1, 7.902908566309983e-14]}, {'pressure': 2.0929217643357814, 'temperature': 197.30302532709436, 'mole_fractions': [0.9, 0.9999999999999535, 0.1, 4.661534784417071e-14]},
            {'pressure': 1.8506227903729588, 'temperature': 194.88257639955148, 'mole_fractions': [0.9, 0.9999999999999727, 0.1, 2.735959197912005e-14]}, {'pressure': 1.6323930490787326, 'temperature': 192.4816939429678, 'mole_fractions': [0.9, 0.9999999999999841, 0.1, 1.59782394159578e-14]},
            {'pressure': 1.4364283799698359, 'temperature': 190.1019632557797, 'mole_fractions': [0.9, 0.9999999999999907, 0.1, 9.285103744189322e-15]}, {'pressure': 1.2609769225448897, 'temperature': 187.74485721874223, 'mole_fractions': [0.9, 0.9999999999999947, 0.1, 5.368871805684536e-15]},
            {'pressure': 1.104349678929136, 'temperature': 185.41173141759538, 'mole_fractions': [0.9, 0.9999999999999969, 0.1, 3.088998207411923e-15]}, {'pressure': 0.9649293260526793, 'temperature': 183.10382079623824, 'mole_fractions': [0.9, 0.9999999999999982, 0.1, 1.7684408522121055e-15]},
            {'pressure': 0.841177261287654, 'temperature': 180.82223781692787, 'mole_fractions': [0.9, 0.9999999999999989, 0.1, 1.0073991867829547e-15]}, {'pressure': 0.731638913589899, 'temperature': 178.56797206877053, 'mole_fractions': [0.9, 0.9999999999999994, 0.1, 5.710192416785545e-16]},
            {'pressure': 0.6349473943838938, 'temperature': 176.3418912349383, 'mole_fractions': [0.9, 0.9999999999999997, 0.1, 3.2206082177856057e-16]}, {'pressure': 0.5498255975043113, 'temperature': 174.1447433035498, 'mole_fractions': [0.9, 0.9999999999999999, 0.1, 1.807436197619397e-16]}
        ]

        mole_fractions = numpy.array([0.9, 0.1])
        critical_temperatures = numpy.array([304.21, 723.00])
        critical_pressures = numpy.array([73.83, 14.00])
        acentric_factors = numpy.array([0.2236, 0.7174])

        pe = PhaseEnvelope(logging_level="DEBUG")
        temperature = 80.0  # K
        pressure = 0.5  # bar

        res = pe.calculate(temperature, pressure, mole_fractions, critical_temperatures, critical_pressures, acentric_factors)
        self.rounder(expected_res)
        self.rounder(res)
        self.assertListEqual(expected_res, res)

    def test_phase_envelope_complex(self):
        self.maxDiff = None

        # molfrac critical_temperatures critical_pressures acentric_factors
        input_data_string = """
        0.00047, 126.094, 34.0, 0.045
        0.0242, 304.039, 73.834, 0.231
        0.6822, 190.372, 46.055, 0.0115
        0.118, 305.261, 48.814, 0.0908
        0.0546, 369.65, 42.503, 0.1454
        0.0083, 407.983, 36.49, 0.1756
        0.0174, 424.983, 37.979, 0.1928
        0.0072, 460.261, 33.821, 0.2273
        0.0074, 469.483, 33.697, 0.251
        0.0107, 507.261, 30.131, 0.2957
        0.0653, 634.331, 25.707, 0.4699
        """
        input_data = input_data_string.strip("\n").strip().split("\n")
        z = []
        critical_temperatures = []
        critical_pressures = []
        acentric_factors = []
        for row in input_data:
            row_data = row.split(", ")
            z.append(float(row_data[0]))
            critical_temperatures.append(float(row_data[1]))
            critical_pressures.append(float(row_data[2]))
            acentric_factors.append(float(row_data[3]))

        pe = PhaseEnvelope(logging_level="DEBUG")
        temperature = 80.0
        pressure = 0.5  

        res = pe.calculate(temperature, pressure, z, critical_temperatures, critical_pressures, acentric_factors)
        self.rounder(res)

        self.assertEqual(2.033, res[10]["pressure"])
        self.assertEqual(383.155, res[10]["temperature"])

        self.assertEqual(246.228, res[84]["pressure"])
        self.assertEqual(320.507, res[84]["temperature"])

        self.assertEqual(190.137, res[101]["pressure"])
        self.assertEqual(267.834, res[101]["temperature"])

    @staticmethod
    def rounder(results_list: list):
        for key in ("pressure", "temperature", "mole_fractions"):
            for item_dict in results_list:
                item = item_dict[key]
                if key == "mole_fractions":
                    item_dict[key] = [round(record, 4) for record in item]
                    continue

                item_dict[key] = round(item, 3)

    class TestSuccessiveSubstitution(unittest.TestCase):
        def test_get_initial_temperature_guess(self):
            guess_temperature = 80.0
            guess_pressure = 0.5
            critical_temperatures = [304.21, 723.0]
            ac = [3.96211105e+06, 1.18021492e+08]
            comp = 2
            vapor_mole_fractions = numpy.array([0.9, 0.1])
            phase = numpy.array([0, 1])
            b = numpy.array([26.65350799, 334.05964905])
            amix = numpy.array([0.0, 0.0])
            bmix = numpy.array([0.0, 0.0])
            acentric_factors = numpy.array([0.2236, 0.7174])
            kij = numpy.array([numpy.array([0.0, 0.0]), numpy.array([0.0, 0.0])])
            lij = numpy.array([numpy.array([0.0, 0.0]), numpy.array([0.0, 0.0])])
            guess_temperature = SuccessiveSubstitution.get_initial_temperature_guess(guess_temperature, guess_pressure, comp, vapor_mole_fractions, phase, b, amix, bmix, acentric_factors, critical_temperatures, ac, kij, lij)

            self.assertEqual(230.0, guess_temperature)
