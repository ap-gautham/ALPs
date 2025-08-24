from iminuit import Minuit as imfit
import numpy as np
from numpy import *
import pyymw16
import time
import gc
import multiprocessing as mp
import distutils.dir_util
import ctypes
from ctypes import *
from numpy.ctypeslib import ndpointer
import matplotlib
matplotlib.use('Agg')

###################################################################################################################

var_name = ['b4', 'b5', 'b6', 'rhoXc', 'b0', 'b1', 'b2', 'b3', 'hdisk', 'Bn', 'Bs', 'rhos', 'wdisk', 'whalo', 'rhon', 'z0', 'rhoX', 'bring', 'BX0', 'ThetaX0', 'b7', 'gamma']

var_mean = {'bring':0.1,'hdisk':0.4,'wdisk':0.27,
       'b0':0.1,'b1':2.,'b2':-0.9,'b3':2.,'b4':-3.,'b5':-3.5,'b6':0.,'b7':2.7,
       'Bn':1.,'Bs':-0.8,'rhon':9.22,'rhos':16.7,'whalo':0.2,'z0':5.3,
       'BX0':3.,'ThetaX0':49. * np.pi/180.,'rhoXc':4.8,'rhoX':2.9,'gamma':2.92}

var_err = {'bring':0.1,'hdisk':0.03,'wdisk':0.08,
       'b0':1.8,'b1':0.6,'b2':0.8,'b3':0.3,'b4':0.1,'b5':0.5,'b6':1.8,'b7':1.8,
       'Bn':0.1,'Bs':0.1,'rhon':0.08,'rhos':0.,'whalo':0.12,'z0':1.6,
       'BX0':0.3,'ThetaX0':np.pi/180.,'rhoXc':0.2,'rhoX':0.1,'gamma':0.14}

psr = ['J1718-3825','J1702-4128','J1648-4611','J1420-6048','J2240+5832','J2021+3651']

param_temp = [[0.1,1.08,0.3],[0.46,0.06,0.078],[0.022,0.49,0.4],[0.007,1.41,0.76],[0.048,0.85,0.18],[0.7,1.03,0.368]]
param_init = []
for i in range(6):
    param_init.append(param_temp[i])

param_init = np.append(param_init,[var_mean['b0'],var_mean['b1'],var_mean['b2'],var_mean['b3'],var_mean['b6']])
    
s1 =  np.array([[3.49],[3.97],[4.47],[5.63],[7.27],[10.51]])
L1 = (np.array([[348.951],[344.744],[339.438],[313.541],[106.566],[75.222]]))
B1 = (np.array([[-0.432],[0.123],[-0.794],[0.227],[-0.111],[0.111]]))

E0_psr = np.array([1.2,0.1,2.9,5.6,1.2,0.8])

path_n = [0,1,2,3,4,5]

g2 = 0.54

###################################################################################################################

gARR_min = 1
gARR_max = 50

mARR_min = 1
mARR_max = 10

g_step = 1.75
m_step = 0.35

gARR = np.append([0.01,0.1,0.25,0.5,0.75],np.arange(gARR_min,gARR_max,g_step)) 
mARR = np.append([0.01,0.1,0.25,0.5,0.75],np.arange(mARR_min,mARR_max,m_step))
gsize = np.size(gARR)
msize = np.size(mARR)
merr = mARR[1] - mARR[0]
gerr = gARR[1] - gARR[0]

###################################################################################################################

Path = 'Result/'
distutils.dir_util.mkpath(Path)

path_n = Path+'ALP_bfield'

fun = CDLL('cpp/comb_b_mod.so')
fun.sq.restype = dtype=ctypes.c_double

for PSR_sim in range(399,401):
    
    step=[]
    size_spec = []
    spec_0=[]
    spec_1=[]
    spec_2=[]
    ne_C=[]
    path_C=[]
    path=[]

    bins = 100
    arr_100 = (ctypes.c_double * 100)

    for psr_n in range(6):

        spec = np.loadtxt(psr[psr_n]+'/actual_bin_eflux_%d.txt'%PSR_sim)
        s,L,B = s1[psr_n],L1[psr_n],B1[psr_n]
        s,l,b = s,L[0],B[0]

        step.append((s/bins)[0]) #-----------------------##-----------------------##-----------------------##-----------------------#

        path.append(np.linspace(s[0],0, bins,endpoint = False)) #-----------------------##-----------------------##-----------------------##-----------------------#

        ne = np.array([pyymw16.calculate_electron_density_lbr(l,b,p*1000).value for p in path[psr_n]])

        size_spec.append(len(spec[:,0])) #-----------------------##-----------------------##-----------------------##-----------------------#
        arr_spec = (ctypes.c_double * size_spec[psr_n])

        spec_0.append(arr_spec())#-----------------------##-----------------------##-----------------------##-----------------------#
        spec_1.append(arr_spec())#-----------------------##-----------------------##-----------------------##-----------------------#
        spec_2.append(arr_spec())#-----------------------##-----------------------##-----------------------##-----------------------#

        for j in range(size_spec[psr_n]):
            spec_0[psr_n][j] = ctypes.c_double(spec[:,0][j])
            spec_1[psr_n][j] = ctypes.c_double(spec[:,1][j])
            spec_2[psr_n][j] = ctypes.c_double(spec[:,2][j])

        ne_C.append(arr_100()) #-----------------------##-----------------------##-----------------------##-----------------------#
        for j in range(100):
            ne_C[psr_n][j] = ctypes.c_double(ne[j])

        path_C.append(arr_100())
        for i in range(np.size(path[psr_n])):
            path_C[psr_n][i] = ctypes.c_double(path[psr_n][i])

    del ne, spec,path

    ###################################################################################################################
    ###################################################################################################################

    def sq1(param):
        b0,b1,b2,b3,b6 = param[3*6+0],param[3*6+1],param[3*6+2],param[3*6+3],param[3*6+4]
        m_a,g_ag = param[-2],param[-1]

        res = 0

        for psr_n in range(6):
            N0 = param[3*psr_n+0]
            G = param[3*psr_n+1]
            Ec = param[3*psr_n+2]
            res += (fun.sq(path_C[psr_n],ctypes.c_double(L1[psr_n][0]*np.pi/180.),ctypes.c_double(B1[psr_n][0]*np.pi/180.),ctypes.c_double(b0),
                                         ctypes.c_double(b1),ctypes.c_double(b2),ctypes.c_double(b3),ctypes.c_double(b6),
                           ne_C[psr_n],spec_0[psr_n],spec_1[psr_n],spec_2[psr_n],ctypes.c_double(N0),ctypes.c_double(G),ctypes.c_double(Ec),
                          ctypes.c_double(m_a),ctypes.c_double(g_ag),ctypes.c_double(step[psr_n]),ctypes.c_int(size_spec[psr_n]),ctypes.c_double(E0_psr[psr_n]),ctypes.c_double(g2)) 
                    + (b0 - var_mean['b0'])**2/var_err['b0']**2 + (b1 - var_mean['b1'])**2/var_err['b1']**2 + (b2 - var_mean['b2'])**2/var_err['b2']**2 + (b3 - var_mean['b3'])**2/var_err['b3']**2
                    + (b6 - var_mean['b6'])**2/var_err['b6']**2)

        return res


    def main(mALP): 
        res = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
        param_init2 = np.append(param_init,[mALP,gARR[0]])
        m1 = imfit.from_array_func(sq1, param_init2,error = 1.5 ,fix=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1),errordef = 1,name=['N0','G0','E0','N1','G1','E1','N2','G2','E2',
                                                                                                                                'N3','G3','E3','N4','G4','E4','N5','G5','E5',
                                                                                                                                         'b0','b1','b2','b3','b6','m_a','g_ag'])
        m1.migrad()
    #(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1)
        for gALP in gARR:
            m1.values['m_a'] = mALP
            m1.values['g_ag'] = gALP 
            m1 = imfit.from_array_func(sq1, m1.values.values(), error = 1.5 ,fix=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1),errordef = 1,
                                       name=['N0','G0','E0','N1','G1','E1','N2','G2','E2','N3','G3','E3','N4','G4','E4','N5','G5','E5','b0','b1','b2','b3','b6','m_a','g_ag'])
            m1.migrad()

            temp = 0
            while m1.migrad_ok() != 1:

                temp += 1
                m1 = imfit.from_array_func(sq1, m1.values.values(), error = 1.5 ,fix=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1),errordef = 1,
                                           name=['N0','G0','E0','N1','G1','E1','N2','G2','E2','N3','G3','E3','N4','G4','E4','N5','G5','E5','b0','b1','b2','b3','b6','m_a','g_ag'])
                m1.migrad()
                if temp == 5: 
                    print m1.migrad_ok(), temp, mALP, gALP
                    break
            temp = m1.values.values()
            temp.append(m1.fval)
            res = np.append(res,[temp],axis=0)
            gc.collect()

        del m1, temp
        gc.collect()

        return res

    ###################################################################################################################
    ###################################################################################################################

    t1 = time.time()

    pool = mp.Pool(processes = 7)
    P = pool.map(main,mARR)
    pool.close()

    t2 = time.time()

    print '\n Time = ', (t2-t1)

    ##################################################################################################################

    x2_min = 100000

    for i in P:
        for j in i:
            if(j[-1]<x2_min and j[0] != 0.):
                x2_min = j[-1]
                g_min = j[-2]
                m_min = j[-3]
                b6_min = j[-4]
                b3_min = j[-5]
                b2_min = j[-6]
                b1_min = j[-7]
                b0_min = j[-8]

    #print '\n m_min = %.4e'%m_min , 'g_min = %.4e'%g_min, 'x2_min = %.4e'%x2_min, 'b0_min = %.4e'%b0_min, 'b0_mean = %.4e'%var_mean['b0'], 'b1_min = %.4e'%b1_min ,'b1_mean = %.4e'%var_mean['b1'] , 'b2_min = %.4e'%b2_min, 'b2_mean = %.4e'%var_mean['b2'],'b3_min = %.4e'%b3_min, 'b3_mean = %.4e'%var_mean['b3'], 'b6_min = %.4e'%b6_min, 'b6_mean = %.4e'%var_mean['b6']

    P = array(P)
    X = P[:,1:,-3]
    Y = P[:,1:,-2]
    Z = P[:,1:,-1]

    import matplotlib.pyplot as plt
    from matplotlib import ticker
    plt.rc('text', usetex=False)
    plt.figure(figsize=(9, 6), dpi= 120)
    cf = plt.contourf(X,Y,Z,100)
    cb = plt.colorbar(cf)
    cb.set_label('$\chi^{2}$')
    plt.ylabel('$m_{a} (neV) $')
    plt.xlabel('$g_{a\gamma} ( \\times 10^{-11} GeV^{-1})$')
    plt.set_cmap('gist_earth')
    plt.title('ALP with bfield')
    if(PSR_sim%20==0): plt.savefig('Result/sim_bfield_%d.png'%PSR_sim)
    plt.close('all')

    with open('Result/sim_bfield_%d.txt'%PSR_sim, 'w') as outfile:
        outfile.write('# mALP_min = %.4e , gALP_min = %.4e , x2_min = %.4e \n \n'%(m_min,g_min,x2_min))
        outfile.write('# Array shape: {0}\n \n'.format(P.shape))
        outfile.write('# N0 \t G0 \t E0 \t N1 \t G1 \t E1 \t N2 \t G2 \t E2 \t N3 \t G3 \t E3 \t b0 \t b1 \t b2 \t b3 \t b6 \t m_a \t g_ag \t x2 \n \n')
        for data_slice in P:
            np.savetxt(outfile, data_slice, fmt='%-10.5f \t')
            outfile.write('\n # New slice\n \n')

