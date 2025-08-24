#!/usr/bin/env python
# coding: utf-8

## Import Modules 
from iminuit import Minuit as imfit
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pyymw16
import gammaALPs
from gammaALPs import core
from gammaALPs.bfields import gmf
import pymultinest
import os

import json
import numpy
from numpy import log, exp, pi
import scipy.stats, scipy
import matplotlib.pyplot as plt

jansson = gmf.GMF()

# Co-ordinate of sun w.r.t centre of galaxy
d = -8.5

def B_helio(s,l,b):
    
    # Galacto-centric cylindrical co-ordinates in terms of helio-centric co-ordinates
    r = np.sqrt(s**2*cos(b)**2 + d**2 + 2*s*d*cos(l)*cos(b))
    p = arctan2(s*sin(l)*cos(b),(s*cos(l)*cos(b)+d))
    z = s*sin(b)
    
    # Disk, Halo and X field in helio-centric co-ordinates
    B_d = jansson.Bdisk(rho=r, phi=p, z=z) 
    B_h = jansson.Bhalo(rho=r, z=z) 
    B_X = jansson.BX(rho=r,z=z)

    # Components of magnetic field in helio-centric co-ordinates
    B_r = B_d[0][0] + B_h[0][0] + B_X[0][0]
    B_p = B_d[0][1] + B_h[0][1] + B_X[0][1]
    B_z = B_d[0][2] + B_h[0][2] + B_X[0][2]
    
    # Magnetic field in helio-centric co-ordinates    
    B_s = cos(b)*(B_r*cos(l-p)+B_p*sin(l-p)) + B_z*sin(b)
    B_b = sin(b)*(B_r*cos(l-p)+B_p*sin(l-p)) - B_z*cos(b)
    B_l = B_r*sin(p-l) + B_p*cos(p-l)
    
    B_perp = (B_b**2 + B_l**2)**0.5
    Psi = np.arctan2(B_b,B_l)
    
    return B_perp,Psi

directory = '/home/gautam/Documents/Axion/MultiNest/'
pulsar_name = 'J1509'
#if not os.path.exists(directory+pulsar_name):
#    os.makedirs(directory+pulsar_name)

spec = np.loadtxt('/home/gautam/Documents/Axion/Codes/Dat/psr_spec_dat/L8Y_J1509.4m5850__flux.dat.txt')

s =  np.array([3.37])
L = np.array([319.972])
B = np.array([-0.621])
e0 = 1.

bins = 100
step = (s/bins)[0]

path = np.linspace(s[0],0, bins,endpoint = False)

#Initial and Finial States
Rf = np.matrix([[1,0,0],[0,1,0],[0,0,0]])
Ri = 0.5*np.matrix([[1,0,0],[0,1,0],[0,0,0]])

ne = np.array([pyymw16.calculate_electron_density_lbr(L[0],B[0],p) for p in path])

BP = np.array([B_helio(p,L*np.pi/180,B*np.pi/180) for p in path])
B_perp,psi = np.transpose(BP[:,0])[0],BP[:,1]
psi = np.squeeze(psi)

cp = cos(psi)
sp = sin(psi)
cp2 = cp**2
sp2 = sp**2
scp = cp*sp

def P_ga(E,m_a,g_ag):
    
    D_ag = 1.52*10**-2*(g_ag*B_perp)
    D_aa = -7.8*10**-2*(m_a**2/E) 
    D_pl = -1.1*10**-7*(ne/E)*1000
    
    # Eigenvalues of mixing matrix
    L1 = D_pl*step
    L2 = 1./2*(D_pl+D_aa-np.sqrt((D_pl-D_aa)**2 + 4*D_ag**2))*step
    L3 = 1./2*(D_pl+D_aa+np.sqrt((D_pl-D_aa)**2 + 4*D_ag**2))*step
    
    alpha = 1./2*arctan2(2.*D_ag,(D_pl-D_aa))
    
    ca = cos(alpha)
    sa = sin(alpha)
    
    ## Computation of probability
    U = np.identity(3)
    el1 = (cos(L1)+1j*sin(L1))
    el2 = (cos(L2)+1j*sin(L2))
    el3 = (cos(L3)+1j*sin(L3))
    
    ca2 = ca**2
    sa2 = sa**2
    sca = sa*ca
    
    sa2sp2 = sa2*sp2
    sa2spcp = sa2*scp
    sacasp = sca*sp
    sa2cp2 = sa2*cp2
    sacacp = sca*cp
    ca2sp2 = ca2*sp2
    ca2spcp = ca2*scp
    ca2cp2 = ca2*cp2
    
    e1cp2 = cp2*el1
    e1scp = scp*el1
    e1sp2 = sp2*el1
    
    e2sa2sp2 = el2*sa2sp2
    e2sa2spcp = el2*sa2spcp
    e2sacasp = el2*sacasp
    e2sa2cp2 = el2*sa2cp2
    e2sacacp = el2*sacacp
    e2ca2 = el2*ca2
    
    e3ca2sp2 = el3*ca2sp2
    e3ca2spcp = el3*ca2spcp
    e3sacasp = el3*sacasp
    e3ca2cp2 = el3*ca2cp2
    e3sacacp = el3*sacacp
    e3sa2 = el3*sa2
    
    T = np.array([[cp2*el1 + e2sa2sp2 + e3ca2sp2,e2sa2spcp-e1scp+e3ca2spcp,e3sacasp-e2sacasp],
                  [e3ca2spcp+e2sa2spcp-e1scp,e2sa2cp2+e1sp2+e3ca2cp2,e3sacacp-e2sacacp],
                  [e3sacasp-e2sacasp,e3sacacp-e2sacacp,e2ca2+e3sa2]])
    
    for i in range(bins): 
        U = np.dot(U,T[:,:,i])
    
    Uc = np.transpose(np.conjugate(U))
    
    return np.real(np.matrix.trace(np.dot(np.dot(np.dot(Rf,U),Ri),Uc))[0,0])

s,l,b = s,L[0],B[0]

def dNdE(E0,E,N0,G,G2,Ec,m_a,g_ag):
    
    if(m_a < 0. ): P = np.array([1. for i in range(np.size(E))])
    else : P = np.array([np.real(P_ga(i,m_a,g_ag)) for i in E])

    spectrum =  np.array([N0 * 1e-3 *(E[i]/E0)**(-G) * exp(-(E[i]/Ec)**G2)*P[i] for i in range(np.size(E))])
    spectrum = np.array([spectrum[i] * E[i]**2 *1./624150.934 for i in range(np.size(E))]) 
    return spectrum

sigma = np.sqrt(spec[:,2]**2 + (0.024*spec[:,1])**2)

def sq(cube,ndim,nparams):
	N0 , G , Ec =  cube[nN0] , cube[nG1] , cube[nEc]  
	m_a , g_ag  =  cube[nm1] , cube[ng1]
	G2,E0 = 0.54,1.
	dnde = dNdE(E0,spec[:,0]/1000,N0,G,G2,Ec,m_a,g_ag)
	chi = (dnde-spec[:,1])/sigma
	return -(np.sum(chi**2))

nN0,nG1,nEc,nm1,ng1,nG2,nE0 = 0,1,2,3,4,5,6

def prior(cube, ndim, nparams):
    cube[nN0] = cube[nN0] * 1
    cube[nG1] = cube[nG1] * 2
    cube[nEc] = cube[nEc] * 2
    cube[nm1] = cube[nm1] * 10.
    cube[ng1] = cube[ng1] * 1000. 

parameters = ["N0","G","Ec","mALP","G_ag"]
n_params = len(parameters)
n_dims = n_params

# run MultiNest
pymultinest.run(sq, prior, n_dims  ,n_params , resume = False, verbose = True, importance_nested_sampling=True,
               outputfiles_basename=(directory+pulsar_name+'/'),n_live_points=2500)
json.dump(parameters, open((directory+pulsar_name+'/params.json'), 'w')) # save parameter names
