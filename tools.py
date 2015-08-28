################################################################################
######################### Import Necessary Modules #############################
################################################################################

import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import subprocess
import pickle
import gzip
import os
import cProfile
import pstats

matplotlib.usetex = 1

from scipy import fftpack
from scipy import misc
from scipy import ndimage

from schrodinger import Schrodinger  #This is the solver that does all the work

################################################################################
######################### Gaussian wave-packet #################################
################################################################################

def gaussian_wavepacket(t, x, a, h, m, x0, p0):
    A = np.pi**(-1/4.)
    beta =  (1 + 1j*(h*t)/(m*a**2))
    E = p0**2/(2*m)
    phase = np.exp(1j*(p0*x - E*t)/h)
    return ((A/np.sqrt(a*beta))*phase*np.exp(-(x - x0 - (p0/m)*t)**2/(2*a**2*beta)))

def squared_wavefunction_FREE(t, x, a, h, m, x0, p0):
    beta = 1 + 1j*(h*t)/(m*a**2)
    return (1/((np.pi)**(1/2.)*a*abs(beta))*
                np.exp(-(x - x0 - (p0/m)*t)**2/(a**2 * abs(beta**2))))
                
def inverse_parabola_wavepacketBARTON(t, x, a, h, m, x0, C):
    b = .5*np.sqrt(a**2 + a**(-2))
    B = 0.5*a**2*np.cosh(C**(.5)*t)**2 + 0.5*(h/(C**(.5)*m*a))**2*np.sinh(C**(.5)*t)**2
    psi = (np.sqrt(np.sqrt(np.pi)*(a*np.cosh(t)+1j*a**(-1)*np.sinh(t))))**(-1) * np.exp(-0.5*x**2*(1-2*1j*b**2*np.sinh(2*t))*B**(-1))          
    return psi
    
def squared_wavefunctionBARTON(t, x, a, h, m, x0, C):
    B = 0.5*a**2*np.cosh(C**(.5)*t)**2 + 0.5*(h/(C**(.5)*m*a))**2*np.sinh(C**(.5)*t)**2
    psisquared = (np.sqrt(2*np.pi)*B**(0.5))**(-1) * np.exp(-0.5*(x-x0)**2*(B)**(-1))       
    return psisquared

################################################################################
######################### Analytic Dispersions #################################
################################################################################

def free_particle_dispersion(t, a, h, m):
    return (a/np.sqrt(2))*np.sqrt(1 + (h*t/(m*a**2))**2)
  
def inverted_oscillator_dispersion(t, a, h, m, C):
    return  0.5*a**2*np.cosh(C**(.5)*t)**2 + 0.5*(h/(C**(.5)*m*a))**2*np.sinh(C**(.5)*t)**2  
    
def pure_cusp_dispersion(t, a):
    return ExtremalCharacteristics(t, a)**2

def pure_cusp_k_dispersion(t, a):
    return ExtremalCharacteristicsK(t, a)**2
    
def ExtremalCharacteristics(t, a):
    nu = (2/(1-a))
    const = nu**(-1)*(2/(1+a))**(0.5)
    return (const*t)**nu
    
def ExtremalCharacteristicsK(t, a):
    nu = (2/(1-a))
    const = (2/(1+a))**(0.5)/nu
    return (const*nu)*(const*t)**(nu-1)

################################################################################
######################### Fractional Brownian Potential ########################
################################################################################

def derivative(x, mu):
    n = len(x)
    xx = np.append( np.append(
                (x[1:n/2+1][::-1]), x ),
                 x[-n/2-1:-1][::-1] ) # boundaries
    nn = 2 * n
    xft = np.fft.fft(xx)
    omega = np.fft.fftfreq(nn, d = 1./n) * 2 *np.pi 
    y = np.nan_to_num(( omega * 1j) ** mu)
    y[0] = 0.
    dxdt = np.real(np.fft.ifft( y * xft ))
    dxdt = dxdt[n/2:-n/2]
    return dxdt
  
def fbm(n = 1e5, 
        H = 1.3, 
        rseed = None):
    mu = 0.5 - H
    np.random.seed(rseed)
    x = derivative( np.random.randn(n), mu)
    x -= x.mean()
    y = derivative(x/x.std(), -1.)
    y -= np.mean(y)
    return y/y.std()

def SmoothedBrownianPotential(ell, 
                              n = 1e5, 
                              H = 1.3, 
                              rseed = None):
    bla = fbm(n, H = H, rseed = rseed)
    return ndimage.gaussian_filter(bla, ell)

################################################################################
############################### Cusp Potential #################################
################################################################################

def LocalizedSmoothConicalPotential(ell,
                                    x, 
                                    e = 1.3, 
                                    zero = 15):
    v = zero**e
    ConicalPotential = .5*((1./v)*(v - np.abs(x)**(e)))
    bumpfunction = np.piecewise(x, [abs(x) < zero, abs(x) >= zero], [1, 0])
    LocalizedConicalPotential = ConicalPotential*bumpfunction
    return  ndimage.gaussian_filter(LocalizedConicalPotential, ell)  
    
def SmoothConicalPotential(ell,
                           x, 
                           a = 1.3):
    PureConicalPotential = -(np.abs(x)**(1+a))/(1+a)
    return  ndimage.gaussian_filter(PureConicalPotential, ell)  

def InvertedOscillatorPotential(x):
    return  -(np.abs(x)**(2))/2

################################################################################
############################### Data Codes #####################################
################################################################################


class GetData:
    def __init__(self,
                 nsteps = 20,
                 resolution = 2**6,
                 timestep = 1.,
                 hbar = 1., 
                 beta = 0.5, 
                 gamma = 0.5, 
                 xmax =  2.**6*np.pi):
        self.nsteps = nsteps
        self.resolution = resolution
        self.timestep = timestep
        self.hbar = float(hbar)
        self.sigma = self.hbar**(beta) 
        self.ell = self.hbar**(gamma) 
        self.dx = np.pi * self.hbar / self.resolution
#        if self.dx / self.ell > 0.1:
#            print ('dx/ell = {0} for hbar = {1}'.format(self.dx/self.ell, self.hbar))
#            self.dx = 0.1*self.ell
        dt = self.hbar/(2*self.resolution)
        self.nsubsteps = int(self.timestep / dt)
        self.x = (self.dx*np.arange(
                      1,
                      2**(int(np.log2(xmax/self.dx)) + 1) + 1).astype(np.float) -
                  xmax)
        self.x0 = 0.0
        self.p0 = 0.0
        self.m  = 1.0
        self.psi_x0 = gaussian_wavepacket(0,
                                          self.x,
                                          self.sigma,
                                          self.hbar,
                                          self.m,
                                          self.x0,
                                          self.p0)
        self.sol = {}
        return None
    def compute(self,
                base_name = None,
                V = None,
                base_info = {}):
        self.sol[base_name] = Schrodinger(
                x = self.x,
                psi_x0 = self.psi_x0,
                V_x = V,
                hbar = self.hbar,
                m = self.m)
        self.sol[base_name].evolve(
            self.nsteps,
            self.nsubsteps,
            self.timestep/self.nsubsteps,
            base_name = base_name,
            base_info = base_info)
        return self.sol[base_name]
    
class read_solution:
    def __init__(
            self,
            base_name = None):
        if type(base_name) != type(None):
            if os.path.isfile(base_name + '_psi_x_full.npy'):
                self.psi_x_full = np.load(
                    base_name + '_psi_x_full.npy')
                self.psi_k_full = np.load(
                    base_name + '_psi_k_full.npy')
                self.time = np.load(base_name + '_t.npy')
                self.x    = np.load(base_name + '_x.npy')
                self.k    = np.load(base_name + '_k.npy')
                self.dx = self.x[1] - self.x[0]
                self.dk = self.k[1] - self.k[0]
                self.k    = np.load(base_name + '_k.npy')
                self.V_x  = np.load(base_name + '_V.npy')
                self.dispersion_vs_t = \
                    np.sqrt(np.sum((np.abs(self.psi_x_full)**2)*self.x**2*self.dx, axis = 1) -
                            np.sum((np.abs(self.psi_x_full)**2)*self.x*self.dx, axis = 1)**2)
                self.kdispersion_vs_t = \
                    np.sqrt(np.sum((np.abs(self.psi_k_full)**2)*self.k**2*self.dk, axis = 1) -
                            np.sum((np.abs(self.psi_k_full)**2)*self.k*self.dk, axis = 1)**2)
        return None

