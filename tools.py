################################################################################
######################### Import Necessary Modules #############################
################################################################################

import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

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

def inverse_parabola_wavepacket(t, x, a, h, m, x0, p0, c):
    const = (2*c*np.sqrt(2*m))/h**2
    scaledT = (const)**(-1)*(1-np.exp(-const*t))
    scaledX = x*np.exp(-c*np.sqrt(2*m)*t/h**2)
    phase = np.exp((1j*np.sqrt(2*m)*c*x**2)/(2*h))
    scale = np.exp(-c*t/np.sqrt(2*m))
    return gaussian_wavepacket(scaledT, scaledX, a, h, m, x0, p0)*phase*scale

def squared_wavefunction(t, x, a, h, m, x0, p0):
    beta = 1 + 1j*(h*t)/(m*a**2)
    return (1/((np.pi)**(1/2.)*a*abs(beta))*
                np.exp(-(x - x0 - (p0/m)*t)**2/(a**2 * abs(beta**2))))

################################################################################
######################### Analytic Dispersions #################################
################################################################################

def free_particle_dispersion(t, a, h, m):
    return (a/np.sqrt(2))*np.sqrt(1 + (h*t/(m*a**2))**2)

def parabolic_dispersion_Hagedorn(t, a, c):
    return  (1/np.sqrt(2))*t*a*np.cosh(2*c*t)

def ExtremalCharacteristics(t, alpha):
    const = ((1-alpha)**2/(2*(1+alpha)))**(1/(1-alpha))
    exponent = (2/(1-alpha))
    return const*t**exponent, - constant0*t**exponent

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
    return  ndimage.gaussian_filter(ConicalPotential*bumpfunction, ell)  

class GetData:
    def __init__(self,
                 nsteps = 20,
                 resolution = 2**6,
                 timestep = 1.,
                 hbar = 1., 
                 xmax =  2.**6*np.pi):
        self.nsteps = nsteps
        self.resolution = resolution
        self.timestep = timestep
        self.hbar = float(hbar)
        self.sigma = self.hbar**(0.5) 
        self.ell = 10.0*self.hbar
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
                self.time = np.load(base_name + '_t.npy')
                self.x    = np.load(base_name + '_x.npy')
                self.dx = self.x[1] - self.x[0]
                self.k    = np.load(base_name + '_k.npy')
                self.V_x  = np.load(base_name + '_V.npy')
                self.dispersion_vs_t = \
                    np.sqrt(np.sum((np.abs(self.psi_x_full)**2)*self.x**2*self.dx, axis = 1) -
                            np.sum((np.abs(self.psi_x_full)**2)*self.x*self.dx, axis = 1)**2)
        return None

