"""
General Numerical Solver for the 1D Time-Dependent Schrodinger Equation.

Authors:
- Jake Vanderplas <vanderplas@astro.washington.edu>
- Andre Xuereb (imaginary time propagation, normalized wavefunction

For a theoretical description of the algorithm, please see
http://jakevdp.github.com/blog/2012/09/05/quantum-python/

License: BSD style
Please feel free to use and modify this, but keep the above information.
"""

import os
import sys
if sys.version_info < (3,):
    import cPickle as pickle
else:
    import pickle
import numpy as np

try:
    import pyfftw
    import pyfftw.interfaces.scipy_fftpack as fftpack
    found_pyfftw = True
    import gzip
except:
    from scipy import fftpack
    found_pyfftw = False



class Schrodinger(object):
    """
    Class which implements a numerical solution of the time-dependent
    Schrodinger equation for an arbitrary potential
    """
    def __init__(self,
                 x,
                 psi_x0,
                 V_x,
                 k0 = None,
                 hbar = 1,
                 m = 1,
                 t0 = 0.0):
        """
        Parameters
        ----------
        x : array_like, float
            Length-N array of evenly spaced spatial coordinates
        psi_x0 : array_like, complex
            Length-N array of the initial wave function at time t0
        V_x : array_like, float
            Length-N array giving the potential at each x
        k0 : float
            The minimum value of k.  Note that, because of the workings of the
            Fast Fourier Transform, the momentum wave-number will be defined
            in the range
              k0 < k < 2*pi / dx ,
            where dx = x[1]-x[0].  If you expect nonzero momentum outside this
            range, you must modify the inputs accordingly.  If not specified,
            k0 will be calculated such that the range is [-k0,k0]
        hbar : float
            Value of Planck's constant (default = 1)
        m : float
            Particle mass (default = 1)
        t0 : float
            Initial time (default = 0)
        """
        # Validation of array inputs
        self.x, psi_x0, self.V_x = map(np.asarray, (x, psi_x0, V_x))
        N = self.x.size
        assert self.x.shape == (N,)
        assert psi_x0.shape == (N,)
        assert self.V_x.shape == (N,)

        # Validate and set internal parameters
        assert hbar > 0
        assert m > 0
        self.hbar = hbar
        self.m = m
        self.t = t0
        self.dt_ = None
        self.N = len(x)
        self.dx = self.x[1] - self.x[0]
        self.dk = 2 * np.pi / (self.N * self.dx)

        # Set momentum scale
        if k0 == None:
            self.k0 = -0.5 * self.N * self.dk
        else:
            assert k0 < 0
            self.k0 = k0
        self.k = self.k0 + self.dk * np.arange(self.N)

        self.psi_x = psi_x0
        self.psi_mod_k = fftpack.fft(self.psi_mod_x)

        # Variables which hold steps in evolution
        self.x_evolve_half = None
        self.x_evolve = None
        self.k_evolve = None

        if found_pyfftw:
            # Align arrays for pyFFTW
            self.psi_mod_k = pyfftw.n_byte_align(
                self.psi_mod_k,
                pyfftw.simd_alignment)
            self.psi_mod_x = pyfftw.n_byte_align(
                self.psi_mod_x,
                pyfftw.simd_alignment)
            # Try to read any wisdom from file
            if (os.path.isfile('fftw_wisdom.pickle.gz')):
                pyfftw.import_wisdom(
                    pickle.load(gzip.open('fftw_wisdom.pickle.gz', 'rb')))
            print('about to initialize the fftw plans, which can take a while')
            self.k_from_x_plan = pyfftw.FFTW(
                    self.psi_mod_x, self.psi_mod_k,
                    direction = 'FFTW_FORWARD',
                    flags = ('FFTW_MEASURE',))
            self.x_from_k_plan = pyfftw.FFTW(
                    self.psi_mod_k, self.psi_mod_x,
                    direction = 'FFTW_BACKWARD',
                    flags = ('FFTW_MEASURE',))
            print('finalized fftw initialization')
            # Save wisdom to file
            bla = pyfftw.export_wisdom()
            pickle.dump(bla, gzip.open('fftw_wisdom.pickle.gz', 'wb'))

    def _set_psi_x(self, psi_x):
        assert psi_x.shape == self.x.shape
        self.psi_mod_x = (psi_x * np.exp(-1j * self.k[0] * self.x)
                          * self.dx / np.sqrt(2 * np.pi))
        self.psi_mod_x /= self.norm
        self.psi_mod_k = fftpack.fft(self.psi_mod_x)

    def _get_psi_x(self):
        return (self.psi_mod_x * np.exp(1j * self.k[0] * self.x)
                * np.sqrt(2 * np.pi) / self.dx)

    def _set_psi_k(self, psi_k):
        assert psi_k.shape == self.x.shape
        self.psi_mod_k = psi_k * np.exp(1j * self.x[0] * self.dk
                                        * np.arange(self.N))
        self.psi_mod_x = fftpack.ifft(self.psi_mod_k)
        self.psi_mod_k = fftpack.fft(self.psi_mod_x)

    def _get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * self.x[0] * self.dk
                                        * np.arange(self.N))

    def _get_dt(self):
        return self.dt_

    def _set_dt(self, dt):
        assert dt != 0
        if dt != self.dt_:
            self.dt_ = dt
            self.x_evolve_half = np.exp(-0.5 * 1j * self.V_x
                                         / self.hbar * self.dt)
            self.x_evolve = self.x_evolve_half * self.x_evolve_half
            self.k_evolve = np.exp(-0.5 * 1j * self.hbar / self.m
                                    * (self.k * self.k) * self.dt)/self.x.shape[0]

    def _get_norm(self):
        return self.wf_norm(self.psi_mod_x)

    psi_x = property(_get_psi_x, _set_psi_x)
    psi_k = property(_get_psi_k, _set_psi_k)
    norm = property(_get_norm)
    dt = property(_get_dt, _set_dt)

    def compute_k_from_x(self):
        self.psi_mod_k = fftpack.fft(self.psi_mod_x)

    def compute_x_from_k(self):
        self.psi_mod_x = fftpack.ifft(self.psi_mod_k)

    def wf_norm(self, wave_fn):
        """
        Returns the norm of a wave function.

        Parameters
        ----------
        wave_fn : array
            Length-N array of the wavefunction in the position representation
        """
        assert wave_fn.shape == self.x.shape
        return np.sqrt((abs(wave_fn) ** 2).sum() * 2 * np.pi / self.dx)

    def solve(self, dt, Nsteps=1, eps=1e-3, max_iter=1000):
        """
        Propagate the Schrodinger equation forward in imaginary
        time to find the ground state.

        Parameters
        ----------
        dt : float
            The small time interval over which to integrate
        Nsteps : float, optional
            The number of intervals to compute (default = 1)
        eps : float
            The criterion for convergence applied to the norm (default = 1e-3)
        max_iter : float
            Maximum number of iterations (default = 1000)
        """
        eps = abs(eps)
        assert eps > 0
        t0 = self.t
        old_psi = self.psi_x
        d_psi = 2 * eps
        num_iter = 0
        while (d_psi > eps) and (num_iter <= max_iter):
            num_iter += 1
            self.time_step(-1j * dt, Nsteps)
            d_psi = self.wf_norm(self.psi_x - old_psi)
            old_psi = 1. * self.psi_x
        self.t = t0

    def time_step_FFTW(self, dt, Nsteps=1):
        """
        Perform a series of time-steps via the time-dependent Schrodinger
        Equation.

        Parameters
        ----------
        dt : float
            The small time interval over which to integrate
        Nsteps : float, optional
            The number of intervals to compute.  The total change in time at
            the end of this method will be dt * Nsteps (default = 1)
        """
        assert Nsteps >= 0
        self.dt = dt
        if Nsteps > 0:
            self.psi_mod_x *= self.x_evolve_half
            for num_iter in xrange(Nsteps - 1):
                self.k_from_x_plan.execute()
                self.psi_mod_k *= self.k_evolve
                self.x_from_k_plan.execute()
                self.psi_mod_x *= self.x_evolve
            self.k_from_x_plan.execute()
            self.psi_mod_k *= self.k_evolve
            self.x_from_k_plan.execute()
            self.psi_mod_x *= self.x_evolve_half
            self.k_from_x_plan.execute()
            self.psi_mod_x /= self.norm
            self.k_from_x_plan.execute()
            self.t += dt * Nsteps
        return None

    def time_step_fftpack(self, dt, Nsteps=1):
        """
        Perform a series of time-steps via the time-dependent Schrodinger
        Equation.

        Parameters
        ----------
        dt : float
            The small time interval over which to integrate
        Nsteps : float, optional
            The number of intervals to compute.  The total change in time at
            the end of this method will be dt * Nsteps (default = 1)
        """
        assert Nsteps >= 0
        self.dt = dt
        if Nsteps > 0:
            self.psi_mod_x *= self.x_evolve_half
            for num_iter in xrange(Nsteps - 1):
                self.psi_mod_k = fftpack.fft(self.psi_mod_x)
                self.psi_mod_k *= self.k_evolve
                self.psi_mod_x = fftpack.ifft(self.psi_mod_k)
                self.psi_mod_x *= self.x_evolve
            self.psi_mod_k = fftpack.fft(self.psi_mod_x)
            self.psi_mod_k *= self.k_evolve
            self.psi_mod_x = fftpack.ifft(self.psi_mod_k)
            self.psi_mod_x *= self.x_evolve_half
            self.psi_mod_k = fftpack.fft(self.psi_mod_x)
            self.psi_mod_x /= self.norm
            self.psi_mod_k = fftpack.fft(self.psi_mod_x)
            self.t += dt * Nsteps
        return None

    def evolve(
            self,
            nsteps,
            nsubsteps,
            dt,
            base_name = None,
            base_info = {}):
        if type(base_name) != type(None):
            if os.path.isfile(base_name + '_psi_x_full.npy'):
                self.psi_x_full = np.load(
                    base_name + '_psi_x_full.npy')
                self.time = np.load(base_name + '_t.npy')
                self.x    = np.load(base_name + '_x.npy')
                self.k    = np.load(base_name + '_k.npy')
                self.V_x  = np.load(base_name + '_V.npy')
                base_info.update(pickle.load(open(base_name + '_info.pickle', 'r')))
                self.dispersion_vs_t = \
                    np.sqrt(np.sum((np.abs(self.psi_x_full)**2)*self.x**2*self.dx, axis = 1) -
                            np.sum((np.abs(self.psi_x_full)**2)*self.x*self.dx, axis = 1)**2)
                return base_info
        self.psi_x_full = np.zeros((nsteps+1,) + self.psi_x.shape,
                                   self.psi_x.dtype)
        self.psi_x_full[0] = self.psi_x
        self.time = np.zeros(nsteps+1, type(self.t))
        self.time[0] = self.t
        if found_pyfftw:
            self.time_step = self.time_step_FFTW
        else:
            self.time_step = self.time_step_fftpack
        for step in range(nsteps):
            print('at step {0} of {1}'.format(step+1, nsteps))
            self.time_step(dt, nsubsteps)
            self.psi_x_full[step+1] = self.psi_x
            self.time[step+1] = self.t
        self.dispersion_vs_t = \
            np.sqrt(np.sum((np.abs(self.psi_x_full)**2)*self.x**2*self.dx, axis = 1) -
                    np.sum((np.abs(self.psi_x_full)**2)*self.x*self.dx, axis = 1)**2)
        if type(base_name) != type(None):
            base_info['hbar'] = self.hbar
            self.save(base_name = base_name,
                      base_info = base_info)
        return base_info

    def save(self, base_name = 'tst', base_info = {}):
        np.save(base_name + '_psi_x_full', self.psi_x_full)
        np.save(base_name + '_t', self.time)
        np.save(base_name + '_x', self.x)
        np.save(base_name + '_k', self.k)
        np.save(base_name + '_V', self.V_x)
        base_info['hbar'] = self.hbar
        pickle.dump(base_info,
                    open(base_name + '_info.pickle', 'w'))
        return None

