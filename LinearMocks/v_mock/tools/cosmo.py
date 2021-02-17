import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import camb
from camb import model, initialpower

speed_of_light = 299792
coord = SkyCoord(l=264.021*u.deg, b=48.253*u.deg, frame='galactic')
v_sun_r_hat = np.array(coord.icrs.cartesian.xyz).reshape((-1,1))
V_sun = 369.82

def z_cos(r_hMpc, Omega_m):
    Omega_L = 1. - Omega_m
    q0 = Omega_m/2.0 - Omega_L
    return (1.0 - np.sqrt(1 - 2*r_hMpc*100*(1 + q0)/speed_of_light))/(1.0 + q0)

def r2dL(r, OmegaM):
    z_cos_arr = z_cos(r, OmegaM)
    return r * (1 + z_cos_arr)

def r2mu(r):
    return 5 * np.log10(r) + 25

def mu2r(mu):
    return 10**((mu - 25.)/5.)

def zCMB2zhelio(zCMB, RA, DEC):
    """
    See e.g, 1812.05135. V_sun taken from 1807.06205
    """
    r_hat = np.array(SkyCoord(ra=RA*u.deg, dec=DEC*u.deg).cartesian.xyz)
    z_sun = V_sun / speed_of_light * np.sum(r_hat * v_sun_r_hat, axis=0)
    return (zCMB - z_sun)/(1. + z_sun)

def camb_PS():
    print("Calculating CAMB power spectrum....")
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)

    pars.set_matter_power(redshifts=[0.], kmax=2.0)

    results = camb.get_results(pars)

    #Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh, _, pk = results.get_matter_power_spectrum(minkh=1e-2, maxkh=5, npoints = 2000)

    return kh, pk[0]
