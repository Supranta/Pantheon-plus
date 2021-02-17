import sys, os
import numpy as np
import h5py as h5
import pandas as pd

from v_mock import VelocityBox
from v_mock.tools.cosmo import z_cos, r2dL, r2mu, zCMB2zhelio, speed_of_light, camb_PS

from scipy.interpolate import RegularGridInterpolator

savedir = '../ForwardLikelihood/data/linear_mocks/mock/'

N_SIDE = 128
L_BOX = 400.
l = L_BOX / N_SIDE
OmegaM = 0.315

sig_v = 150.

kh, pk = camb_PS()

VelocityBox = VelocityBox(N_SIDE, L_BOX, kh, pk)

delta_k = VelocityBox.generate_delta_k()

delta_x = VelocityBox.get_delta_grid(delta_k, smooth_R=4.)
V_field = VelocityBox.Vr_grid(delta_k, cartesian_V=True)
Vr0 = VelocityBox.Vr_grid(delta_k, smooth_R=4.)

delta_g = delta_x.copy()
select_negative = (delta_g < 0.)
delta_g[select_negative] = -0.999999

with h5.File(savedir+'/mock_fields.h5', 'w') as f:
    f['density']  = delta_g
    f['velocity'] = V_field

def get_XYZ_grid(N_SIDE, l):
    l_grid = l * np.linspace(-0.5*(N_SIDE - 1), 0.5*(N_SIDE - 1), N_SIDE)

    X_grid = np.zeros((N_SIDE, N_SIDE, N_SIDE))
    Y_grid = np.zeros((N_SIDE, N_SIDE, N_SIDE))
    Z_grid = np.zeros((N_SIDE, N_SIDE, N_SIDE))

    for i in range(N_SIDE):
        X_grid[i,:,:] = l_grid[i]
        Y_grid[:,i,:] = l_grid[i]
        Z_grid[:,:,i] = l_grid[i]

    return X_grid, Y_grid, Z_grid

def get_pos(delta_g, nbar, N_SIDE, l):
    mu = nbar * (1. + delta_g)
    N = np.random.poisson(mu)

    X_grid, Y_grid, Z_grid = get_XYZ_grid(N_SIDE, l)

    N_max = np.max(N)

    X = np.empty(0)
    Y = np.empty(0)
    Z = np.empty(0)

    for n in range(1,N_max+1):
        select_n = (N==n)
        for i in range(n):
            X = np.hstack([X, X_grid[select_n]])
            Y = np.hstack([Y, Y_grid[select_n]])
            Z = np.hstack([Z, Z_grid[select_n]])
    assert len(X)==np.sum(N),"The X arr should have the shape of the total number of objects"
    assert len(Y)==np.sum(N),"The Y arr should have the shape of the total number of objects"
    assert len(Z)==np.sum(N),"The Z arr should have the shape of the total number of objects"

    X = X + np.random.uniform(-0.5*l, 0.5*l, size=len(X))
    Y = Y + np.random.uniform(-0.5*l, 0.5*l, size=len(Y))
    Z = Z + np.random.uniform(-0.5*l, 0.5*l, size=len(Z))
    return X, Y, Z

X, Y, Z = get_pos(delta_g, 0.005, N_SIDE, l)

def get_v_interp(X_grid, Y_grid, Z_grid, V):
    v_x_data, v_y_data, v_z_data = V
    v_x_interp = RegularGridInterpolator((X_grid, Y_grid, Z_grid), v_x_data)
    v_y_interp = RegularGridInterpolator((X_grid, Y_grid, Z_grid), v_y_data)
    v_z_interp = RegularGridInterpolator((X_grid, Y_grid, Z_grid), v_z_data)
    return v_x_interp, v_y_interp, v_z_interp

def get_Vr(X, Y, Z, v_interp):
    v_x_interp, v_y_interp, v_z_interp = v_interp

    pos = np.array([X, Y, Z])

    VX = v_x_interp(pos.T)
    VY = v_y_interp(pos.T)
    VZ = v_z_interp(pos.T)

    V_halos = np.array([VX, VY, VZ])

    pos = np.array([X, Y, Z])
    r_hat = pos / np.linalg.norm(pos, axis=0)

    V_r_halos = np.sum(V_halos * r_hat, axis=0)

    return V_r_halos

def set_boundaries(X, N_SIDE, l):
    negative_boundary = -0.5 * (N_SIDE - 1) * l
    select_negative = (X < negative_boundary)
    X[select_negative] = negative_boundary

    positive_boundary = 0.5 * (N_SIDE - 1) * l
    select_positive = (X > positive_boundary)
    X[select_positive] = positive_boundary

    return X
#=======================================
l_grid = np.linspace(-0.5*L_BOX+0.5*l, 0.5*L_BOX-0.5*l, N_SIDE)
v_interp = get_v_interp(l_grid, l_grid, l_grid, V_field)

X = set_boundaries(X, N_SIDE, l)
Y = set_boundaries(Y, N_SIDE, l)
Z = set_boundaries(Z, N_SIDE, l)

Vr_halos = get_Vr(X, Y, Z, v_interp)

Vr = Vr_halos + sig_v * np.random.normal(size=len(Vr_halos))

from astropy.coordinates import cartesian_to_spherical

coord = cartesian_to_spherical(x=X, y=Y, z=Z)

r_hMpc = np.array(coord[0])
DEC = np.array(coord[1].deg)
RA  = np.array(coord[2].deg)

dL = r2dL(r_hMpc, OmegaM)
mu_true = r2mu(dL)

M = -18.
sigma_int = 0.1
mu_obs  = mu_true + sigma_int * np.random.normal(size=len(mu_true))
app_mag = M + mu_obs

select_R = (app_mag < 17.)
N_OBJ = np.sum(select_R)

#mu_cov = np.diag(sigma_int**2 * np.ones(N_OBJ))
#print("mu_cov.shape: "+str(mu_cov.shape))
#np.save(savedir+'/mu_cov.npy',mu_cov)

print("Total number of objects: %d" %(N_OBJ))

z_cos_arr = z_cos(r_hMpc, OmegaM)
z_CMB = (1. + z_cos_arr)*(1. + Vr / speed_of_light) - 1.
z_helio = zCMB2zhelio(z_CMB, RA, DEC)

print("Max-min of r_hMpc: %2.3f, %2.3f"%(np.min(r_hMpc[select_R]), np.max(r_hMpc[select_R])))
print("Max-min of RA: %2.3f, %2.3f"%(np.min(RA), np.max(RA)))
print("Max-min of DEC: %2.3f, %2.f"%(np.min(DEC), np.max(DEC)))

df = pd.DataFrame()

df['zCMB']   = z_CMB[select_R]
df['zhelio'] = z_helio[select_R]
df['mu']     = mu_obs[select_R]
df['e_mu']   = sigma_int
df['RA']     = RA[select_R]
df['DEC']    = DEC[select_R]

df.to_csv(savedir+'/mock_PV_catalog.csv')
