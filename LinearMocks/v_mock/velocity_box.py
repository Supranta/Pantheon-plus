import numpy as np
from scipy.interpolate import interp1d

from .tools.fft import grid_r_hat, Fourier_ks

class VelocityBox:
    def __init__(self, N_SIDE, L_BOX, kh, pk):
        self.L_BOX  = L_BOX
        self.N_SIDE = N_SIDE
        self.V = L_BOX**3
        l = L_BOX/N_SIDE
        self.J = np.complex(0, 1)
        self.l = l
        self.dV = l**3
        self.k, self.k_norm = Fourier_ks(N_SIDE, l)
        OmegaM = 0.315
        self.OmegaM = OmegaM
        self.f = OmegaM**0.55
        self.r_hat_grid = grid_r_hat(N_SIDE)
        self.Pk_interp = interp1d(kh, pk)
        self.Pk_3d = self.get_Pk_3d()

    def get_Pk_3d(self):
        k_abs = np.array(self.k_norm)
        Pk_3d = 1e-20 * np.ones(k_abs.shape)
        select_positive_k = (k_abs > 1e-10)
        Pk_3d[select_positive_k] = self.Pk_interp(k_abs[select_positive_k])
        return Pk_3d

    def generate_delta_k(self):
        delta_k_real = np.random.normal(0., np.sqrt(self.Pk_3d / self.V / 2))
        delta_k_imag = np.random.normal(0., np.sqrt(self.Pk_3d / self.V / 2))

        delta_k_real[0,0,0] = 0.
        delta_k_imag[0,0,0] = 0.

        return np.array([delta_k_real, delta_k_imag])

    def get_delta_grid(self, delta_k, smooth_R=None):
        if smooth_R is None:
            smoothing_filter = 1.
        else:
            smoothing_filter = np.exp(-0.5 * self.k_norm**2 * smooth_R**2)
        delta_k_complex = delta_k[0] + self.J * delta_k[1]
        delta_x = self.V / self.dV * np.fft.irfftn(delta_k_complex * smoothing_filter)
        return delta_x

    def measure_Pk(self, delta_k, k_bins):
        Pk_sample = []
        Pk_camb   = []
        k_bin_centre_list = []
        for i in range(len(k_bins)-1):
            select_k = (self.k_norm > k_bins[i])&(self.k_norm < k_bins[i+1])
            k_bin_centre = np.exp(np.mean(np.log(self.k_norm[select_k])))
            k_bin_centre_list.append(k_bin_centre)
            if(np.sum(select_k) < 1):
                Pk_sample.append(0.)
                Pk_camb.append(0.)
            else:
                Pk_sample.append(np.mean(delta_k[0,select_k]**2 + delta_k[1,select_k]**2) * self.V)
                Pk_camb.append(np.mean(self.Pk_interp(self.k_norm[select_k])))
        return np.array(k_bin_centre_list), np.array(Pk_sample), np.array(Pk_camb)
    
    def Vr_grid(self, delta_k, cartesian_V=False, smooth_R=None):
        if smooth_R is None:
            smoothing_filter = 1.
        else:
            smoothing_filter = np.exp(-0.5 * self.k_norm**2 * smooth_R**2)
            
        delta_k_complex = delta_k[0] + self.J * delta_k[1]

        v_kx = smoothing_filter * self.J * 100 * self.f * delta_k_complex * self.k[0] / self.k_norm / self.k_norm
        v_ky = smoothing_filter * self.J * 100 * self.f * delta_k_complex * self.k[1] / self.k_norm / self.k_norm
        v_kz = smoothing_filter * self.J * 100 * self.f * delta_k_complex * self.k[2] / self.k_norm / self.k_norm

        vx = (np.fft.irfftn(v_kx) * self.V / self.dV)
        vy = (np.fft.irfftn(v_ky) * self.V / self.dV)
        vz = (np.fft.irfftn(v_kz) * self.V / self.dV)

        V = np.array([vx, vy, vz])
        if cartesian_V:
            return V
        return np.sum(V * self.r_hat_grid, axis=0)
