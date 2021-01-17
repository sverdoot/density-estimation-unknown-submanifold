import torch
import numpy as np
import math

from .kernel_estimation import kernel_estimator

pi = np.pi


def bandwidth_lower_bound(d, n, l, f_max, kernel_inf_norm):
    zeta = unit_ball_volume(d)
    k_inf_norm = kernel_inf_norm
    w = 4.**d * zeta * k_inf_norm ** 2 * f_max
    h_lower = (k_inf_norm / 2. / w)**(1./d) * n ** (-1./d)
    return h_lower


def omega(h, d, n, l, f_max, kernel_inf_norm):
    zeta = unit_ball_volume(d)
    k_inf_norm = kernel_inf_norm
    w = 4.**d * zeta * k_inf_norm ** 2 * f_max
    omega_ = (2 * w / n / h ** d) **.5 + k_inf_norm / n / h**d
    return omega_

def omega2(h, d, n):
    omega_ = (1./n/h**d)**.5
    return omega_


def lamda(h, d, theta):
    return np.clip((theta * d * np.log(1. / h))**.5, a_min=1.0, a_max=None)


def psy(h, eta, d, n, l, f_max, kernel_inf_norm, theta):
    return omega(h, d, n, l, f_max, kernel_inf_norm) * lamda(h, d, theta) + \
        omega(eta, d, n, l, f_max, kernel_inf_norm) * lamda(eta, d, theta)

    
def psy2(h, eta, d, n, l, f_max, kernel_inf_norm, theta):
    return omega2(h, d, n) * lamda(h, d, theta) + \
        omega2(eta, d, n) * lamda(eta, d, theta)


def select_bandwidth_lepski(x, sample, kernel, f_max, theta, npts=100, psy_type:int = 1):
    d, l, kernel_inf_norm = kernel.d, kernel.l, kernel.inf_norm
    n = sample.shape[1]
    lb = bandwidth_lower_bound(d, n, l, f_max, kernel_inf_norm)
    j = np.linspace(0, np.log(1./lb) / np.log(2.), npts)
    H = 2**(-j)[::-1]

    result = np.zeros(x.shape[0])
    f_etas = np.empty([x.shape[0]])
    for i, h in enumerate(H):
        etas = H[:i+1]
        
        if psy_type == 1:
            psy_f = psy
        elif psy_type == 2:
            psy_f = psy2
        else:
            raise KeyError
            
        psys = psy_f(h, etas, d, n, l, f_max, kernel_inf_norm, theta)
        
        f_h = kernel_estimator(x, sample, h, kernel).detach().cpu().numpy()
        if i == 0:
            f_etas = f_h.reshape(-1, 1)
        else:
            f_etas = np.concatenate([f_etas, f_h.reshape(-1, 1)], axis=1)
        diff = np.abs(f_etas - f_h[:, None])
        check = np.all(diff <= psys[None, :], axis=1)
        result[check] = h
    return result
    

def unit_ball_volume(d):
    gamma = math.gamma(d / 2. + 1.)
    volume = pi ** (d / 2.) / gamma
    return volume
    