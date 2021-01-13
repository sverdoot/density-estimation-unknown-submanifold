import numpy as np
import torch
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from kernel_estimator import kernel_estimator

pi = np.pi


def collect_mse_convergence(beta, l, d, lambda_d, sample_f, x, f_gt, grid, n_repet):
    mse_n = np.zeros(len(grid))
    for i, n in tqdm(enumerate(grid)):
        n = int(n)
        mse = 0.
        for _ in range(n_repet):
            manifold_pts = sample_f(beta, int(n))

            bandwidth = n**(-1./(2.*beta + 1.))

            f_est = kernel_estimator(x, manifold_pts, bandwidth, l=l, d=d, lambda_d=lambda_d)

            mse += (f_gt - f_est)**2 / n_repet
        mse_n[i] = mse
    model = LinearRegression()
    model.fit(np.log(grid).reshape(len(grid), 1), np.log(mse_n))
    return model.coef_, mse_n


def plot_convergence(grid, mse_n, coef, expected_rate, savepath=None):
    plt.plot(np.log(grid)/np.log(10), np.log(mse_n)/np.log(10), label='MSE')
    plt.plot(np.log(grid)/np.log(10), np.log(grid)/np.log(10)*coef - np.log(grid)[0]/np.log(10)*coef+np.log(mse_n)[0]/np.log(10), color='black', alpha=.3, label='linear regression', linestyle='dotted')
    plt.plot(np.log(grid)/np.log(10), np.log(grid)/np.log(10)*expected_rate - np.log(grid)[0]/np.log(10)*expected_rate+np.log(mse_n)[0]/np.log(10), color='black', alpha=.3, label='expected rate', linestyle='--')

    plt.xticks()
    plt.xlabel(r'$\log_{10} (n)$')
    plt.ylabel(r'$\log_{10} (mse)$')
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    
