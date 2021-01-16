import numpy as np
import torch
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from .kernel_estimation import kernel_estimator
from .adaptation import select_bandwidth_lepski, omega

pi = np.pi


# def collect_mse_convergence(beta, kernel, sample_f, x, f_gt, grid, n_repet):
#     l, d, lambda_d = kernel.l, kernel.d, kernel.lambda_d
#     mse_n = np.zeros(len(grid))
#     #try:
#     for i, n in tqdm(enumerate(grid)):
#         n = int(n)
#         mse = 0.
#         for _ in range(n_repet):
#             manifold_pts = sample_f(beta, n)

#             bandwidth = n**(-1./(2.*beta + d))

#             f_est = kernel_estimator(x, manifold_pts, bandwidth, kernel)

#             mse += (f_gt - f_est)**2 / n_repet
#         mse_n[i] = mse
#     model = LinearRegression()
#     model.fit(np.log(grid).reshape(len(grid), 1), np.log(mse_n))
#     return model.coef_, model.intercept_, mse_n
    # except KeyboardInterrupt:
    #     mse_n = mse_n[:i]
    #     model = LinearRegression()
    #     model.fit(np.log(grid[:i]).reshape(len(mse_n), 1), np.log(mse_n))
    #     return model.coef_, model.intercept_, mse_n

def collect_mse_convergence(beta, kernel, sample_f, x, f_gt, grid, n_repet):
    l, d, lambda_d = kernel.l, kernel.d, kernel.lambda_d
    mse_n = np.zeros(len(grid))
    x = x.unsqueeze(0).repeat(n_repet, 1)
    #try:
    for i, n in tqdm(enumerate(grid)):
        n = int(n)
        manifold_pts = sample_f(beta, n*n_repet)
        manifold_pts = manifold_pts.reshape(n_repet, n, -1)

        bandwidth = n**(-1./(2.*beta + d))

        f_est = kernel_estimator(x, manifold_pts, bandwidth, kernel)
        mse = ((f_gt - f_est)**2).mean(0)
        mse_n[i] = mse
    model = LinearRegression()
    model.fit(np.log(grid).reshape(len(grid), 1), np.log(mse_n))
    return model.coef_, model.intercept_, mse_n


# def collect_mse_convergence_adaptation(beta, kernel, sample_f, x, f_gt, grid, n_repet, f_max, theta):
#     l, d, lambda_d = kernel.l, kernel.d, kernel.lambda_d
#     mse_n = np.zeros(len(grid))
#     #try:
#     for i, n in tqdm(enumerate(grid)):
#         n = int(n)
#         mse = 0.
#         band = 0
#         for _ in range(n_repet):
#             manifold_pts = sample_f(beta, int(n))

#             optimal_bandwidth = n**(-1./(2.*beta + d))
#             bandwidth = select_bandwidth_lepski(x, manifold_pts, kernel, f_max, theta, npts=100)
#             band += bandwidth / n_repet
            
#             f_est = kernel_estimator(x, manifold_pts, bandwidth, kernel)

#             mse += (f_gt - f_est)**2 / n_repet
#         print(band, optimal_bandwidth)
#         mse_n[i] = mse
#     model = LinearRegression()
#     model.fit(np.log(grid).reshape(len(grid), 1), np.log(mse_n))
#     return model.coef_, model.intercept_, mse_n
    # except KeyboardInterrupt:
    #     mse_n = mse_n[:i]
    #     model = LinearRegression()
    #     model.fit(np.log(grid[:i]).reshape(len(mse_n), 1), np.log(mse_n))
    #     return model.coef_, model.intercept_, mse_n


def collect_mse_convergence_adaptation(beta, kernel, sample_f, x, f_gt, grid, n_repet, f_max, theta):
    l, d, lambda_d = kernel.l, kernel.d, kernel.lambda_d
    mse_n = np.zeros(len(grid))
    x = x.unsqueeze(0).repeat(n_repet, 1)
    #try:
    for i, n in tqdm(enumerate(grid)):
        n = int(n)
        manifold_pts = sample_f(beta, n*n_repet)
        manifold_pts = manifold_pts.reshape(n_repet, n, -1)

        optimal_bandwidth = n**(-1./(2.*beta + d))
        #print(omega(optimal_bandwidth, d, n, l, f_max, kernel.inf_norm))
        bandwidth = select_bandwidth_lepski(x[:1], manifold_pts[:1], kernel, f_max, theta, npts=100)
        bandwidth = bandwidth[0]

        f_est = kernel_estimator(x, manifold_pts, bandwidth, kernel)
        mse = ((f_gt - f_est)**2).median(0)[0]
        mse_n[i] = mse
        print(f'bandwidth: {bandwidth:.4f}, optimal: {optimal_bandwidth:.4f}')
    model = LinearRegression()
    model.fit(np.log(grid).reshape(len(grid), 1), np.log(mse_n))
    return model.coef_, model.intercept_, mse_n


def plot_convergence(grid, mse_n, coef, bias, expected_rate, savepath=None, error_name='MSE'):
    plt.plot(np.log(grid)/np.log(10), np.log(mse_n)/np.log(10), label=error_name)
    plt.plot(np.log(grid)/np.log(10), np.log(grid)/np.log(10)*coef + bias/np.log(10), color='black', alpha=.3, label='linear regression', linestyle='dotted')
    plt.plot(np.log(grid)/np.log(10), np.log(grid)/np.log(10)*expected_rate - np.log(grid)[0]/np.log(10)*expected_rate+np.log(mse_n)[0]/np.log(10), color='black', alpha=.3, label='expected rate', linestyle='--')

    plt.xticks()
    plt.xlabel(r'$\log_{10} (n)$')
    plt.ylabel(fr'$\log_{10} ({error_name.lower()})$')
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

