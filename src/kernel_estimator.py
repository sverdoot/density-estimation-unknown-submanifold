import torch
import numpy as np
from matplotlib import pyplot as plt

pi = np.pi


def Lambda(z):
    l = torch.zeros(z.shape[0])
    z_norm = z.norm(p=2, dim=-1)
    l[z_norm < 1.] = torch.exp(-1./(1. - z_norm[z_norm < 1.]**2))
    return l


def gauss(x):
    sigma = .5
    return 1./(2*pi*sigma**2)**.5 * torch.exp(-x**2/(2*sigma**2))


def compute_lambda_d(d, sigma=.5, n_pts=1e7):
    n_pts = int(n_pts)
    proposal = torch.distributions.Normal(loc=torch.zeros(1), scale=sigma*torch.ones(1))
    sample = proposal.sample_n(n_pts)

    lam_val = Lambda(sample)
    n = proposal.log_prob(sample).sum(-1).exp()
    ratio = lam_val*n**(-1)
    lambda_d = ratio.mean(0).item()

    return lambda_d


def initial_kernel(z, lambda_d):
    return lambda_d ** (-1) * Lambda(z)


def kernel(z, l: int, d, lambda_d):
    if l == 1:
        return initial_kernel(z, lambda_d)
    else:
        return 2**(1 + float(d) / l) * kernel(2**(1. / l) * z, l - 1, d, lambda_d) - kernel(z, l - 1, d, lambda_d)


def kernel_estimator(x, sample, bandwidth, l, d, lambda_d=None):

    return 1./(bandwidth**d) * kernel((x[None, :] - sample) / bandwidth, l, d, lambda_d).mean()


def plot_kernels(savepath=None):
    linsp = torch.linspace(-1., 1., 100).unsqueeze(1)

    d = 1
    lambda_d = compute_lambda_d(d=d)

    k1 = kernel(linsp, 1, d, lambda_d)
    k2 = kernel(linsp, 2, d, lambda_d)
    k3 = kernel(linsp, 3, d, lambda_d)

    plt.plot(linsp[:, 0], k1, label=r'$l=1$')
    plt.plot(linsp[:, 0], k2, linestyle='--', label=r'$l=2$')
    plt.plot(linsp[:, 0], k3, linestyle='-.', label=r'$l=3$')

    plt.hlines(0, -1., 1., color='black', alpha=.3)
    plt.vlines(0, -1.5, 2.5, color='black', alpha=.3)
    plt.xlim(-1., 1.)
    plt.ylim(-1.5, 2.5)

    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
