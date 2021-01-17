import torch
import numpy as np
from matplotlib import pyplot as plt

pi = np.pi


def Lambda(z):
    l = torch.zeros(z.shape[:-1])
    z_norm = z.norm(p=2, dim=-1)
    l[z_norm < 1.] = torch.exp(-1./(1. - z_norm[z_norm < 1.]**2))
    return l


def gauss(x, sigma=.5):
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


class Kernel(object):
    def __init__(self, l, d, lambda_d=None):
        self.l = l
        self.d = d
        if lambda_d is None:
            self.lambda_d = compute_lambda_d(d)
        else:
            self.lambda_d = lambda_d
        self.inf_norm = self.kernel_inf_norm()

    @staticmethod
    def initial_kernel(z, lambda_d):
        return lambda_d ** (-1) * Lambda(z)

    def kernel_inf_norm(self):
        linsp = torch.linspace(-1., 1., 100).unsqueeze(1)
        d = 1.
        k = self(linsp)
        max_val = torch.max(torch.abs(k)).item()
        return max_val

    def __call__(self, z, l=None, d=None):
        l = self.l if l is None else l
        d = self.d #if d is None else d
        if l == 1:
            return Kernel.initial_kernel(z, self.lambda_d)
        else:
            return 2**(1. + float(d) / l) * self(2**(1. / l) * z, l - 1, d) - self(z, l - 1, d)


def kernel_estimator(x, sample, bandwidth, kernel): # l, d, lambda_d=None):
    if x.ndim == 1 and sample.ndim == 2:
        x = x.unsqueeze(0)
        sample = sample.unsqueeze(0)
    d = kernel.d
    if isinstance(bandwidth, (int, float, complex)):
        f = 1./(bandwidth**d) * (kernel((x[:, None, :] - sample) / bandwidth).mean(-1))
    else:
        f = 1./(bandwidth**d) * (kernel((x[:, None, :] - sample) / bandwidth[:, None, None]).mean(-1))
    return f


def plot_kernels(*ls, savepath=None):
    linestyles = [
     'solid',
     'dotted',
     'dashed',
     'dashdot',] 

    linsp = torch.linspace(-1., 1., 100).unsqueeze(1)

    d = 1
    lambda_d = compute_lambda_d(d=d)
    y_max = 0.
    y_min = 0.

    for i, l in enumerate(ls):
        linestyle = linestyles[i % len(linestyles)]
        kernel = Kernel(l, d, lambda_d)
        k = kernel(linsp)
        if torch.max(k).item() > y_max:
            y_max =  torch.max(k).item()
        if torch.min(k).item() < y_min:
            y_min =  torch.min(k).item()
        plt.plot(linsp[:, 0], k, linestyle=linestyle, label=fr'$l={l}$')

    plt.hlines(0, -1., 1., color='black', alpha=.3)
    plt.vlines(0, y_min, y_max, color='black', alpha=.3)
    plt.xlim(-1., 1.)
    plt.ylim(y_min, y_max)

    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
