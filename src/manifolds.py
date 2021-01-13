import numpy as np
import torch
import math
from functools import partial
from matplotlib import pyplot as plt
import random
import seaborn as sns
from matplotlib import cm

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, jacobian

from sklearn.linear_model import LinearRegression
from tqdm import tqdm

pi = np.pi


def example_1d_density(v: torch.Tensor, beta: float):
    
    assert (v <= .5).all() and (v >= -.5).all()

    g = torch.zeros_like(v)
    g[v < 0] = (1 - (-2*v[v < 0])**beta)
    g[v >= 0] = (1 - (2*v[v >= 0])**(beta + 1))

    return g


def example_2d_density(v: torch.Tensor, u: torch.Tensor, beta: float):
    
    gu = example_1d_density(u, beta)
    gv = example_1d_density(v, beta)

    return gu * gv


def compute_norm_constant(density_f: callable, min_value, max_value, n_pts=1000):
    assert max_value - min_value < float('inf')
    assert max_value > min_value
    linsp = torch.linspace(min_value, max_value, n_pts+2)[1:-1]
    values = density_f(linsp)
    norm_const = values.sum() * (linsp[1] - linsp[0])
    return norm_const


def plot_example_1d_density(beta: float, linestyle='-'):
    min_value = -.5
    max_value = .5

    unnormed_g = partial(example_1d_density, beta=beta)
    norm_const = compute_norm_constant(unnormed_g, min_value, max_value, n_pts=1000)

    linsp = torch.linspace(min_value, max_value, 1000)
    values = unnormed_g(linsp) / norm_const

    plt.plot(linsp.numpy(), values.numpy(), label=fr'$\beta={beta}$', linestyle=linestyle)


def plot_example_2d_density(ax, beta: float, linestyle='-'):
    min_value = -.5
    max_value = .5

    unnormed_g = partial(example_2d_density, beta=beta)

    norm_const = compute_norm_constant(partial(example_1d_density, beta=beta), min_value, max_value, n_pts=1000)

    linsp_x = torch.linspace(min_value, max_value, 1000)
    linsp_y = torch.linspace(min_value, max_value, 1000)
    mesh = torch.meshgrid(linsp_x, linsp_y)

    values = unnormed_g(mesh[0].reshape(-1), mesh[1].reshape(-1)) / norm_const**2

    values = values.view(mesh[0].shape)

    surf = ax.plot_surface(mesh[0], mesh[1], values.numpy(), cmap=cm.Blues, linewidth=0)#, antialiased=False) #, label=fr'$\beta={beta}$', linestyle=linestyle)
    return surf


def plot_example_1d_densities(savepath=None):
    fig = plt.figure(figsize=(5,4))

    plot_example_1d_density(beta=2., linestyle='-')
    plot_example_1d_density(beta=4., linestyle='--')
    plot_example_1d_density(beta=8., linestyle='-.')

    plt.hlines(0, -.5, .5, linestyle='--', color='black', alpha=.5)
    plt.vlines(0, 0, 2., linestyle='--', color='black', alpha=.5)
    plt.ylim(-.05, 1.5)

    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


def examlpe_1d_density_sampling(beta, n_pts):
    unnormed_g = partial(example_1d_density, beta=beta)

    pts = torch.zeros(n_pts)
    cnt = 0
    while cnt < n_pts:
        u = random.random()
        v = torch.rand(1) -.5
        if u < unnormed_g(v) / 1.:
           pts[cnt] = v 
           cnt += 1
    return pts


def example_2d_density_sampling(beta, n_pts):
    unnormed_g = partial(example_1d_density, beta=beta)

    pts = torch.zeros(n_pts, 2)
    pts[:, 0] = examlpe_1d_density_sampling(beta, n_pts)
    pts[:, 1] = examlpe_1d_density_sampling(beta, n_pts)

    return pts


def example_parametric_curve(v, a, w):
    # if isinstance(v, float):
    #     return np.hstack((np.cos(2*pi*v) + a * np.cos(2*pi*w*v), np.sin(2*pi*v) + a * np.sin(2*pi*w*v)))
    #     #v = torch.tensor([v])
    # v = torch.tensor(v)
    if isinstance(v, np.ndarray):
        return np.vstack((np.cos(2*pi*v) + a * np.cos(2*pi*w*v), np.sin(2*pi*v) + a * np.sin(2*pi*w*v))).T

    elif isinstance(v, jnp.ndarray):
        return jnp.vstack((jnp.cos(2*pi*v) + a * jnp.cos(2*pi*w*v), jnp.sin(2*pi*v) + a * jnp.sin(2*pi*w*v))).T

    elif isinstance(v, torch.Tensor):
        return torch.stack((torch.cos(2*pi*v) + a * torch.cos(2*pi*w*v), torch.sin(2*pi*v) + a * torch.sin(2*pi*w*v)), dim=1)


def example_parametric_surface(v, u, a, b, w):
    if isinstance(v, np.ndarray):
        return np.vstack((np.cos(2*pi*v) + a * np.cos(2*pi*w*v), np.sin(2*pi*v) + a * np.sin(2*pi*w*v))).T

    elif isinstance(v, jnp.ndarray):
        return jnp.vstack((jnp.cos(2*pi*v) + a * jnp.cos(2*pi*w*v), jnp.sin(2*pi*v) + a * jnp.sin(2*pi*w*v))).T

    elif isinstance(v, torch.Tensor):
        return torch.stack((
            (b+torch.cos(2*pi*v))*torch.cos(2*pi*u) + a * torch.sin(2*pi*w*v), 
            (b+torch.cos(2*pi*v))*torch.sin(2*pi*u) + a * torch.cos(2*pi*w*v),
            torch.sin(2*pi*v) + a * torch.sin(2*pi*w*u)
            ), dim=1)

