import torch as tr

if tr.backends.mps.is_available():
    device = tr.device("mps")
    x = tr.ones(1, device=device)
    print (x)
elif tr.cuda.is_available():
    device = tr.device("cuda")
    x = tr.ones(1, device=device)
    print (x)
else:
    print ("MPS or cuda device not found.")

from GP import *
import torch as tr
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
#from puwr import *
#import torchinterp1d

from torch import nn
from torch.nn import functional as Func
from sklearn import datasets
import scipy.special as sc




import scipy.integrate as integrate
from torch.special import gammaln
#from orthogonal_poly import legendre_01

from torch.autograd.functional import hessian

import scipy.integrate as integrate

import h5py as h5

# import all packages and set plots to be embedded inline
import numpy as np 
import datetime
from scipy.optimize import minimize 
from scipy import special 
from scipy.optimize import Bounds 
from scipy.linalg import cho_solve 
#from pyDOE import lhs 
import time
#load all the Models and Kernels
from functions import *


import matplotlib.pyplot as plt
import matplotlib as mpl

import os
os.environ["PATH"] = "/sciclone/home/yacahuanamedra/texlive/bin/x86_64-linux:" + os.environ["PATH"]
from typing import Optional
import glob
import subprocess
import argparse

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')
import pickle


from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

mpl.rc('font', **font)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

mpl.rc('font', **font)




def parse_cli_args():
    parser = argparse.ArgumentParser(description="Train INN for PDF/ITD mapping.")
    parser.add_argument("--cycles", type=int, default=3, help="Number of training cycles.")
    parser.add_argument("--gp-kernel", choices=["Krbflog", "rbf_logrbf"], default="Krbflog", help="Covariance kernel for the GP prior.")
    parser.add_argument("--gp-kernel-mode", choices=["single", "mixture"], default="single", help="Use one GP prior kernel or a mixture of kernels for the training samples.")
    parser.add_argument("--gp-kernel-list", type=str, default=None, help="Comma-separated kernel list for mixture mode. Example: Krbflog,rbf_logrbf")
    parser.add_argument("--itd-points", type=int, default=11, help="Number of observed ITD points.")
    parser.add_argument("--itd-min", type=float, default=0.0, help="Minimum nu value for observed ITD grid.")
    parser.add_argument("--itd-max", type=float, default=12.0, help="Maximum nu value for observed ITD grid.")
    parser.add_argument("--noise-dim", type=int, default=None, help="Number of padding/noise dimensions appended to ITD samples. Default: same as itd-points.")
    parser.add_argument("--noise-type", choices=["diagonal", "correlated"], default="diagonal", help="Use independent Gaussian padding noise or correlated Gaussian padding noise.")
    parser.add_argument("--itd-part", choices=["real", "imag"], default="real", help="Use real or imaginary ITD component.")
    parser.add_argument("--normalize-data", action="store_true", help="Normalize PDF and observed ITD data using dataset mean/covariance whitening during training.")
    parser.add_argument("--x1-constraint", action="store_true", help="Activate the optional x=1 precision constraint in the PDF prior kernel.")
    return parser.parse_known_args()[0]


cli_args = parse_cli_args()
kernel_tag = cli_args.gp_kernel if cli_args.gp_kernel_mode == "single" else "mixture"
plot_noise_dim = cli_args.noise_dim if cli_args.noise_dim is not None else cli_args.itd_points
plot_x1_suffix = "_x1" if cli_args.x1_constraint else ""
plot_norm_suffix = "_norm" if cli_args.normalize_data else ""
PLOT_DIR = os.path.join(
    "plots_inn",
    f"{cli_args.itd_part}_{kernel_tag}_dat_{cli_args.itd_points:02d}_noise_{plot_noise_dim:02d}{plot_x1_suffix}{plot_norm_suffix}",
)
os.makedirs(PLOT_DIR, exist_ok=True)
GIF_PDF_YLIM = (-3.0, 14.0)
GIF_ITD_YLIM = (-1.5, 2.5) if cli_args.itd_part == "real" else (-0.3, 1.0)
ITD_TO_PDF_ITD_YLIM = (-0.4, 1.2) if cli_args.itd_part == "real" else (-0.3, 1.0)
PDF_DATA_STATS = None
ITD_DATA_STATS = None
PDF_DATA_MEAN = None
PDF_DATA_COV = None
ITD_DATA_MEAN = None
ITD_DATA_COV = None
FULL_NU_GRID = None
FULL_ITD_REAL_INTEGRATOR = None
FULL_ITD_IMAG_INTEGRATOR = None
ACTIVE_FULL_ITD_INTEGRATOR = None


def build_integrator_matrix(x_grid, nu_values, mode="real"):
    fe = FE2_Integrator(x_grid)
    matrix = np.zeros((len(nu_values), len(x_grid)))
    kernel_fn = np.cos if mode == "real" else np.sin
    for idx, nu in enumerate(nu_values):
        matrix[idx, :] = fe.set_up_integration(Kernel=lambda x, nu=nu: kernel_fn(nu * x))
    return matrix


def build_real_imag_integrators(x_grid, nu_values):
    return {
        "real": build_integrator_matrix(x_grid, nu_values, mode="real"),
        "imag": build_integrator_matrix(x_grid, nu_values, mode="imag"),
    }


def build_gp_prior_kernel(x_grid_tr, kernel_name="Krbflog"):
    if kernel_name == "Krbflog":
        return Krbflog(x_grid_tr, s=30.0, w=0.6) + 1e-4 * tr.eye(x_grid_tr.shape[0], dtype=tr.float64)
    if kernel_name == "rbf_logrbf":
        return rbf_logrbf(
            x_grid_tr,
            s1=6.0,
            w1=np.log(2.5),
            s2=50.0,
            w2=0.6,
            scale=1.0,
            sp=0.1,
        ) + 1e-4 * tr.eye(x_grid_tr.shape[0], dtype=tr.float64)
    raise ValueError(f"Unsupported GP prior kernel: {kernel_name}")


def get_selected_gp_kernels(args):
    if args.gp_kernel_mode == "single":
        return [args.gp_kernel]

    if args.gp_kernel_list is None:
        return ["Krbflog", "rbf_logrbf"]

    kernel_names = [name.strip() for name in args.gp_kernel_list.split(",") if name.strip()]
    valid_names = {"Krbflog", "rbf_logrbf"}
    invalid = [name for name in kernel_names if name not in valid_names]
    if invalid:
        raise ValueError(f"Unsupported kernels in --gp-kernel-list: {invalid}")
    if not kernel_names:
        raise ValueError("--gp-kernel-list did not contain any valid kernel names.")
    return kernel_names


def create_pdf_prior_samples(x_grid, n_samples, kernel_name="Krbflog"):
    x_grid_tr = tr.tensor(x_grid, dtype=tr.float64)
    kernel = build_gp_prior_kernel(x_grid_tr, kernel_name=kernel_name)
    mean = PDF_N(
        x_grid_tr,
        a=tr.tensor(-0.2, dtype=tr.float64),
        b=tr.tensor(3.0, dtype=tr.float64),
        N=tr.tensor(1.0, dtype=tr.float64),
    )

    iB0 = tr.tensor(FE2_Integrator(x_grid).set_up_integration(Kernel=lambda x: 1), dtype=tr.float64)
    lamb = 1e-3
    precision = tr.outer(iB0, iB0) / lamb + tr.linalg.inv(kernel)
    constrained_mean = tr.linalg.inv(precision) @ (iB0 / lamb + tr.linalg.inv(kernel) @ mean)

    # constraint the behavior of GP at x=1 in the kernel
    # lamb1 = 1e-6
    # precision[-1, -1] = tr.tensor(1 / lamb1, dtype=tr.float64)
    if cli_args.x1_constraint:
        lamb1 = 1e-5
        precision[-1, -1] = tr.tensor(1 / lamb1, dtype=tr.float64)
        constrained_mean = tr.linalg.inv(precision) @ (iB0 / lamb + tr.linalg.inv(kernel) @ mean)

    dist = tr.distributions.MultivariateNormal(constrained_mean, precision_matrix=precision)
    samples = dist.sample((n_samples,))

    return {
        "x_grid": x_grid_tr,
        "kernel": kernel,
        "mean": constrained_mean,
        "precision": precision,
        "samples": samples,
        "normalization_integrator": iB0,
        "sample_kernel_labels": [kernel_name] * n_samples,
    }


def create_pdf_prior_samples_mixture(x_grid, n_samples, kernel_names):
    samples_per_kernel = [n_samples // len(kernel_names)] * len(kernel_names)
    for idx in range(n_samples % len(kernel_names)):
        samples_per_kernel[idx] += 1

    sample_blocks = []
    sample_kernel_labels = []
    kernel_blocks = {}
    mean_blocks = {}
    precision_blocks = {}
    iB0 = None
    x_grid_tr = tr.tensor(x_grid, dtype=tr.float64)

    for kernel_name, n_block in zip(kernel_names, samples_per_kernel):
        if n_block == 0:
            continue
        prior_block = create_pdf_prior_samples(x_grid, n_block, kernel_name=kernel_name)
        sample_blocks.append(prior_block["samples"])
        sample_kernel_labels.extend(prior_block["sample_kernel_labels"])
        kernel_blocks[kernel_name] = prior_block["kernel"]
        mean_blocks[kernel_name] = prior_block["mean"]
        precision_blocks[kernel_name] = prior_block["precision"]
        if iB0 is None:
            iB0 = prior_block["normalization_integrator"]

    samples = tr.cat(sample_blocks, dim=0)
    shuffle_idx = tr.randperm(samples.shape[0])
    samples = samples[shuffle_idx]
    sample_kernel_labels = [sample_kernel_labels[idx] for idx in shuffle_idx.tolist()]

    return {
        "x_grid": x_grid_tr,
        "kernel": kernel_blocks,
        "mean": samples.mean(dim=0),
        "precision": precision_blocks,
        "samples": samples,
        "normalization_integrator": iB0,
        "kernel_names": kernel_names,
        "sample_kernel_labels": sample_kernel_labels,
    }


def create_parametric_pdf_samples(
    x_grid,
    n_samples,
    a_mean=-0.2,
    a_std=0.5,
    b_mean=3.0,
    b_std=0.5,
    N_mean=1.0,
):
    x_grid_tr = tr.tensor(x_grid, dtype=tr.float64)
    x_eval = tr.clamp(x_grid_tr, min=1e-6, max=1.0 - 1e-6)

    a_dist = tr.distributions.Normal(
        loc=tr.tensor(a_mean, dtype=tr.float64),
        scale=tr.tensor(a_std, dtype=tr.float64),
    )
    b_dist = tr.distributions.Normal(
        loc=tr.tensor(b_mean, dtype=tr.float64),
        scale=tr.tensor(b_std, dtype=tr.float64),
    )
    a_samples = a_dist.sample((n_samples,))
    b_samples = b_dist.sample((n_samples,))
    N_samples = tr.full((n_samples,), fill_value=N_mean, dtype=tr.float64)

    a_samples = tr.clamp(a_samples, min=-0.45, max=2.0)
    b_samples = tr.clamp(b_samples, min=0.1)
    N_samples = tr.clamp(N_samples, min=0.05)

    pdf_samples = PDF_N(
        x_eval.unsqueeze(0),
        a_samples.unsqueeze(1),
        b_samples.unsqueeze(1),
        N_samples.unsqueeze(1),
    )

    return {
        "x_grid": x_grid_tr,
        "mean": pdf_samples.mean(dim=0),
        "samples": pdf_samples,
        "params": {
            "a": a_samples,
            "b": b_samples,
            "N": N_samples,
        },
    }


def create_itd_samples(pdf_samples, real_integrator, imag_integrator):
    real_integrator_tr = tr.tensor(real_integrator, dtype=tr.float64)
    imag_integrator_tr = tr.tensor(imag_integrator, dtype=tr.float64)
    pdf_samples_t = pdf_samples.transpose(1, 0)
    return {
        "real": tr.matmul(real_integrator_tr, pdf_samples_t),
        "imag": tr.matmul(imag_integrator_tr, pdf_samples_t),
        "real_integrator": real_integrator_tr,
        "imag_integrator": imag_integrator_tr,
    }


def add_itd_padding_noise(itd_real_samples, n_full_points, noise_kernel, n_samples):
    observed_points = itd_real_samples.shape[0]
    gaussian = tr.distributions.MultivariateNormal(
        tr.zeros(n_full_points - observed_points, dtype=tr.float64),
        noise_kernel.to(tr.float64),
    )
    noise_samples = gaussian.sample((n_samples,)).transpose(1, 0)
    out_samples = tr.cat((itd_real_samples, noise_samples), dim=0).transpose(1, 0)
    return out_samples, noise_samples


def build_noise_kernel(noise_dim, base_kernel=None, default_scale=0.1):
    if noise_dim <= 0:
        return None
    if base_kernel is None:
        return default_scale * tr.eye(noise_dim, dtype=tr.float64)
    base_kernel = base_kernel.to(tr.float64)
    if base_kernel.shape[0] == noise_dim:
        return base_kernel
    diag_value = tr.diag(base_kernel).mean().item() if base_kernel.numel() > 0 else default_scale
    return diag_value * tr.eye(noise_dim, dtype=tr.float64)


def build_padding_noise_kernel(noise_dim, noise_type, nu_fake_grid=None):
    if noise_dim <= 0:
        return None
    if noise_type == "diagonal":
        return 0.1 * tr.eye(noise_dim, dtype=tr.float64)
    if nu_fake_grid is None or nu_fake_grid.shape[0] != noise_dim:
        raise ValueError("Correlated noise requires a nu_fake_grid with the same length as noise_dim.")
    return Krbflog(
        nu_fake_grid.to(dtype=tr.float64),
        s=0.2,
        w=0.2,
    ) + 1e-5 * tr.eye(noise_dim, dtype=tr.float64)


def build_aux_noise_grid(noise_dim, nu_start):
    if noise_dim <= 0:
        return tr.empty(0, dtype=tr.float64)
    return tr.linspace(
        float(nu_start),
        float(nu_start) + float(noise_dim),
        steps=noise_dim,
        dtype=tr.float64,
    )


def sample_padding_noise(n_samples, noise_dim, noise_kernel=None, device=None, dtype=tr.float64, nu_start=None):
    if noise_dim <= 0:
        return tr.empty(n_samples, 0, device=device, dtype=dtype)
    if noise_kernel is None or noise_kernel.shape[0] != noise_dim:
        if cli_args.noise_type == "correlated":
            if nu_start is None:
                raise ValueError("Correlated padding noise requires nu_start when building a new covariance.")
            noise_kernel = build_padding_noise_kernel(
                noise_dim,
                cli_args.noise_type,
                nu_fake_grid=build_aux_noise_grid(noise_dim, nu_start),
            )
        else:
            noise_kernel = build_padding_noise_kernel(noise_dim, cli_args.noise_type)
    gaussian = tr.distributions.MultivariateNormal(
        tr.zeros(noise_dim, device=device, dtype=dtype),
        covariance_matrix=noise_kernel.to(device=device, dtype=dtype),
    )
    return gaussian.sample((n_samples,))


def fit_gaussian_normalization_stats(data, name, jitter=1e-8):
    print(f"Computing {name} normalization stats...", flush=True)
    mean = data.mean(dim=0)
    centered = data - mean
    denom = max(data.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    eigvals, eigvecs = tr.linalg.eigh(cov)
    eigvals = eigvals.clamp_min(jitter)
    whitener = eigvecs @ tr.diag(eigvals.rsqrt()) @ eigvecs.T
    dewhitener = eigvecs @ tr.diag(eigvals.sqrt()) @ eigvecs.T
    return {
        "mean": mean,
        "cov": cov,
        "whitener": whitener,
        "dewhitener": dewhitener,
    }


def apply_normalization(data, stats):
    mean = stats["mean"].to(device=data.device, dtype=data.dtype)
    whitener = stats["whitener"].to(device=data.device, dtype=data.dtype)
    return (data - mean) @ whitener.T


def invert_normalization(data, stats):
    mean = stats["mean"].to(device=data.device, dtype=data.dtype)
    dewhitener = stats["dewhitener"].to(device=data.device, dtype=data.dtype)
    return data @ dewhitener.T + mean


def normalize_pdf_data(data):
    if not cli_args.normalize_data or PDF_DATA_STATS is None:
        return data
    return apply_normalization(data, PDF_DATA_STATS)


def denormalize_pdf_data(data):
    if not cli_args.normalize_data or PDF_DATA_STATS is None:
        return data
    return invert_normalization(data, PDF_DATA_STATS)


def normalize_itd_data(data):
    if not cli_args.normalize_data or ITD_DATA_STATS is None:
        return data
    return apply_normalization(data, ITD_DATA_STATS)


def denormalize_itd_data(data):
    if not cli_args.normalize_data or ITD_DATA_STATS is None:
        return data
    return invert_normalization(data, ITD_DATA_STATS)


def project_pdf_to_full_itd(pdf_samples):
    pdf_raw = denormalize_pdf_data(pdf_samples)
    integrator = ACTIVE_FULL_ITD_INTEGRATOR.to(device=pdf_raw.device, dtype=pdf_raw.dtype)
    return pdf_raw @ integrator.T


def save_parametric_dataset_snapshot(pdf_samples, itd_samples, x_grid, nu_values, tag, gp_pdf_samples=None, gp_itd_samples=None):
    pdf_cpu = denormalize_pdf_data(pdf_samples).detach().cpu()
    itd_obs_cpu = denormalize_itd_data(itd_samples).detach().cpu()
    itd_cpu = project_pdf_to_full_itd(pdf_samples).detach().cpu()

    mean_pdf = pdf_cpu.mean(dim=0)
    std_pdf = pdf_cpu.std(dim=0)
    mean_itd = itd_cpu.mean(dim=0)
    std_itd = itd_cpu.std(dim=0)
    mean_itd_obs = itd_obs_cpu.mean(dim=0)
    std_itd_obs = itd_obs_cpu.std(dim=0)

    x_axis = np.asarray(x_grid[:mean_pdf.shape[0]])
    nu_axis = FULL_NU_GRID[:mean_itd.shape[0]].detach().cpu().numpy()
    nu_obs_axis = nu_values[:mean_itd_obs.shape[0]].detach().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    axs[0].plot(x_axis, mean_pdf.numpy(), color="darkorange", label="Parametric PDF")
    axs[0].fill_between(
        x_axis,
        (mean_pdf - std_pdf).numpy(),
        (mean_pdf + std_pdf).numpy(),
        color="darkorange",
        alpha=0.3,
    )
    if gp_pdf_samples is not None:
        gp_pdf_cpu = gp_pdf_samples.detach().cpu()
        mean_pdf_gp = gp_pdf_cpu.mean(dim=0)
        std_pdf_gp = gp_pdf_cpu.std(dim=0)
        axs[0].plot(x_axis, mean_pdf_gp.numpy(), color="purple", label="GP prior")
        axs[0].fill_between(
            x_axis,
            (mean_pdf_gp - std_pdf_gp).numpy(),
            (mean_pdf_gp + std_pdf_gp).numpy(),
            color="purple",
            alpha=0.2,
        )
    axs[0].set_title("Parametric PDF Dataset")
    axs[0].set_xlabel(r"$x$")
    axs[0].set_ylabel(r"$f(x)$")
    if gp_pdf_samples is not None:
        set_pdf_axis_limits(axs[0], mean_pdf, std_pdf, mean_pdf_gp, std_pdf_gp)
    else:
        axs[0].set_ylim(-2.0, (mean_pdf + std_pdf).max().item() * 1.05)
    axs[0].legend()

    itd_label = "Real ITD" if cli_args.itd_part == "real" else "Imag ITD"
    axs[1].plot(nu_axis, mean_itd.numpy(), color="teal", label=f"Parametric {itd_label}")
    axs[1].fill_between(
        nu_axis,
        (mean_itd - std_itd).numpy(),
        (mean_itd + std_itd).numpy(),
        color="teal",
        alpha=0.3,
    )
    if gp_itd_samples is not None:
        gp_itd_cpu = project_pdf_to_full_itd(gp_pdf_samples).detach().cpu()
        mean_itd_gp = gp_itd_cpu.mean(dim=0)
        std_itd_gp = gp_itd_cpu.std(dim=0)
        axs[1].plot(nu_axis, mean_itd_gp.numpy(), color="purple", label=f"GP prior {itd_label}")
        axs[1].fill_between(
            nu_axis,
            (mean_itd_gp - std_itd_gp).numpy(),
            (mean_itd_gp + std_itd_gp).numpy(),
            color="purple",
            alpha=0.2,
        )
    axs[1].errorbar(
        nu_obs_axis,
        mean_itd_obs.numpy(),
        yerr=std_itd_obs.numpy(),
        fmt='o',
        color='navy',
        markersize=4,
        linewidth=1.0,
        capsize=3,
        label='Data',
    )
    axs[1].set_title(f"Parametric {itd_label} Dataset")
    axs[1].set_xlabel(r"$\nu$")
    axs[1].set_ylabel(r"$\mathcal{M}(\nu)$")
    if cli_args.itd_part == "real":
        axs[1].set_ylim(-0.4, 1.2)
    else:
        axs[1].set_ylim(-0.3, 1.0)
    axs[1].legend()

    plt.tight_layout()
    filename = os.path.join(PLOT_DIR, f"parametric_dataset_{tag}.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved parametric dataset plot: {filename}", flush=True)


def set_pdf_axis_limits(ax, mean_a, std_a, mean_b, std_b):
    pdf_min = tr.min(tr.cat((mean_a - std_a, mean_b - std_b))).item()
    pdf_max = tr.max(tr.cat((mean_a + std_a, mean_b + std_b))).item()
    margin = 0.05 * max(pdf_max - pdf_min, 1e-8)
    ax.set_ylim(-2.0, pdf_max + margin)


def create_gif_and_cleanup(pattern, gif_filename, fps=2):
    image_files = sorted(glob.glob(pattern))
    if not image_files:
        print(f"No images matched pattern: {pattern}", flush=True)
        return

    gif_path = os.path.join(PLOT_DIR, gif_filename)
    created = False

    try:
        import imageio.v2 as imageio

        frames = [imageio.imread(image_file) for image_file in image_files]
        max_height = max(frame.shape[0] for frame in frames)
        max_width = max(frame.shape[1] for frame in frames)
        normalized_frames = []
        for frame in frames:
            if frame.ndim == 2:
                canvas = np.full((max_height, max_width), 255, dtype=frame.dtype)
                canvas[:frame.shape[0], :frame.shape[1]] = frame
            else:
                canvas = np.full((max_height, max_width, frame.shape[2]), 255, dtype=frame.dtype)
                canvas[:frame.shape[0], :frame.shape[1], :] = frame
            normalized_frames.append(canvas)
        duration = max(0.1, 1.0 / fps)
        imageio.mimsave(gif_path, normalized_frames, duration=duration, loop=0)
        created = True
    except Exception as exc:
        print(f"imageio GIF creation unavailable ({exc}); trying ImageMagick.", flush=True)
        try:
            delay = max(1, int(round(100 / fps)))
            subprocess.run(
                ["convert", "-delay", str(delay), "-loop", "0", *image_files, gif_path],
                check=True,
            )
            created = True
        except Exception as convert_exc:
            print(f"GIF creation failed for {gif_filename}: {convert_exc}", flush=True)

    if created:
        print(f"Created GIF: {gif_path}", flush=True)
        for image_file in image_files:
            try:
                os.remove(image_file)
            except OSError as exc:
                print(f"Could not remove {image_file}: {exc}", flush=True)


def create_gif_from_gifs(pattern, gif_filename):
    gif_files = sorted(glob.glob(pattern))
    if not gif_files:
        print(f"No GIFs matched pattern: {pattern}", flush=True)
        return

    gif_path = os.path.join(PLOT_DIR, gif_filename)
    try:
        subprocess.run(
            ["convert", *gif_files, gif_path],
            check=True,
        )
        print(f"Created combined GIF: {gif_path}", flush=True)
        for gif_file in gif_files:
            try:
                os.remove(gif_file)
            except OSError as remove_exc:
                print(f"Could not remove {gif_file}: {remove_exc}", flush=True)
    except Exception as exc:
        print(f"Combined GIF creation failed for {gif_filename}: {exc}", flush=True)


def nnpdfdata(x_grid):
    pdfminus = np.loadtxt('NNPDF/NNPDF40_nnlo_as_01180_1000_pdf_minus.dat',dtype=np.float64)
    pdfplus = np.loadtxt('NNPDF/NNPDF40_nnlo_as_01180_1000_pdf_plus.dat',dtype=np.float64)
    covmin=np.cov(pdfminus[:,1:])
    covplus=np.cov(pdfplus[:,1:])
    covmindiag=np.diag(covmin)
    covplusdiag=np.diag(covplus)
    meanplus=pdfplus[:,1:].mean(axis=1)
    meanminus=pdfminus[:,1:].mean(axis=1)

    Nx=x_grid.shape[0]
    nn = np.linspace(0,100,128)
    x_grid_orig = pdfplus[:,0]
    Nxx=x_grid_orig.shape[0]
    fe_orig=FE2_Integrator(x_grid_orig)
    #ITD="Re"
    pdf_im=meanplus
    cov_im=covplus
    pdf_interp_im=np.interp(x_grid,pdfplus[:,0],meanplus)
    cov_interp_im=np.interp(x_grid,pdfplus[:,0],np.diag(covplus))
    iB_im=np.zeros((nn.shape[0],x_grid_orig.shape[0]))
    for k in range(nn.shape[0]):
        iB_im[k,:] = fe_orig.set_up_integration(Kernel= lambda x : np.sin(nn[k]*x))
    pdf_re=meanminus
    cov_re=covmin
    pdf_interp_re=np.interp(x_grid,pdfminus[:,0],meanminus)
    cov_interp_re=np.interp(x_grid,pdfminus[:,0],np.diag(covmin))
    iB_re=np.zeros((nn.shape[0],x_grid_orig.shape[0]))
    for k in range(nn.shape[0]):
        iB_re[k,:] = fe_orig.set_up_integration(Kernel= lambda x : np.cos(nn[k]*x))

    return pdf_re,cov_re,pdf_interp_re,cov_interp_re,iB_re,pdf_im,cov_im,pdf_interp_im,cov_interp_im,iB_im,x_grid_orig,x_grid


x_grid_log=generategrid(129,"log_lin")
x_grid_lin=generategrid(128,"linh")
x_grid_model = x_grid_lin

# NNPDF loading is kept available through `nnpdfdata(...)`, but disabled here
# so the synthetic PDF/ITD pipeline can run without external NNPDF inputs.
# print("Loading NNPDF data on log grid...", flush=True)
# nnpdf_all_log=nnpdfdata(x_grid_log)
# print("Loading NNPDF data on linear grid...", flush=True)
# nnpdf_all_lin=nnpdfdata(x_grid_lin)


SAMPLES=1000000
x_grid_model_tr=tr.tensor(x_grid_model,dtype=tr.float64)
print(f"Building PDF prior and drawing {SAMPLES} samples...", flush=True)
selected_gp_kernels = get_selected_gp_kernels(cli_args)
print(f"Using GP prior kernel mode: {cli_args.gp_kernel_mode} | kernels: {selected_gp_kernels}", flush=True)
if cli_args.gp_kernel_mode == "single":
    pdf_prior = create_pdf_prior_samples(x_grid_model, SAMPLES, kernel_name=selected_gp_kernels[0])
else:
    pdf_prior = create_pdf_prior_samples_mixture(x_grid_model, SAMPLES, kernel_names=selected_gp_kernels)
Kernel = pdf_prior["kernel"]
newmean = pdf_prior["mean"]
newkernel = pdf_prior["precision"]
samples = pdf_prior["samples"]
iB0 = pdf_prior["normalization_integrator"]
sample_kernel_labels = pdf_prior.get("sample_kernel_labels", [selected_gp_kernels[0]] * samples.shape[0])
print(newmean.shape)

nu_grid = tr.linspace(cli_args.itd_min, cli_args.itd_max, steps=cli_args.itd_points)
Nnu=nu_grid.shape[0]
Nx=x_grid_model.shape[0]
FULL_NU_GRID = tr.linspace(0.0, 100.0, steps=128, dtype=tr.float64)

iB_dict = build_real_imag_integrators(x_grid_model, nu_grid)
print("Built real/imag ITD integrators.", flush=True)
itd_real_integrator = iB_dict["real"]
itd_imag_integrator = iB_dict["imag"]
active_itd_integrator = itd_real_integrator if cli_args.itd_part == "real" else itd_imag_integrator
full_iB_dict = build_real_imag_integrators(x_grid_model, FULL_NU_GRID)
FULL_ITD_REAL_INTEGRATOR = tr.tensor(full_iB_dict["real"], dtype=tr.float64)
FULL_ITD_IMAG_INTEGRATOR = tr.tensor(full_iB_dict["imag"], dtype=tr.float64)
ACTIVE_FULL_ITD_INTEGRATOR = FULL_ITD_REAL_INTEGRATOR if cli_args.itd_part == "real" else FULL_ITD_IMAG_INTEGRATOR

print("Building parametric PDF_N dataset and its ITD projection...", flush=True)
parametric_dataset = create_parametric_pdf_samples(x_grid_model, 1000)
parametric_itd_dict = create_itd_samples(parametric_dataset["samples"], itd_real_integrator, itd_imag_integrator)
parametric_itd_real = parametric_itd_dict["real"].transpose(1, 0)
parametric_itd_imag = parametric_itd_dict["imag"].transpose(1, 0)
parametric_pdf_eval_raw = parametric_dataset["samples"].to(device=device, dtype=tr.float64)
parametric_itd_eval_raw = (
    parametric_itd_real if cli_args.itd_part == "real" else parametric_itd_imag
).to(device=device, dtype=tr.float64)

#real and imaginary part of the Fourier transform of the mean and the samples
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
print("Projecting PDF samples to ITD real/imag parts...", flush=True)
itd_samples = create_itd_samples(samples, itd_real_integrator, itd_imag_integrator)
M_Re_mean = tr.matmul(itd_samples["real_integrator"],newmean)
M_Im_mean = tr.matmul(itd_samples["imag_integrator"],newmean)
M_Re_sample = itd_samples["real"]
M_Im_sample = itd_samples["imag"]
selected_itd_samples = M_Re_sample if cli_args.itd_part == "real" else M_Im_sample
save_parametric_dataset_snapshot(
    parametric_dataset["samples"],
    parametric_itd_eval_raw.detach().cpu(),
    x_grid_model,
    nu_grid,
    "initial",
    gp_pdf_samples=samples,
    gp_itd_samples=selected_itd_samples.transpose(1, 0),
)


### Add correlated noise yo ITD

print("Adding correlated padding noise to ITD samples...", flush=True)
noise_dim = cli_args.noise_dim if cli_args.noise_dim is not None else Nnu
nu_fake_grid = build_aux_noise_grid(noise_dim, nu_grid[-1].item())
if noise_dim > 0:
    # Previous prescription kept for reference:
    # if "kernoise" not in globals():
    #     print("kernoise not found; using diagonal Gaussian padding noise.", flush=True)
    #     kernoise = 0.1 * tr.eye(noise_dim, dtype=tr.float64)
    # else:
    #     kernoise = build_noise_kernel(noise_dim, kernoise)
    # out_samples, noise_samples = add_itd_padding_noise(selected_itd_samples, Nnu + noise_dim, kernoise, SAMPLES)
    kernoise = build_padding_noise_kernel(
        noise_dim,
        cli_args.noise_type,
        nu_fake_grid=nu_fake_grid,
    )
    print(
        f"Padding noise mode: {cli_args.noise_type} | dim={noise_dim}",
        flush=True,
    )
    out_samples, noise_samples = add_itd_padding_noise(selected_itd_samples, Nnu + noise_dim, kernoise, SAMPLES)
else:
    noise_samples = tr.empty(0, SAMPLES, dtype=tr.float64)
    out_samples = selected_itd_samples.transpose(1, 0)
print(out_samples.shape)
nu_grid_full = tr.cat((nu_grid, nu_fake_grid))


observed_itd_samples = selected_itd_samples.transpose(1, 0)
PDF_DATA_STATS = fit_gaussian_normalization_stats(samples, "PDF")
ITD_DATA_STATS = fit_gaussian_normalization_stats(observed_itd_samples, "ITD")
PDF_DATA_MEAN = PDF_DATA_STATS["mean"]
PDF_DATA_COV = PDF_DATA_STATS["cov"]
ITD_DATA_MEAN = ITD_DATA_STATS["mean"]
ITD_DATA_COV = ITD_DATA_STATS["cov"]

if cli_args.normalize_data:
    print("Applying PDF/ITD whitening normalization for training and evaluation.", flush=True)
    samples_train = normalize_pdf_data(samples)
    observed_itd_train = normalize_itd_data(observed_itd_samples)
    out_samples_train = out_samples.clone()
    out_samples_train[:, :Nnu] = observed_itd_train
    parametric_pdf_eval = normalize_pdf_data(parametric_pdf_eval_raw)
    parametric_itd_eval = normalize_itd_data(parametric_itd_eval_raw)
else:
    print("Data normalization disabled; using raw PDF and ITD values.", flush=True)
    samples_train = samples
    out_samples_train = out_samples
    parametric_pdf_eval = parametric_pdf_eval_raw
    parametric_itd_eval = parametric_itd_eval_raw





# --- model 1--- #
class CouplingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask,device):
        super().__init__()
        self.device=device
        self.s_fc1 = nn.Linear(input_dim, hid_dim)
        self.s_fc2 = nn.Linear(hid_dim, hid_dim)
        self.s_fc3 = nn.Linear(hid_dim, output_dim)
        self.t_fc1 = nn.Linear(input_dim, hid_dim)
        self.t_fc2 = nn.Linear(hid_dim, hid_dim)
        self.t_fc3 = nn.Linear(hid_dim, output_dim)
        self.mask = mask.to(device)
        nn.init.xavier_uniform_(self.s_fc1.weight)
        nn.init.xavier_uniform_(self.s_fc2.weight)
        nn.init.xavier_uniform_(self.s_fc3.weight)
        nn.init.xavier_uniform_(self.t_fc1.weight)
        nn.init.xavier_uniform_(self.t_fc2.weight)
        nn.init.xavier_uniform_(self.t_fc3.weight)

    
    def forward(self, x):
        mask = self.mask.to(self.device).to(tr.float64)
        x = x.to(self.device).to(tr.float64)

        x_m = mask * x
        s_out = tr.tanh(self.s_fc3(Func.relu(self.s_fc2(Func.relu(self.s_fc1(x_m))))))
        t_out = self.t_fc3(Func.relu(self.t_fc2(Func.relu(self.t_fc1(x_m)))))

        y = x_m + (1 - mask) * (x * tr.exp(s_out) + t_out)
        log_det_jacobian = ((1 - mask) * s_out).sum(dim=1)
        return y, log_det_jacobian

    # we define the global backward mapping for our neural network

    def backward(self, y):
        mask = self.mask.to(self.device).to(tr.float64)
        y = y.to(self.device).to(tr.float64)

        y_m = mask * y
        s_out = tr.tanh(self.s_fc3(Func.relu(self.s_fc2(Func.relu(self.s_fc1(y_m))))))
        t_out = self.t_fc3(Func.relu(self.t_fc2(Func.relu(self.t_fc1(y_m)))))

        x = y_m + (1 - mask) * (y - t_out) * tr.exp(-s_out)
        return x

def pairwise_sq_dists(x, y):
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).T
    return x_norm + y_norm - 2.0 * x @ y.T

def median_heuristic(x, y, eps=1e-8):
    with tr.no_grad():
        d = pairwise_sq_dists(x, y)
        d = d[d > 0]
        if d.numel() == 0:
            return tr.tensor(1.0, device=x.device, dtype=x.dtype)
        return tr.median(d).clamp_min(eps)

def MMD_rbf(x, y, bandwidths=None):
    dxx = pairwise_sq_dists(x, x)
    dyy = pairwise_sq_dists(y, y)
    dxy = pairwise_sq_dists(x, y)

    if bandwidths is None:
        sigma2 = median_heuristic(x, y)
        bandwidths = [sigma2 / 4, sigma2 / 2, sigma2, 2 * sigma2, 4 * sigma2]

    XX = 0.0
    YY = 0.0
    XY = 0.0

    for bw in bandwidths:
        bw = bw.clamp_min(1e-8)
        XX = XX + tr.exp(-dxx / (2.0 * bw))
        YY = YY + tr.exp(-dyy / (2.0 * bw))
        XY = XY + tr.exp(-dxy / (2.0 * bw))

    return XX.mean() + YY.mean() - 2.0 * XY.mean()


def pairwise_sq_dists(x, y):
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).T
    return x_norm + y_norm - 2.0 * x @ y.T

def median_scale(x, y, eps=1e-8):
    with tr.no_grad():
        d = pairwise_sq_dists(x, y)
        d = d[d > 0]
        if d.numel() == 0:
            return tr.tensor(1.0, device=x.device, dtype=x.dtype)
        return tr.median(d).clamp_min(eps)

def MMD_multiquadratic(x, y, scales=None, alpha=2.0):
    # Biased estimator used in training:
    # MMD^2(X,Y) =
    #   1/n^2 sum_{i, j} k(x_i, x_j)
    # + 1/m^2 sum_{i, j} k(y_i, y_j)
    # - 2/(nm) sum_{i, j} k(x_i, y_j)
    #
    # Unbiased estimator kept for reference:
    # MMD^2(X,Y) =
    #   1/(n(n-1)) sum_{i != j} k(x_i, x_j)
    # + 1/(m(m-1)) sum_{i != j} k(y_i, y_j)
    # - 2/(nm)      sum_{i, j}   k(x_i, y_j)
    #
    # with multi-scale multiquadratic kernel
    # k(x, y) = sum_c ( c / (c + ||x-y||^2) )^alpha
    dxx = pairwise_sq_dists(x, x)
    dyy = pairwise_sq_dists(y, y)
    dxy = pairwise_sq_dists(x, y)

    if scales is None:
        s = median_scale(x, y)
        scales = [0.25 * s, 0.5 * s, s, 2.0 * s, 4.0 * s]

    XX = 0.0
    YY = 0.0
    XY = 0.0

    for c in scales:
        c = c.clamp_min(1e-8)
        XX = XX + (c / (c + dxx))**alpha
        YY = YY + (c / (c + dyy))**alpha
        XY = XY + (c / (c + dxy))**alpha

    return XX.mean() + YY.mean() - 2.0 * XY.mean()


class RealNVP(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask, device, n_layers=6):
        super().__init__()
        assert n_layers >= 2, "num of coupling layers should be greater or equal to 2"

        layers = [CouplingLayer(input_dim, output_dim, hid_dim, mask, device)]
        for _ in range(n_layers - 2):
            mask = 1 - mask
            layers.append(CouplingLayer(input_dim, output_dim, hid_dim, mask, device))
        layers.append(CouplingLayer(input_dim, output_dim, hid_dim, 1 - mask, device))

        self.layer_list = nn.ModuleList(layers)
        self.device = device

    def forward(self, x):
        ldj_sum = 0
        for layer in self.layer_list:
            x, ldj = layer(x)
            ldj_sum += ldj
        return x, ldj_sum

    def backward(self, z):
        for layer in reversed(self.layer_list):
            z = layer.backward(z)
        return z


def mask1(n,rand=3):
#n = 1000  # number of elements of the mask
    half = n // 2

    tensor = tr.cat((tr.ones(half), tr.zeros(n - half)))
    if rand==1:
        index = tr.randperm(tensor.size(0))
        shuffled_tensor = tensor[index]
        return shuffled_tensor
    elif rand==2:
        return tensor
    else:
        tensor = tr.zeros(n)
        tensor[::2] = 1
        return tensor


# --- model 2--- #

class Subnet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid_dim),nn.LeakyReLU(0.2),nn.Linear(hid_dim, hid_dim),nn.LeakyReLU(0.2),nn.Linear(hid_dim, out_dim),)

    def forward(self, x):
        return self.net(x)

class ReversibleBlock(nn.Module):
    def __init__(self, dim, hid_dim, clamp=1.0):
        super().__init__()
        self.d1 = dim // 2
        self.d2 = dim - self.d1
        self.clamp = clamp

        self.s2 = Subnet(self.d2, hid_dim, self.d1)
        self.t2 = Subnet(self.d2, hid_dim, self.d1)
        self.s1 = Subnet(self.d1, hid_dim, self.d2)
        self.t1 = Subnet(self.d1, hid_dim, self.d2)

    def _scale(self, s):
        return self.clamp * tr.tanh(s / self.clamp)

    def forward(self, u):
        u1, u2 = u[:, :self.d1], u[:, self.d1:]

        s2 = self._scale(self.s2(u2))
        t2 = self.t2(u2)
        v1 = u1 * tr.exp(s2) + t2

        s1 = self._scale(self.s1(v1))
        t1 = self.t1(v1)
        v2 = u2 * tr.exp(s1) + t1

        logdet = s2.sum(dim=1) + s1.sum(dim=1)
        return tr.cat([v1, v2], dim=1), logdet

    def backward(self, v):
        u, _ = self.backward_with_logdet(v)
        return u

    def backward_with_logdet(self, v):
        v1, v2 = v[:, :self.d1], v[:, self.d1:]

        s1 = self._scale(self.s1(v1))
        t1 = self.t1(v1)
        u2 = (v2 - t1) * tr.exp(-s1)

        s2 = self._scale(self.s2(u2))
        t2 = self.t2(u2)
        u1 = (v1 - t2) * tr.exp(-s2)

        logdet = -(s1.sum(dim=1) + s2.sum(dim=1))
        return tr.cat([u1, u2], dim=1), logdet

class Permutation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        perm = tr.randperm(dim)
        inv_perm = tr.argsort(perm)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, x):
        return x[:, self.perm]

    def backward(self, x):
        return x[:, self.inv_perm]

class INNforPDF_ITD(nn.Module):
    def __init__(self, pdf_dim, itd_dim, z_dim, hid_dim, n_blocks=6, clamp=1.0):
        super().__init__()
        self.pdf_dim = pdf_dim
        self.itd_dim = itd_dim
        self.z_dim = z_dim

        self.yz_dim = itd_dim + z_dim
        self.common_dim = max(pdf_dim, self.yz_dim)

        self.pdf_pad_dim = self.common_dim - pdf_dim
        self.yz_pad_dim = self.common_dim - self.yz_dim

        self.blocks = nn.ModuleList([
            ReversibleBlock(self.common_dim, hid_dim, clamp=clamp)
            for _ in range(n_blocks)
        ])
        self.perms = nn.ModuleList([
            Permutation(self.common_dim)
            for _ in range(n_blocks - 1)
        ])

    def pad_pdf(self, x):
        if self.pdf_pad_dim == 0:
            return x
        pad = tr.zeros(x.shape[0], self.pdf_pad_dim, device=x.device, dtype=x.dtype)
        return tr.cat([x, pad], dim=1)

    def pad_yz(self, y, z, pad_noise: Optional[tr.Tensor] = None):
        h = tr.cat([y, z], dim=1)
        if self.yz_pad_dim == 0:
            return h
        if pad_noise is None:
            pad_noise = tr.zeros(h.shape[0], self.yz_pad_dim, device=h.device, dtype=h.dtype)
        return tr.cat([h, pad_noise], dim=1)

    def split_output(self, h):
        itd = h[:, :self.itd_dim]
        z = h[:, self.itd_dim:self.itd_dim + self.z_dim]
        pad = h[:, self.itd_dim + self.z_dim:]
        return itd, z, pad

    def forward(self, pdf):
        itd, z, pad, logdet = self.forward_with_logdet(pdf)
        return itd, z, pad, logdet

    def forward_with_logdet(self, pdf):
        h = self.pad_pdf(pdf)
        logdet = tr.zeros(pdf.shape[0], device=pdf.device, dtype=pdf.dtype)
        for i, block in enumerate(self.blocks):
            h, ld = block(h)
            logdet = logdet + ld
            if i < len(self.perms):
                h = self.perms[i](h)
        itd, z, pad = self.split_output(h)
        return itd, z, pad, logdet

    def inverse(self, itd, z, pad_noise=None):
        pdf, _ = self.inverse_with_logdet(itd, z, pad_noise)
        return pdf

    def inverse_with_logdet(self, itd, z, pad_noise: Optional[tr.Tensor] = None):
        h = self.pad_yz(itd, z, pad_noise)
        logdet = tr.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        for i in reversed(range(len(self.blocks))):
            if i < len(self.perms):
                h = self.perms[i].backward(h)
            h, ld = self.blocks[i].backward_with_logdet(h)
            logdet = logdet + ld
        return h[:, :self.pdf_dim], logdet



def pairwise_sq_dists(x, y):
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).T
    return x_norm + y_norm - 2.0 * x @ y.T

def MMD_IMQ(x, y, scales=(0.2, 0.5, 1.0, 2.0, 5.0)):
    dxx = pairwise_sq_dists(x, x)
    dyy = pairwise_sq_dists(y, y)
    dxy = pairwise_sq_dists(x, y)

    XX = 0.0
    YY = 0.0
    XY = 0.0

    for h in scales:
        c = h * h
        XX = XX + c / (c + dxx)
        YY = YY + c / (c + dyy)
        XY = XY + c / (c + dxy)

    return XX.mean() + YY.mean() - 2.0 * XY.mean()


x_grid_model_tr = tr.tensor(x_grid_model, device=device, dtype=tr.float64)

def first_derivative(y, x):
    dx = x[1:] - x[:-1]
    return (y[:, 1:] - y[:, :-1]) / dx.unsqueeze(0)

def second_derivative(y, x):
    dy = first_derivative(y, x)
    dx_mid = x[2:] - x[:-2]
    return 2.0 * (dy[:, 1:] - dy[:, :-1]) / dx_mid.unsqueeze(0)

def smoothness_loss(y_pred, y_true, x):
    dy_pred = first_derivative(y_pred, x)
    dy_true = first_derivative(y_true, x)
    return Func.l1_loss(dy_pred, dy_true)

def curvature_loss(y_pred, y_true, x):
    d2y_pred = second_derivative(y_pred, x)
    d2y_true = second_derivative(y_true, x)
    return Func.l1_loss(d2y_pred, d2y_true)


def get_mmd_loss_weights(cycle):
    # Alternative schedule: keep MMD off in the first cycle, then reintroduce it gradually.
    if cycle == 0:
        return 0.0, 0.0
    if cycle == 1:
        return 2.0, 2.0
    if cycle == 2:
        return 5.0, 5.0
    return 10.0, 10.0


def get_smoothness_weight(cycle, epoch, epochs_per_cycle):
    # Keep smoothness strong at the beginning, then reintroduce a small amount
    # in the last two epochs of every cycle to preserve sample regularity.
    if cycle == 0:
        return 1.0
    if epoch >= max(epochs_per_cycle - 2, 0):
        return 0.2
    return 0.0


def show_training_snapshot(pdf_pred, pdf_true, itd_pred, itd_true, x_grid, nu_values, cycle, epoch):
    pdf_points = min(len(x_grid), pdf_pred.shape[1], pdf_true.shape[1])
    itd_points = min(len(nu_values), itd_pred.shape[1], itd_true.shape[1])

    pdf_pred_cpu = denormalize_pdf_data(pdf_pred[:, :pdf_points]).detach().cpu()
    pdf_true_cpu = denormalize_pdf_data(pdf_true[:, :pdf_points]).detach().cpu()
    itd_pred_cpu = project_pdf_to_full_itd(pdf_pred[:, :pdf_points]).detach().cpu()
    itd_true_cpu = project_pdf_to_full_itd(pdf_true[:, :pdf_points]).detach().cpu()
    itd_true_obs_cpu = denormalize_itd_data(itd_true[:, :itd_points]).detach().cpu()

    mean_pdf_pred = pdf_pred_cpu.mean(dim=0)
    mean_pdf_true = pdf_true_cpu.mean(dim=0)
    std_pdf_pred = pdf_pred_cpu.std(dim=0)
    std_pdf_true = pdf_true_cpu.std(dim=0)

    mean_itd_pred = itd_pred_cpu.mean(dim=0)
    mean_itd_true = itd_true_cpu.mean(dim=0)
    std_itd_pred = itd_pred_cpu.std(dim=0)
    std_itd_true = itd_true_cpu.std(dim=0)
    mean_itd_true_obs = itd_true_obs_cpu.mean(dim=0)
    std_itd_true_obs = itd_true_obs_cpu.std(dim=0)
    safe_std_pdf_true = std_pdf_true.clamp_min(1e-8)
    mean_pdf_true_norm = (mean_pdf_true - mean_pdf_true) / safe_std_pdf_true
    std_pdf_true_norm = std_pdf_true / safe_std_pdf_true
    mean_pdf_pred_norm = (mean_pdf_pred - mean_pdf_true) / safe_std_pdf_true
    std_pdf_pred_norm = std_pdf_pred / safe_std_pdf_true

    x_axis = np.asarray(x_grid[:pdf_points])
    nu_axis = FULL_NU_GRID[:mean_itd_pred.shape[0]].detach().cpu().numpy()
    nu_obs_axis = nu_values[:itd_points].detach().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    axs[0].plot(x_axis, mean_pdf_pred.numpy(), color='orange', label='INN')
    axs[0].fill_between(
        x_axis,
        (mean_pdf_pred - std_pdf_pred).numpy(),
        (mean_pdf_pred + std_pdf_pred).numpy(),
        color='orange',
        alpha=0.3,
    )
    axs[0].plot(x_axis, mean_pdf_true.numpy(), color='skyblue', label='True')
    axs[0].fill_between(
        x_axis,
        (mean_pdf_true - std_pdf_true).numpy(),
        (mean_pdf_true + std_pdf_true).numpy(),
        color='skyblue',
        alpha=0.3,
    )
    axs[0].set_title(f'PDF | cycle={cycle+1} epoch={epoch}')
    axs[0].set_xlabel(r'$x$')
    axs[0].set_ylabel(r'$f(x)$')
    axs[0].set_ylim(*GIF_PDF_YLIM)
    axs[0].legend()

    axs[1].plot(nu_axis, mean_itd_pred.numpy(), color='orange', label='INN')
    axs[1].fill_between(
        nu_axis,
        (mean_itd_pred - std_itd_pred).numpy(),
        (mean_itd_pred + std_itd_pred).numpy(),
        color='orange',
        alpha=0.3,
    )
    axs[1].plot(nu_axis, mean_itd_true.numpy(), color='skyblue', label='True')
    axs[1].fill_between(
        nu_axis,
        (mean_itd_true - std_itd_true).numpy(),
        (mean_itd_true + std_itd_true).numpy(),
        color='skyblue',
        alpha=0.3,
    )
    axs[1].errorbar(
        nu_obs_axis,
        mean_itd_true_obs.numpy(),
        yerr=std_itd_true_obs.numpy(),
        fmt='o',
        color='navy',
        markersize=4,
        linewidth=1.0,
        capsize=3,
        label='Data',
    )
    axs[1].set_title('ITD')
    axs[1].set_xlabel(r'$\nu$')
    axs[1].set_ylabel(r'$\mathcal{M}(\nu)$')
    axs[1].set_ylim(*GIF_ITD_YLIM)
    axs[1].legend()

    axs[2].plot(x_axis, mean_pdf_pred_norm.numpy(), color='orange', label='INN')
    axs[2].fill_between(
        x_axis,
        (mean_pdf_pred_norm - std_pdf_pred_norm).numpy(),
        (mean_pdf_pred_norm + std_pdf_pred_norm).numpy(),
        color='orange',
        alpha=0.3,
    )
    axs[2].plot(x_axis, mean_pdf_true_norm.numpy(), color='skyblue', label='Prior mean')
    axs[2].fill_between(
        x_axis,
        (mean_pdf_true_norm - std_pdf_true_norm).numpy(),
        (mean_pdf_true_norm + std_pdf_true_norm).numpy(),
        color='skyblue',
        alpha=0.3,
    )
    axs[2].axhline(0.0, color='k', linewidth=1.0, linestyle='--')
    axs[2].set_title('PDF vs prior')
    axs[2].set_xlabel(r'$x$')
    axs[2].set_ylabel(r'$(f-\mu_{\mathrm{prior}})/\sigma_{\mathrm{prior}}$')
    axs[2].set_ylim(-4.0, 4.0)
    axs[2].legend()

    plt.tight_layout()
    filename = os.path.join(PLOT_DIR, f"snapshot_cycle_{cycle+1:02d}_epoch_{epoch:05d}.png")
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved snapshot: {filename}", flush=True)


def show_itd_to_pdf_uncertainty_snapshot(itd_input, itd_from_pdf, pdf_from_itd, pdf_true, x_grid, nu_values, cycle, epoch):
    pdf_points = min(len(x_grid), pdf_from_itd.shape[1], pdf_true.shape[1])
    itd_points = min(len(nu_values), itd_input.shape[1])

    itd_input_cpu = denormalize_itd_data(itd_input[:, :itd_points]).detach().cpu()
    itd_from_pdf_cpu = project_pdf_to_full_itd(pdf_from_itd[:, :pdf_points]).detach().cpu()
    itd_true_full_cpu = project_pdf_to_full_itd(pdf_true[:, :pdf_points]).detach().cpu()
    pdf_from_itd_cpu = denormalize_pdf_data(pdf_from_itd[:, :pdf_points]).detach().cpu()
    pdf_true_cpu = denormalize_pdf_data(pdf_true[:, :pdf_points]).detach().cpu()

    mean_itd = itd_input_cpu.mean(dim=0)
    std_itd = itd_input_cpu.std(dim=0)
    mean_itd_from_pdf = itd_from_pdf_cpu.mean(dim=0)
    std_itd_from_pdf = itd_from_pdf_cpu.std(dim=0)
    mean_itd_true_full = itd_true_full_cpu.mean(dim=0)
    std_itd_true_full = itd_true_full_cpu.std(dim=0)
    mean_pdf = pdf_from_itd_cpu.mean(dim=0)
    std_pdf = pdf_from_itd_cpu.std(dim=0)
    mean_pdf_true = pdf_true_cpu.mean(dim=0)
    std_pdf_true = pdf_true_cpu.std(dim=0)
    safe_std_pdf_true = std_pdf_true.clamp_min(1e-8)
    mean_pdf_true_norm = (mean_pdf_true - mean_pdf_true) / safe_std_pdf_true
    std_pdf_true_norm = std_pdf_true / safe_std_pdf_true
    mean_pdf_norm = (mean_pdf - mean_pdf_true) / safe_std_pdf_true
    std_pdf_norm = std_pdf / safe_std_pdf_true

    x_axis = np.asarray(x_grid[:pdf_points])
    nu_axis = FULL_NU_GRID[:mean_itd_from_pdf.shape[0]].detach().cpu().numpy()
    nu_obs_axis = nu_values[:itd_points].detach().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    axs[0].plot(nu_axis, mean_itd_true_full.numpy(), color='teal', label='ITD input')
    axs[0].fill_between(
        nu_axis,
        (mean_itd_true_full - std_itd_true_full).numpy(),
        (mean_itd_true_full + std_itd_true_full).numpy(),
        color='teal',
        alpha=0.3,
    )
    axs[0].errorbar(
        nu_obs_axis,
        mean_itd.numpy(),
        yerr=std_itd.numpy(),
        fmt='o',
        color='navy',
        markersize=4,
        linewidth=1.0,
        capsize=3,
        label='Data',
    )
    axs[0].plot(nu_axis, mean_itd_from_pdf.numpy(), color='orange', label='Forward(INN inverse)')
    axs[0].fill_between(
        nu_axis,
        (mean_itd_from_pdf - std_itd_from_pdf).numpy(),
        (mean_itd_from_pdf + std_itd_from_pdf).numpy(),
        color='orange',
        alpha=0.25,
    )
    axs[0].set_title(f'ITD uncertainty | cycle={cycle+1} epoch={epoch}')
    axs[0].set_xlabel(r'$\nu$')
    axs[0].set_ylabel(r'$\mathcal{M}(\nu)$')
    axs[0].set_ylim(*ITD_TO_PDF_ITD_YLIM)
    axs[0].legend()

    axs[1].plot(x_axis, mean_pdf.numpy(), color='orange', label='INN inverse')
    axs[1].fill_between(
        x_axis,
        (mean_pdf - std_pdf).numpy(),
        (mean_pdf + std_pdf).numpy(),
        color='orange',
        alpha=0.3,
    )
    axs[1].plot(x_axis, mean_pdf_true.numpy(), color='skyblue', label='True PDF batch')
    axs[1].fill_between(
        x_axis,
        (mean_pdf_true - std_pdf_true).numpy(),
        (mean_pdf_true + std_pdf_true).numpy(),
        color='skyblue',
        alpha=0.3,
    )
    axs[1].set_title('Mapped PDF uncertainty')
    axs[1].set_xlabel(r'$x$')
    axs[1].set_ylabel(r'$f(x)$')
    axs[1].set_ylim(*GIF_PDF_YLIM)
    axs[1].legend()

    axs[2].plot(x_axis, mean_pdf_norm.numpy(), color='orange', label='INN inverse')
    axs[2].fill_between(
        x_axis,
        (mean_pdf_norm - std_pdf_norm).numpy(),
        (mean_pdf_norm + std_pdf_norm).numpy(),
        color='orange',
        alpha=0.3,
    )
    axs[2].plot(x_axis, mean_pdf_true_norm.numpy(), color='skyblue', label='True mean')
    axs[2].fill_between(
        x_axis,
        (mean_pdf_true_norm - std_pdf_true_norm).numpy(),
        (mean_pdf_true_norm + std_pdf_true_norm).numpy(),
        color='skyblue',
        alpha=0.3,
    )
    axs[2].axhline(0.0, color='k', linewidth=1.0, linestyle='--')
    axs[2].set_title('PDF vs true')
    axs[2].set_xlabel(r'$x$')
    axs[2].set_ylabel(r'$(f-\mu_{\mathrm{true}})/\sigma_{\mathrm{true}}$')
    axs[2].set_ylim(-4.0, 4.0)
    axs[2].legend()

    plt.tight_layout()
    filename = os.path.join(PLOT_DIR, f"itd_to_pdf_cycle_{cycle+1:02d}_epoch_{epoch:05d}.png")
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"Saved ITD->PDF uncertainty snapshot: {filename}", flush=True)


def initialize_loss_history():
    return {
        "total": [],
        "Ly": [],
        "Lz": [],
        "Lx": [],
        "Lpad": [],
        "Ls": [],
        "cycles": [],
    }


def record_loss(history, cycle, epoch, loss, Ly, Lz, Lx, Lpad, Ls):
    history["total"].append(loss.item())
    history["Ly"].append(Ly.item())
    history["Lz"].append(Lz.item())
    history["Lx"].append(Lx.item())
    history["Lpad"].append(Lpad.item())
    history["Ls"].append(Ls.item())
    history["cycles"].append((cycle, epoch))


def plot_loss_history(history, title):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    keys = ["total", "Ly", "Lz", "Lx", "Lpad", "Ls"]
    titles = ["Total", "Ly (ITD)", "Lz (Z)", "Lx (PDF)", "Lpad", "Ls"]

    for ax, key, key_title in zip(axs.flat, keys, titles):
        ax.plot(history[key], linewidth=1.5)
        ax.set_title(key_title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")

    fig.suptitle(title)
    plt.tight_layout()
    safe_title = title.lower().replace(" ", "_").replace("|", "").replace("(", "").replace(")", "")
    filename = os.path.join(PLOT_DIR, f"{safe_title}.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot: {filename}", flush=True)


def plot_loss_history_combined(history, title):
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    for key in ["total", "Ly", "Lz", "Lx", "Lpad", "Ls"]:
        ax.plot(history[key], linewidth=1.5, label=key)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    safe_title = title.lower().replace(" ", "_").replace("|", "").replace("(", "").replace(")", "")
    filename = os.path.join(PLOT_DIR, f"{safe_title}_combined.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined loss plot: {filename}", flush=True)


def save_evaluation_snapshot(pdf_pred, pdf_true, itd_pred, itd_true, x_grid, nu_values, tag):
    pdf_points = min(len(x_grid), pdf_pred.shape[1], pdf_true.shape[1])
    itd_points = min(len(nu_values), itd_pred.shape[1], itd_true.shape[1])

    pdf_pred_cpu = denormalize_pdf_data(pdf_pred[:, :pdf_points]).detach().cpu()
    pdf_true_cpu = denormalize_pdf_data(pdf_true[:, :pdf_points]).detach().cpu()
    itd_pred_cpu = project_pdf_to_full_itd(pdf_pred[:, :pdf_points]).detach().cpu()
    itd_true_cpu = project_pdf_to_full_itd(pdf_true[:, :pdf_points]).detach().cpu()
    itd_true_obs_cpu = denormalize_itd_data(itd_true[:, :itd_points]).detach().cpu()

    mean_pdf_pred = pdf_pred_cpu.mean(dim=0)
    mean_pdf_true = pdf_true_cpu.mean(dim=0)
    std_pdf_pred = pdf_pred_cpu.std(dim=0)
    std_pdf_true = pdf_true_cpu.std(dim=0)

    mean_itd_pred = itd_pred_cpu.mean(dim=0)
    mean_itd_true = itd_true_cpu.mean(dim=0)
    std_itd_pred = itd_pred_cpu.std(dim=0)
    std_itd_true = itd_true_cpu.std(dim=0)
    mean_itd_true_obs = itd_true_obs_cpu.mean(dim=0)
    std_itd_true_obs = itd_true_obs_cpu.std(dim=0)

    x_axis = np.asarray(x_grid[:pdf_points])
    nu_axis = FULL_NU_GRID[:mean_itd_pred.shape[0]].detach().cpu().numpy()
    nu_obs_axis = nu_values[:itd_points].detach().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    axs[0].plot(x_axis, mean_pdf_pred.numpy(), color='orange', label='INN')
    axs[0].fill_between(x_axis, (mean_pdf_pred - std_pdf_pred).numpy(), (mean_pdf_pred + std_pdf_pred).numpy(), color='orange', alpha=0.3)
    axs[0].plot(x_axis, mean_pdf_true.numpy(), color='skyblue', label='True')
    axs[0].fill_between(x_axis, (mean_pdf_true - std_pdf_true).numpy(), (mean_pdf_true + std_pdf_true).numpy(), color='skyblue', alpha=0.3)
    axs[0].set_title('PDF')
    axs[0].set_xlabel(r'$x$')
    axs[0].set_ylabel(r'$f(x)$')
    set_pdf_axis_limits(axs[0], mean_pdf_pred, std_pdf_pred, mean_pdf_true, std_pdf_true)
    axs[0].legend()

    axs[1].plot(nu_axis, mean_itd_pred.numpy(), color='orange', label='INN')
    axs[1].fill_between(nu_axis, (mean_itd_pred - std_itd_pred).numpy(), (mean_itd_pred + std_itd_pred).numpy(), color='orange', alpha=0.3)
    axs[1].plot(nu_axis, mean_itd_true.numpy(), color='skyblue', label='True')
    axs[1].fill_between(nu_axis, (mean_itd_true - std_itd_true).numpy(), (mean_itd_true + std_itd_true).numpy(), color='skyblue', alpha=0.3)
    axs[1].errorbar(
        nu_obs_axis,
        mean_itd_true_obs.numpy(),
        yerr=std_itd_true_obs.numpy(),
        fmt='o',
        color='navy',
        markersize=4,
        linewidth=1.0,
        capsize=3,
        label='Data',
    )
    axs[1].set_title('ITD')
    axs[1].set_xlabel(r'$\nu$')
    axs[1].set_ylabel(r'$\mathcal{M}(\nu)$')
    axs[1].legend()

    plt.tight_layout()
    filename = os.path.join(PLOT_DIR, f"evaluation_{tag}.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation snapshot: {filename}", flush=True)


def evaluate_on_parametric_dataset(model, x_grid, nu_values, active_integrator, z_dim, device, dtype, n_eval_samples=1000):
    print(f"Building parametric evaluation dataset with {n_eval_samples} samples...", flush=True)
    eval_data = create_parametric_pdf_samples(x_grid, n_eval_samples)
    pdf_eval_raw = eval_data["samples"].to(device=device, dtype=dtype)
    itd_eval_raw = eval_data["samples"].to(device=device, dtype=dtype) @ tr.tensor(
        active_integrator,
        device=device,
        dtype=dtype,
    ).T
    pdf_eval = normalize_pdf_data(pdf_eval_raw)
    itd_eval = normalize_itd_data(itd_eval_raw)

    with tr.no_grad():
        itd_pred, z_pred, pad_pred, logdet = model(pdf_eval)
        z_rand = tr.randn(pdf_eval.shape[0], z_dim, device=device, dtype=dtype)
        if model.yz_pad_dim > 0:
            pad_rand = sample_padding_noise(
                pdf_eval.shape[0],
                model.yz_pad_dim,
                kernoise,
                device=device,
                dtype=dtype,
                nu_start=nu_grid[-1].item(),
            )
        else:
            pad_rand = None
        pdf_rec = model.inverse(itd_eval, z_rand, pad_rand)
        pdf_rec_raw = denormalize_pdf_data(pdf_rec)
        itd_pred_raw = denormalize_itd_data(itd_pred)

        metrics = {
            "itd_l1": nn.L1Loss()(itd_pred_raw, itd_eval_raw).item(),
            "pdf_l1": nn.L1Loss()(pdf_rec_raw, pdf_eval_raw).item(),
            "itd_mmd": MMD_multiquadratic(itd_pred_raw, itd_eval_raw).item(),
            "pdf_mmd": MMD_multiquadratic(pdf_rec_raw, pdf_eval_raw).item(),
        }

    print(
        f"Parametric evaluation | ITD L1 {metrics['itd_l1']:.6f} | PDF L1 {metrics['pdf_l1']:.6f} | "
        f"ITD MMD {metrics['itd_mmd']:.6f} | PDF MMD {metrics['pdf_mmd']:.6f}",
        flush=True,
    )
    save_evaluation_snapshot(pdf_rec, pdf_eval, itd_pred, itd_eval, x_grid, nu_values, "parametric_dataset")
    return eval_data, metrics


def summarize_inn_pdf_from_itd_dataset(model, itd_dataset, z_dim, device, dtype, chunk_size=2048):
    n_samples = itd_dataset.shape[0]
    pdf_sum = None
    pdf_sq_sum = None
    pdf_outer_sum = None

    with tr.no_grad():
        for start in range(0, n_samples, chunk_size):
            stop = min(start + chunk_size, n_samples)
            itd_chunk = itd_dataset[start:stop].to(device=device, dtype=dtype)
            z_rand = tr.randn(itd_chunk.shape[0], z_dim, device=device, dtype=dtype)
            if model.yz_pad_dim > 0:
                pad_rand = sample_padding_noise(
                    itd_chunk.shape[0],
                    model.yz_pad_dim,
                    kernoise,
                    device=device,
                    dtype=dtype,
                    nu_start=nu_grid[-1].item(),
                )
            else:
                pad_rand = None

            pdf_chunk = model.inverse(itd_chunk, z_rand, pad_rand)
            pdf_chunk_raw = denormalize_pdf_data(pdf_chunk)

            if pdf_sum is None:
                pdf_dim_local = pdf_chunk_raw.shape[1]
                pdf_sum = tr.zeros(pdf_dim_local, device=device, dtype=dtype)
                pdf_sq_sum = tr.zeros(pdf_dim_local, device=device, dtype=dtype)
                pdf_outer_sum = tr.zeros(pdf_dim_local, pdf_dim_local, device=device, dtype=dtype)

            pdf_sum = pdf_sum + pdf_chunk_raw.sum(dim=0)
            pdf_sq_sum = pdf_sq_sum + (pdf_chunk_raw ** 2).sum(dim=0)
            pdf_outer_sum = pdf_outer_sum + pdf_chunk_raw.T @ pdf_chunk_raw

    pdf_mean = pdf_sum / n_samples
    pdf_var = (pdf_sq_sum / n_samples) - pdf_mean ** 2
    pdf_std = tr.sqrt(tr.clamp(pdf_var, min=0.0))

    if n_samples > 1:
        pdf_cov = (pdf_outer_sum - n_samples * tr.outer(pdf_mean, pdf_mean)) / (n_samples - 1)
    else:
        pdf_cov = tr.zeros(pdf_mean.shape[0], pdf_mean.shape[0], device=device, dtype=dtype)

    return {
        "mean": pdf_mean.detach().cpu(),
        "std": pdf_std.detach().cpu(),
        "cov": pdf_cov.detach().cpu(),
        "n_samples": tr.tensor(n_samples, dtype=tr.int64),
    }


def split_samples_across_kernels(n_samples, kernel_names):
    samples_per_kernel = [n_samples // len(kernel_names)] * len(kernel_names)
    for idx in range(n_samples % len(kernel_names)):
        samples_per_kernel[idx] += 1
    return samples_per_kernel


def prepare_cycle_training_batches(x_grid, n_samples, kernel_names, batch_size, reuse_stats=False):
    global PDF_DATA_STATS, ITD_DATA_STATS, PDF_DATA_MEAN, PDF_DATA_COV, ITD_DATA_MEAN, ITD_DATA_COV

    samples_per_kernel = split_samples_across_kernels(n_samples, kernel_names)
    kernel_blocks = []

    print(
        f"Refreshing training samples for cycle using kernels {kernel_names}...",
        flush=True,
    )
    for kernel_name, n_block in zip(kernel_names, samples_per_kernel):
        if n_block == 0:
            continue
        prior_block = create_pdf_prior_samples(x_grid, n_block, kernel_name=kernel_name)
        itd_block = create_itd_samples(prior_block["samples"], itd_real_integrator, itd_imag_integrator)
        observed_itd_raw = (
            itd_block["real"] if cli_args.itd_part == "real" else itd_block["imag"]
        ).transpose(1, 0)
        kernel_blocks.append(
            {
                "kernel_name": kernel_name,
                "pdf_raw": prior_block["samples"],
                "itd_raw": observed_itd_raw,
            }
        )

    all_pdf_raw = tr.cat([block["pdf_raw"] for block in kernel_blocks], dim=0)
    all_itd_raw = tr.cat([block["itd_raw"] for block in kernel_blocks], dim=0)

    if reuse_stats and PDF_DATA_STATS is not None and ITD_DATA_STATS is not None:
        print("Reusing PDF/ITD normalization statistics from the first cycle.", flush=True)
    else:
        PDF_DATA_STATS = fit_gaussian_normalization_stats(all_pdf_raw, "PDF")
        ITD_DATA_STATS = fit_gaussian_normalization_stats(all_itd_raw, "ITD")
        PDF_DATA_MEAN = PDF_DATA_STATS["mean"]
        PDF_DATA_COV = PDF_DATA_STATS["cov"]
        ITD_DATA_MEAN = ITD_DATA_STATS["mean"]
        ITD_DATA_COV = ITD_DATA_STATS["cov"]

    if cli_args.normalize_data:
        print("Applying PDF/ITD whitening normalization for current cycle.", flush=True)
        observed_itd_train_cycle = normalize_itd_data(all_itd_raw)
        parametric_pdf_eval_cycle = normalize_pdf_data(parametric_pdf_eval_raw)
        parametric_itd_eval_cycle = normalize_itd_data(parametric_itd_eval_raw)
    else:
        print("Data normalization disabled for current cycle; using raw PDF and ITD values.", flush=True)
        observed_itd_train_cycle = all_itd_raw
        parametric_pdf_eval_cycle = parametric_pdf_eval_raw
        parametric_itd_eval_cycle = parametric_itd_eval_raw

    batch_groups = []
    for block in kernel_blocks:
        pdf_train = normalize_pdf_data(block["pdf_raw"]) if cli_args.normalize_data else block["pdf_raw"]
        itd_train = normalize_itd_data(block["itd_raw"]) if cli_args.normalize_data else block["itd_raw"]
        n_block_batches = pdf_train.shape[0] // batch_size
        if n_block_batches == 0:
            continue
        pdf_batches = pdf_train[:n_block_batches * batch_size].reshape(n_block_batches, batch_size, pdf_train.shape[1])
        itd_batches = itd_train[:n_block_batches * batch_size].reshape(n_block_batches, batch_size, itd_train.shape[1])
        batch_groups.append(
            {
                "kernel_name": block["kernel_name"],
                "pdf_batches": pdf_batches,
                "itd_batches": itd_batches,
                "n_batches": n_block_batches,
            }
        )
        print(
            f"Cycle data | kernel={block['kernel_name']} | samples={pdf_train.shape[0]} | batches={n_block_batches}",
            flush=True,
        )

    return batch_groups, observed_itd_train_cycle, parametric_pdf_eval_cycle, parametric_itd_eval_cycle



#resize data in batches
# (n_batches, BATCH_SIZE, n_points)
BATCH_SIZE = 500
print(f"Preparing cycle-wise batches of size {BATCH_SIZE}...", flush=True)

itd_dim = Nnu
pdf_dim = samples.shape[-1]
z_dim = 12
hid_dim = 256
typeof_data = tr.float64
n_epochs = 2000
LOG_INTERVAL = 100


INN1 = INNforPDF_ITD(pdf_dim, itd_dim, z_dim, hid_dim, n_blocks=6, clamp=1.0).to(device).to(typeof_data)
optimizer = tr.optim.Adam(INN1.parameters(), lr=1e-4)
#scheduler = tr.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3,threshold=1e-3,min_lr=1e-6)
scheduler = tr.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2,eta_min=1e-6)
loss_y = nn.L1Loss()
loss_pad = nn.MSELoss()
lambda_y = 1.0 / 3.0
lambda_z = 1.0 / 3.0
lambda_x = 1.0 / 3.0
lambda_pad = 1.0 / 6.0

CYCLES = cli_args.cycles
loss_history = initialize_loss_history()
cycle_loss_histories = []

print(
    f"Starting training: cycles={CYCLES}, epochs_per_cycle={n_epochs}, "
    f"pdf_dim={pdf_dim}, itd_dim={itd_dim}, z_dim={z_dim}",
    flush=True,
)

observed_itd_train_final = observed_itd_train
for cycle in range(CYCLES):
    cycle_history = initialize_loss_history()
    cycle_loss_histories.append(cycle_history)
    print(f"Cycle {cycle+1}/{CYCLES}", flush=True)
    cycle_batch_groups, observed_itd_train_final, parametric_pdf_eval, parametric_itd_eval = prepare_cycle_training_batches(
        x_grid_model,
        SAMPLES,
        selected_gp_kernels,
        BATCH_SIZE,
        reuse_stats=(cycle > 0 and cli_args.normalize_data),
    )
    n_kernel_groups = len(cycle_batch_groups)
    for epoch in range(n_epochs):
        if n_kernel_groups == 1:
            batch_group = cycle_batch_groups[0]
            b = epoch % batch_group["n_batches"]
        else:
            batch_group = cycle_batch_groups[epoch % n_kernel_groups]
            b = (epoch // n_kernel_groups) % batch_group["n_batches"]

        pdf_batch = batch_group["pdf_batches"][b].to(device=device, dtype=typeof_data)
        itd_batch = batch_group["itd_batches"][b].to(device=device, dtype=typeof_data)
        kernel_tag_batch = batch_group["kernel_name"]
        z_rand = tr.randn(pdf_batch.shape[0], z_dim, device=device, dtype=typeof_data)

        if INN1.yz_pad_dim > 0:
            pad_rand = sample_padding_noise(
                pdf_batch.shape[0],
                INN1.yz_pad_dim,
                kernoise,
                device=device,
                dtype=typeof_data,
                nu_start=nu_grid[-1].item(),
            )
        else:
            pad_rand = None

        optimizer.zero_grad()

        itd_pred, z_pred, pad_pred, logdet = INN1(pdf_batch)
        pdf_pred = INN1.inverse(itd_batch, z_rand, pad_rand)

        yz_pred = tr.cat([itd_pred.detach(), z_pred], dim=1)
        yz_true = tr.cat([itd_batch, z_rand], dim=1)

        Ly = loss_y(itd_pred, itd_batch)
        Lz = MMD_multiquadratic(yz_pred, yz_true)
        Lx = MMD_multiquadratic(pdf_pred, pdf_batch)
        Lpad = loss_pad(pad_pred, tr.zeros_like(pad_pred))
        # Previous prescription kept for reference:
        # if cycle == 0:
        #     Ls = smoothness_loss(pdf_pred[:, :], pdf_batch[:, :], x_grid_model_tr[:])
        # else:
        #     Ls = tr.tensor(0.0, device=device, dtype=typeof_data)
        Ls = smoothness_loss(pdf_pred[:, :], pdf_batch[:, :], x_grid_model_tr[:])

        # Extended prescription kept here for reference:
        # w_mmd_z, w_mmd_x = get_mmd_loss_weights(cycle)
        # w_s = get_smoothness_weight(cycle, epoch, n_epochs)
        # loss = 5.0 * Ly + w_mmd_z * Lz + w_mmd_x * Lx + 1.0 * Lpad + w_s * Ls
        # INN paper-style objective:
        # supervised forward loss Ly
        # latent-space MMD loss Lz
        # inverse-space MMD loss Lx
        # padding loss Lpad (needed because this architecture uses output padding)
        loss = lambda_y * Ly + lambda_z * Lz + lambda_x * Lx + lambda_pad * Lpad
        record_loss(loss_history, cycle, epoch, loss, Ly, Lz, Lx, Lpad, Ls)
        record_loss(cycle_history, cycle, epoch, loss, Ly, Lz, Lx, Lpad, Ls)
        loss.backward()
        tr.nn.utils.clip_grad_norm_(INN1.parameters(), 1.0)
        optimizer.step()
        scheduler.step(cycle * n_epochs + epoch)
        if epoch % LOG_INTERVAL == 0:
            print(
                f"Epoch {epoch:5d} | kernels {kernel_tag_batch} | "
                f"ly {lambda_y:.2f} lz {lambda_z:.2f} lx {lambda_x:.2f} lpad {lambda_pad:.2f} | "
                f"Loss {loss.item():.6f} | Ly(ITD) {Ly.item():.6f} | "
                f"Lz(Z) {Lz.item():.6f} | Lx(PDF) {Lx.item():.6f} | "
                f"Lpad {Lpad.item():.6f} | Ls {Ls.item():.6f}",
                flush=True,
            )
            show_training_snapshot(
                pdf_pred,
                pdf_batch,
                itd_pred,
                itd_batch,
                x_grid_model,
                nu_grid,
                cycle,
                epoch,
            )
            with tr.no_grad():
                z_param = tr.randn(parametric_pdf_eval.shape[0], z_dim, device=device, dtype=typeof_data)
                if INN1.yz_pad_dim > 0:
                    pad_param = sample_padding_noise(
                        parametric_pdf_eval.shape[0],
                        INN1.yz_pad_dim,
                        kernoise,
                        device=device,
                        dtype=typeof_data,
                        nu_start=nu_grid[-1].item(),
                    )
                else:
                    pad_param = None
                parametric_pdf_from_itd = INN1.inverse(parametric_itd_eval, z_param, pad_param)
                parametric_pdf_from_itd_raw = denormalize_pdf_data(parametric_pdf_from_itd)
                parametric_itd_from_pdf_raw = parametric_pdf_from_itd_raw @ tr.tensor(
                    active_itd_integrator,
                    device=device,
                    dtype=typeof_data,
                ).T
                parametric_itd_from_pdf = normalize_itd_data(parametric_itd_from_pdf_raw)
            show_itd_to_pdf_uncertainty_snapshot(
                parametric_itd_eval,
                parametric_itd_from_pdf,
                parametric_pdf_from_itd,
                parametric_pdf_eval,
                x_grid_model,
                nu_grid,
                cycle,
                epoch,
            )
    plot_loss_history(cycle_history, f"Loss History | Cycle {cycle+1}")
    plot_loss_history_combined(cycle_history, f"Loss History | Cycle {cycle+1}")
    create_gif_and_cleanup(
        os.path.join(PLOT_DIR, f"snapshot_cycle_{cycle+1:02d}_epoch_*.png"),
        f"snapshot_cycle_{cycle+1:02d}.gif",
        fps=2,
    )
    create_gif_and_cleanup(
        os.path.join(PLOT_DIR, f"itd_to_pdf_cycle_{cycle+1:02d}_epoch_*.png"),
        f"itd_to_pdf_cycle_{cycle+1:02d}.gif",
        fps=2,
    )

plot_loss_history(loss_history, "Loss History | All Cycles")
plot_loss_history_combined(loss_history, "Loss History | All Cycles")
create_gif_from_gifs(
    os.path.join(PLOT_DIR, "snapshot_cycle_*.gif"),
    "snapshot_all_cycles.gif",
)
create_gif_from_gifs(
    os.path.join(PLOT_DIR, "itd_to_pdf_cycle_*.gif"),
    "itd_to_pdf_all_cycles.gif",
)

parametric_eval_data, parametric_eval_metrics = evaluate_on_parametric_dataset(
    INN1,
    x_grid_model,
    nu_grid,
    active_itd_integrator,
    z_dim,
    device,
    typeof_data,
    n_eval_samples=1000,
)

print("Saving INN PDF summary from sampled data ITDs...", flush=True)
inn_pdf_summary_name = f"INN_PDF_dat_{cli_args.itd_points:02d}"
globals()[inn_pdf_summary_name] = summarize_inn_pdf_from_itd_dataset(
    INN1,
    observed_itd_train_final[:, :Nnu],
    z_dim,
    device,
    typeof_data,
)
inn_pdf_summary_path = os.path.join(PLOT_DIR, f"{inn_pdf_summary_name}.pt")
tr.save(globals()[inn_pdf_summary_name], inn_pdf_summary_path)
print(f"Saved INN PDF summary: {inn_pdf_summary_path}", flush=True)
