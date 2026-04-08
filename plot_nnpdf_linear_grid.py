import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

from functions import FE2_Integrator, generategrid


BASE_DIR = Path(__file__).resolve().parent
NNPDF_DIR = BASE_DIR / "NNPDF"


# Match the LaTeX styling used in INN.py
os.environ["PATH"] = "/sciclone/home/yacahuanamedra/texlive/bin/x86_64-linux:" + os.environ["PATH"]
rcParams.update({"figure.autolayout": True})
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amsfonts}")
font = {
    "family": "normal",
    "weight": "bold",
    "size": 26,
}
mpl.rc("font", **font)


def load_nnpdf_replicas():
    pdf_minus = np.loadtxt(NNPDF_DIR / "NNPDF40_nnlo_as_01180_1000_pdf_minus.dat", dtype=np.float64)
    pdf_plus = np.loadtxt(NNPDF_DIR / "NNPDF40_nnlo_as_01180_1000_pdf_plus.dat", dtype=np.float64)

    x_orig = pdf_minus[:, 0]
    q_minus_replicas = pdf_minus[:, 1:]
    q_plus_replicas = pdf_plus[:, 1:]
    return x_orig, q_minus_replicas, q_plus_replicas


def interpolate_replicas(x_orig, replicas, x_target):
    n_rep = replicas.shape[1]
    out = np.empty((x_target.shape[0], n_rep), dtype=np.float64)
    for idx in range(n_rep):
        out[:, idx] = np.interp(x_target, x_orig, replicas[:, idx])
    return out


def build_itd_integrators(x_grid, nu_grid):
    fe = FE2_Integrator(x_grid)
    real_integrator = np.zeros((nu_grid.shape[0], x_grid.shape[0]), dtype=np.float64)
    imag_integrator = np.zeros((nu_grid.shape[0], x_grid.shape[0]), dtype=np.float64)
    for k, nu in enumerate(nu_grid):
        real_integrator[k, :] = fe.set_up_integration(Kernel=lambda x, nu=nu: np.cos(nu * x))
        imag_integrator[k, :] = fe.set_up_integration(Kernel=lambda x, nu=nu: np.sin(nu * x))
    return real_integrator, imag_integrator


def plot_component_1x2(nu_grid, itd_samples, x_grid, pdf_samples, itd_label, pdf_label, itd_ylim, pdf_ymax, out_path):
    mean_itd = itd_samples.mean(axis=1)
    std_itd = itd_samples.std(axis=1)
    mean_pdf = pdf_samples.mean(axis=1)
    std_pdf = pdf_samples.std(axis=1)
    nu_margin = 0.02 * (nu_grid.max() - nu_grid.min())
    x_margin = 0.02 * (x_grid.max() - x_grid.min())
    pdf_epsilon = 0.1

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    axs[0].plot(nu_grid, mean_itd, color="C0", linewidth=2.0)
    axs[0].fill_between(
        nu_grid,
        mean_itd - std_itd,
        mean_itd + std_itd,
        color="C0",
        alpha=0.2,
    )
    axs[0].set_xlabel(r"$\nu$")
    axs[0].set_ylabel(itd_label)
    axs[0].set_xlim(nu_grid.min() - nu_margin, nu_grid.max() + nu_margin)
    axs[0].set_ylim(*itd_ylim)

    axs[1].plot(x_grid, mean_pdf, color="C0", linewidth=2.0)
    axs[1].fill_between(
        x_grid,
        mean_pdf - std_pdf,
        mean_pdf + std_pdf,
        color="C0",
        alpha=0.2,
    )
    axs[1].set_xlabel(r"$x$")
    axs[1].set_ylabel(pdf_label)
    axs[1].set_xlim(x_grid.min() - x_margin, x_grid.max() + x_margin)
    axs[1].set_ylim(-pdf_epsilon, pdf_ymax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    x_grid_lin = generategrid(128, "linh")
    nu_grid_full = np.linspace(0.0, 100.0, 128)

    x_orig, q_minus_replicas, q_plus_replicas = load_nnpdf_replicas()
    q_minus_lin = interpolate_replicas(x_orig, q_minus_replicas, x_grid_lin)
    q_plus_lin = interpolate_replicas(x_orig, q_plus_replicas, x_grid_lin)

    real_integrator, imag_integrator = build_itd_integrators(x_grid_lin, nu_grid_full)
    real_itd = real_integrator @ q_minus_lin
    imag_itd = imag_integrator @ q_plus_lin

    real_path = BASE_DIR / "NNPDF_linear_grid_real_1x2.pdf"
    imag_path = BASE_DIR / "NNPDF_linear_grid_imag_1x2.pdf"

    plot_component_1x2(
        nu_grid_full,
        real_itd,
        x_grid_lin,
        q_minus_lin,
        r"$\mathrm{Re}M_q(\nu,z^2=0)$",
        r"$q-\bar{q}$",
        (-0.03, 1.1),
        6.0,
        real_path,
    )
    plot_component_1x2(
        nu_grid_full,
        imag_itd,
        x_grid_lin,
        q_plus_lin,
        r"$\mathrm{Im}M_q(\nu,z^2=0)$",
        r"$q+\bar{q}$",
        (-0.05, 0.5),
        4.0,
        imag_path,
    )

    print(f"Saved real figure: {real_path}")
    print(f"Saved imag figure: {imag_path}")


if __name__ == "__main__":
    main()
