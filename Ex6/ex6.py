"""Implementation question 6: CG and SSOR-preconditioned CG.

Solves A^h u^h = f^h with unpreconditioned CG and with SSOR(omega=1.5)-PCG,
plots ||r^k||_2 / ||f^h||_2 against k and reports n, iterations and CPU time.
Usage:  python Ex6/ex6.py

Implemented by us: CG/PCG iteration, SSOR preconditioner application.
Library routines:  sparse mat-vec (scipy.sparse), sparse triangular solves
(SuperLU via scipy.sparse.linalg.splu, see ssor_preconditioner), dot
products/norms (numpy).
"""
import os

# Single-threaded BLAS: multithreaded dot products on mid-size vectors are
# slower and make the timings erratic.  Must be set before importing numpy.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Ex1"))
from ex1 import assemble  # noqa: E402

Q_LEVELS = range(4, 9)  # N = 16, ..., 256 (same sizes as the SSOR experiments)
OMEGA = 1.5
TOL = 1e-10
FIG_DIR = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(FIG_DIR, "..", "latex.mplstyle"))  # Computer Modern fonts


# ---------- Preconditioner ----------
def ssor_preconditioner(A, omega=OMEGA):
    """Return z = M_SSOR^{-1} r, with A = D - E - F and
    M_SSOR = 1/(omega(2-omega)) (D - omega E) D^{-1} (D - omega F).

    Setup (done once): the lower triangular matrix L = D - omega E.  Since A is
    symmetric, D - omega F = L^T.  We hand L to SuperLU with natural ordering
    and no pivoting: as L is already triangular, no elimination/fill-in occurs
    (perm_r = perm_c = identity) and lu.solve is just a fast compiled
    forward substitution with L; lu.solve(., trans="T") is the backward
    substitution with L^T.  (spsolve_triangular gives the same result but is
    ~6x slower per call.)
    Application: forward solve, diagonal scaling, backward solve (Q5b).
    """
    d = A.diagonal()
    L = (sp.diags(d) + omega * sp.tril(A, k=-1)).tocsc()  # D - omega E  (-E = tril(A))
    lu = splu(L, permc_spec="NATURAL", diag_pivot_thresh=0.0,
              options={"SymmetricMode": True})
    scale = omega * (2.0 - omega)

    def apply(r):
        y = lu.solve(r)                           # (D - omega E) y = r
        return lu.solve(scale * d * y, trans="T")  # (D - omega F) z = omega(2-omega) D y

    return apply


# ---------- (P)CG ----------
def pcg(A, f, apply_prec=None, tol=TOL, maxit=100_000):
    """(Preconditioned) conjugate gradient with u^0 = 0.

    Stops when ||r^k||_2 / ||f||_2 <= tol, using the true (unpreconditioned)
    residual r^k = f - A u^k, updated recursively.  apply_prec=None gives CG.
    Returns u and the history of ||r^k||_2 / ||f||_2, k = 0, 1, ...
    """
    u = np.zeros_like(f)
    r = f.copy()                      # r^0 = f - A u^0 = f
    norm_f = np.linalg.norm(f)
    hist = [1.0]
    z = r if apply_prec is None else apply_prec(r)
    p = z.copy()
    rz = r @ z
    for _ in range(maxit):
        Ap = A @ p
        alpha = rz / (p @ Ap)
        u += alpha * p
        r -= alpha * Ap
        hist.append(np.linalg.norm(r) / norm_f)
        if hist[-1] <= tol:
            break
        z = r if apply_prec is None else apply_prec(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return u, np.array(hist)


def main():
    methods = {"CG": None, "SSOR-PCG": "ssor"}
    results = {m: [] for m in methods}  # (N, n, iterations, time, history)

    for q in Q_LEVELS:
        N = 2**q
        A, f, _ = assemble(N)
        for name, prec in methods.items():
            t0 = time.perf_counter()
            apply_prec = ssor_preconditioner(A) if prec == "ssor" else None
            t_setup = time.perf_counter() - t0  # preconditioner setup, not in t_iter
            t0 = time.perf_counter()
            u, hist = pcg(A, f, apply_prec)
            t_iter = time.perf_counter() - t0
            true_res = np.linalg.norm(f - A @ u) / np.linalg.norm(f)  # sanity check
            results[name].append((N, A.shape[0], len(hist) - 1, t_iter, hist))
            print(f"{name:>9}  N={N:>4}  n={A.shape[0]:>6}  it={len(hist) - 1:>5}  "
                  f"time={t_iter:8.3f} s  (setup {t_setup:.3f} s)  true relres={true_res:.2e}")

    # ---------- Table ----------
    print(f"\n{'n':>7} | {'CG it':>6} {'CG time[s]':>10} | {'PCG it':>6} {'PCG time[s]':>11}")
    for (N, n, itc, tc, _), (_, _, itp, tp, _) in zip(results["CG"], results["SSOR-PCG"]):
        print(f"{n:>7} | {itc:>6} {tc:>10.3f} | {itp:>6} {tp:>11.3f}")

    # ---------- Plots ----------
    # One figure per method, same size and same axis limits so that they can be
    # compared side by side (subfigures (a) CG, (b) SSOR-PCG in the report).
    k_max = max(len(h) - 1 for res in results.values() for *_, h in res)
    files = {"CG": "cg_convergence.pdf", "SSOR-PCG": "pcg_ssor_convergence.pdf"}
    for name in methods:
        fig, ax = plt.subplots(figsize=(7.0, 2.3))
        for N, n, _, _, hist in results[name]:
            (line,) = ax.semilogy(hist, label=rf"$N={N},\ n={n}$")
            ax.plot(len(hist) - 1, hist[-1], "o", color=line.get_color())  # converged iterate
        ax.axhline(TOL, color="k", ls=":", lw=1)
        ax.set_xlim(-0.02 * k_max, 1.02 * k_max)
        ax.set_ylim(TOL / 5, 2.0)
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$\dfrac{\|\mathbf{r}^k\|_2}{\|\mathbf{f}^h\|_2}$",
                      rotation=0, ha="right", va="center")
        if name == "SSOR-PCG":
            ax.legend(fontsize=10, loc="center right", title=r"$n=(N+1)^2$", title_fontsize=10)
        out = os.path.join(FIG_DIR, files[name])
        fig.savefig(out)
        plt.close(fig)
        print(f"Figure saved to {os.path.normpath(out)}")

    # ---------- Iterations vs mesh size, and CG/PCG ratio ----------
    Ns = np.array([r[0] for r in results["CG"]])
    K_cg = np.array([r[2] for r in results["CG"]])
    K_pcg = np.array([r[2] for r in results["SSOR-PCG"]])

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.loglog(Ns, K_cg, "o-", label="CG")
    ax.loglog(Ns, K_pcg, "s-", label="SSOR-PCG")
    ax.loglog(Ns, K_cg[-1] * Ns / Ns[-1], "k--", lw=1, label=r"$\mathcal{O}(N)$")
    ax.loglog(Ns, K_pcg[-1] * Ns / Ns[-1], "k--", lw=1)
    ax.set_xticks(Ns, [str(N) for N in Ns])
    ax.minorticks_off()
    ax.set_xlabel(r"$N=1/h$")
    ax.set_ylabel(r"$K$", rotation=0, ha="right", va="center")
    ax.legend(fontsize=10)
    out = os.path.join(FIG_DIR, "iterations_vs_N.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure saved to {os.path.normpath(out)}")

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.semilogx(Ns, K_cg / K_pcg, "o-", color="C2")
    for N, ratio in zip(Ns, K_cg / K_pcg):
        ax.annotate(f"{ratio:.1f}", (N, ratio), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=10)
    ax.set_xticks(Ns, [str(N) for N in Ns])
    ax.minorticks_off()
    ax.set_ylim(0, 1.15 * (K_cg / K_pcg).max())
    ax.set_xlabel(r"$N=1/h$")
    ax.set_ylabel(r"$\dfrac{K_{\mathrm{CG}}}{K_{\mathrm{PCG}}}$", rotation=0, ha="right", va="center")
    out = os.path.join(FIG_DIR, "iteration_ratio.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure saved to {os.path.normpath(out)}")

    # ---------- CPU time: total and per iteration ----------
    ns = np.array([r[1] for r in results["CG"]])
    T = {m: np.array([r[3] for r in results[m]]) for m in methods}
    K = {"CG": K_cg, "SSOR-PCG": K_pcg}
    markers = {"CG": "o-", "SSOR-PCG": "s-"}

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    for m in methods:
        ax.loglog(ns, T[m], markers[m], label=m)
    ax.loglog(ns, T["CG"][-1] * (ns / ns[-1]) ** 1.5, "k--", lw=1, label=r"$\mathcal{O}(n^{3/2})$")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("time [s]")
    ax.legend(fontsize=10)
    out = os.path.join(FIG_DIR, "time_vs_n.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure saved to {os.path.normpath(out)}")

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    for m in methods:
        ax.loglog(ns, 1e3 * T[m] / K[m], markers[m], label=m)
    ax.loglog(ns, 1e3 * T["SSOR-PCG"][-1] / K_pcg[-1] * ns / ns[-1], "k--", lw=1, label=r"$\mathcal{O}(n)$")
    ax.loglog(ns, 1e3 * T["CG"][-1] / K_cg[-1] * ns / ns[-1], "k--", lw=1)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("time / iteration [ms]")
    ax.legend(fontsize=10)
    out = os.path.join(FIG_DIR, "time_per_iteration.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure saved to {os.path.normpath(out)}")

if __name__ == "__main__":
    main()
