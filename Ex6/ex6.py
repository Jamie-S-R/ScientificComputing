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

    # ---------- Plot ----------
    # Stacked panels with a common k-axis, each panel keeps the 10:4 aspect ratio
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(7.0, 5.6))
    for ax, name, tag in zip(axes, methods, "ab"):
        for N, n, _, _, hist in results[name]:
            (line,) = ax.semilogy(hist, label=rf"$N={N},\ n={n}$")
            ax.plot(len(hist) - 1, hist[-1], "o", color=line.get_color())  # converged iterate
        ax.axhline(TOL, color="k", ls=":", lw=1)
        ax.set_title(f"({tag}) {name}")
        ax.set_ylabel(r"$\|\mathbf{r}^k\|_2/\|\mathbf{f}^h\|_2$")
    axes[-1].set_xlabel(r"$k$")
    axes[1].legend(fontsize=10, loc="center right", title=r"$n=(N+1)^2$", title_fontsize=10)
    out = os.path.join(FIG_DIR, "cg_pcg_convergence.pdf")
    fig.savefig(out)
    print(f"Figure saved to {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
