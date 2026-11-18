"""Implementation question 2: discretization error of the direct solve.

Solves A^h u^h = f^h with a sparse direct solver for h = 2^-q and plots
||u^h - u^h_ex||_inf against h.  Usage:  python Ex2/ex2.py
"""
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Ex1"))
from ex1 import assemble, u_exact  # noqa: E402

Q_LEVELS = range(2, 10)  # h = 1/4, ..., 1/512
FIG_DIR = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(FIG_DIR, "..", "latex.mplstyle"))  # Computer Modern fonts


def main():
    hs, errs = [], []
    print(f"{'q':>2} {'N':>5} {'n':>8} {'h':>10} {'error_inf':>12} {'order':>6} {'t_solve[s]':>10}")
    for q in Q_LEVELS:
        N = 2**q
        h = 1.0 / N
        A, f, (X, Y, _) = assemble(N)

        t0 = time.perf_counter()
        u = spsolve(A, f)  # SuperLU sparse direct solver
        t_solve = time.perf_counter() - t0

        err = np.max(np.abs(u - u_exact(X, Y)))
        order = np.log2(errs[-1] / err) if errs else np.nan
        hs.append(h)
        errs.append(err)
        print(f"{q:>2} {N:>5} {A.shape[0]:>8} {h:>10.3e} {err:>12.4e} {order:>6.3f} {t_solve:>10.3f}")

    hs, errs = np.array(hs), np.array(errs)
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    print(f"\nLeast-squares slope of log(error) vs log(h): {slope:.3f}")

    # ---------- Plot ----------
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots()
    ax.loglog(hs, errs, "o-", label="sparse direct solver")
    ax.loglog(hs, errs[0] * (hs / hs[0]) ** 2, "k--", label=r"$\mathcal{O}(h^2)$")
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\|u^h-u^h_{\mathrm{ex}}\|_\infty$")
    ax.grid(True, which="both")
    ax.legend()
    out = os.path.join(FIG_DIR, "discretization_error.pdf")
    fig.savefig(out)
    print(f"Figure saved to {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
