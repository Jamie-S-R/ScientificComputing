"""Finite-difference discretization of the 2D Poisson problem on the unit square.

    -Laplace(u) = f  in (0,1)^2,      u = b  on the boundary.

Grid: N subintervals per direction, h = 1/N, nodes x_i = i*h, i = 0..N.
All (N+1)^2 nodes (boundary included) are unknowns, ordered lexicographically
with x running fastest:  k = i + j*(N+1)  <->  (x_i, y_j).
"""
import numpy as np
import scipy.sparse as sp


# ---------- Exact solution (pen-and-paper Q1) ----------
def u_exact(x, y):
    return np.sin(x * y)


def f_source(x, y):
    # -Laplace(sin(xy)) = (x^2 + y^2) sin(xy)
    return (x**2 + y**2) * np.sin(x * y)


def b_boundary(x, y):
    return u_exact(x, y)


# ---------- Grid ----------
def grid(N):
    """Nodal coordinates (flattened, lexicographic) and boundary mask."""
    x = np.linspace(0.0, 1.0, N + 1)
    X, Y = np.meshgrid(x, x, indexing="xy")  # X[j, i] = x_i, Y[j, i] = y_j
    X, Y = X.ravel(), Y.ravel()               # k = i + j*(N+1)
    on_bnd = (X == 0.0) | (X == 1.0) | (Y == 0.0) | (Y == 1.0)
    return X, Y, on_bnd


# ---------- Assembly (implementation question 1) ----------
def laplacian_1d(N):
    """1D operator (1/h^2) tridiag(-1, 2, -1) on all N+1 nodes."""
    h = 1.0 / N
    e = np.ones(N + 1)
    return sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1], format="csr") / h**2


def assemble(N, f=f_source, b=b_boundary):
    """Return A^h (CSR, SPD), f^h and the grid (X, Y, on_bnd).

    1. 2D operator via tensor products: A = I (x) T + T (x) I.
    2. Dirichlet conditions, keeping the boundary unknowns:
       - move the known boundary values to the right-hand side,
         f_I <- f_I - A_IB b_B   (symmetric elimination),
       - zero the boundary rows AND columns and put 1 on their diagonal,
         so the boundary equations read u_k = b_k.
       Result: A = [[A_II, 0], [0, I]] (up to ordering), which is SPD.
    """
    T = laplacian_1d(N)
    I1 = sp.identity(N + 1, format="csr")
    A = sp.kron(I1, T) + sp.kron(T, I1)  # 5-point stencil, lexicographic order

    X, Y, on_bnd = grid(N)
    interior = (~on_bnd).astype(float)

    # Right-hand side: f at interior nodes, b at boundary nodes
    u_B = np.where(on_bnd, b(X, Y), 0.0)
    rhs = np.where(on_bnd, u_B, f(X, Y) - A @ u_B)

    # Zero boundary rows/columns, identity on boundary diagonal
    P = sp.diags(interior)
    A = (P @ A @ P + sp.diags(on_bnd.astype(float))).tocsr()
    A.eliminate_zeros()
    return A, rhs, (X, Y, on_bnd)
