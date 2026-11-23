import numpy as np
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd


def soft_threshold(X, eps):
    """Element-wise soft-thresholding operator S_eps[X]."""
    return np.sign(X) * np.maximum(np.abs(X) - eps, 0.0)


def _J_norm(D, lam):
    """Normalization factor J(D) = max(||D||_2, lam^{-1} * ||D||_inf)."""
    spectral_norm = np.linalg.norm(D, ord=2)
    inf_norm = np.max(np.abs(D))
    return max(spectral_norm, inf_norm / lam)


def ialm_rpca(
    D,
    lambda_param=None,
    mu0=None,
    rho=1.5,
    mu_max=1e7,
    tol=1e-7,
    max_iter=500,
    use_randomized_svd=True,
    n_components=10,
    n_iter_svd=2,
    random_state=None,
    return_info=False,
):
    """
    Algorithm 5: RPCA via the Inexact ALM Method from Lin et al. (2010).
    """
    D = np.asarray(D, dtype=np.float64)
    m, n = D.shape

    if lambda_param is None:
        lambda_param = 1.0 / np.sqrt(max(m, n))

    # Line 1: Initialization
    norm_two = np.linalg.norm(D, ord=2)
    if mu0 is None:
        # Standard heuristic in Lin et al. (2010)
        mu = 1.25 / (norm_two + 1e-12)
    else:
        mu = float(mu0)

    J_D = _J_norm(D, lambda_param)
    Y = D / (J_D + 1e-12)
    E = np.zeros((m, n), dtype=np.float64)
    A = np.zeros((m, n), dtype=np.float64)

    D_norm = np.linalg.norm(D, ord="fro")
    if D_norm == 0.0:
        result = (A.copy(), E.copy())
        if return_info:
            return (*result, {"iterations_run": 0, "converged": True})
        return result

    iterations_run = 0
    converged = False

    with tqdm(
        total=max_iter,
        desc="IALM-RPCA",
        unit="iter",
        dynamic_ncols=True,
        mininterval=0.1,
    ) as progress:
        for _ in range(1, max_iter + 1):
            iterations_run += 1
            inv_mu = 1.0 / mu

            # Lines 4-5: Solve A_{k+1} = arg min_A L(A, E_k, Y_k, mu_k)
            progress.set_postfix(stage="SVD", refresh=True)
            svd_input = D - E + inv_mu * Y

            if use_randomized_svd:
                k_rank = min(n_components, min(m, n) - 1) if min(m, n) > 1 else 1
                U, S, Vt = randomized_svd(
                    svd_input,
                    n_components=k_rank,
                    n_iter=n_iter_svd,
                    random_state=random_state,
                )
            else:
                U, S, Vt = np.linalg.svd(svd_input, full_matrices=False)

            s_thresh = np.maximum(S - inv_mu, 0.0)
            A_next = (U * s_thresh) @ Vt

            # Line 7: Solve E_{k+1} = arg min_E L(A_{k+1}, E, Y_k, mu_k)
            E_next = soft_threshold(D - A_next + inv_mu * Y, lambda_param * inv_mu)

            # Line 8: Update Lagrange multiplier Y_{k+1}
            residual_mat = D - A_next - E_next
            Y_next = Y + mu * residual_mat

            # Convergence check (relative primal residual)
            rel_err = np.linalg.norm(residual_mat, ord="fro") / D_norm
            progress.set_postfix(err=f"{rel_err:.2e}", mu=f"{mu:.2e}", refresh=False)
            progress.update(1)

            A, E, Y = A_next, E_next, Y_next

            if rel_err < tol:
                converged = True
                progress.set_postfix_str(f"converged err={rel_err:.2e}")
                progress.close()
                break

            # Line 9: Update mu_k -> mu_{k+1}
            mu = min(rho * mu, mu_max)

    if return_info:
        return A, E, {"iterations_run": iterations_run, "converged": converged}
    return A, E
