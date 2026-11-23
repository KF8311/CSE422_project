# RPCA via Accelerated Proximal Gradient
import numpy as np
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd


def apg_rpca(
    D,
    lambda_param=None,
    mu_bar=1e-5,
    eta=0.9,
    tol=1e-7,
    max_iter=500,
    use_randomized_svd=True,
):
    """
    Algorithm 2: RPCA via Accelerated Proximal Gradient from Lin et al. (2010)
    """
    m, n = D.shape
    if lambda_param is None:
        # lambda_param = 1.0 / np.sqrt(
        #     max(m, n)
        # )  # Recommended default value for regularization parameter lambda by the paper.
        lambda_param = (1.0 / np.sqrt(max(m, n))) * 2.0
    A_prev = np.zeros((m, n))
    A = np.zeros((m, n))
    E_prev = np.zeros((m, n))
    E = np.zeros((m, n))
    t_prev = 1.0
    t = 1.0
    mu = 1.0
    with tqdm(
        total=max_iter,
        desc="APG-RPCA",
        unit="iter",
        dynamic_ncols=True,
        mininterval=0.1,
    ) as progress:
        for k in range(1, max_iter + 1):
            progress.set_postfix(stage="forming G", refresh=True)
            factor = (t_prev - 1) / t
            Y_A = A + factor * (A - A_prev)
            Y_E = E + factor * (E - E_prev)
            G_A = Y_A - 0.5 * (Y_A + Y_E - D)
            progress.set_postfix(stage="SVD", refresh=True)
            if use_randomized_svd:
                U, S, Vt = randomized_svd(
                    G_A, n_components=5, n_iter=2, random_state=None
                )
            else:
                U, S, Vt = np.linalg.svd(G_A, full_matrices=False)
            threshold_A = mu / 2.0
            s_thresh = np.maximum(S - threshold_A, 0)
            A_next = np.dot(U, np.dot(np.diag(s_thresh), Vt))
            G_E = Y_E - 0.5 * (Y_A + Y_E - D)
            threshold_E = (lambda_param * mu) / 2.0
            E_next = np.sign(G_E) * np.maximum(np.abs(G_E) - threshold_E, 0)
            t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2.0
            mu_next = max(eta * mu, mu_bar)
            residual = np.linalg.norm(D - A_next - E_next, "fro") / np.linalg.norm(
                D, "fro"
            )
            progress.set_postfix(residual=f"{residual:.6e}", refresh=False)
            progress.update(1)

            if residual < tol:
                progress.set_postfix_str(f"converged residual={residual:.6e}")
                progress.close()
                A, E = A_next, E_next
                break

            A_prev, E_prev = A, E
            A, E = A_next, E_next
            t_prev, t = t, t_next
            mu = mu_next

    return A, E
