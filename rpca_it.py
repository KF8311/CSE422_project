# RPCA via Iterative Thresholding
import numpy as np
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd


def soft_threshold(X, eps):
    """Element-wise soft-thresholding operator S_eps[X] (Eq. 5)."""
    return np.sign(X) * np.maximum(np.abs(X) - eps, 0.0)


def it_rpca(
    D,
    lambda_param=None,
    tau=None,
    delta=None,
    tol=1e-7,
    max_iter=500,
    n_components=5,
    use_randomized_svd=True,
):
    """
    Algorithm 1: RPCA via Iterative Thresholding from Lin et al. (2010)
    """
    m, n = D.shape
    if lambda_param is None:
        # Paper uses lambda = m^(-1/2) for square matrices (or 1/sqrt(max(m, n)))
        lambda_param = (1.0 / np.sqrt(max(m, n))) * 2.0

    # Algorithm 1 requires tau (large positive scalar) and step size delta_k.
    # Note: The paper does not specify fixed defaults for tau and delta_k,
    # noting that choosing delta_k for speedup is difficult.
    if tau is None:
        tau = 10.0 * np.linalg.norm(D, ord=2)
    if delta is None:
        delta = 1.2

    # Initialize Y_0, A_0, E_0
    Y = np.zeros((m, n))
    A = np.zeros((m, n))
    E = np.zeros((m, n))

    D_norm = np.linalg.norm(D, "fro")

    with tqdm(
        total=max_iter,
        desc="IT-RPCA",
        unit="iter",
        dynamic_ncols=True,
        mininterval=0.1,
    ) as progress:
        for k in range(1, max_iter + 1):
            # Step 2: (U, S, V) = svd(Y_{k-1})
            progress.set_postfix(stage="SVD", refresh=True)
            if use_randomized_svd:
                U, S, Vt = randomized_svd(
                    Y, n_components=n_components, n_iter=2, random_state=None
                )
            else:
                U, S, Vt = np.linalg.svd(Y, full_matrices=False)

            # Step 3: A_k = U * S_tau[S] * V^T
            s_thresh = np.maximum(S - tau, 0.0)
            A_next = (U * s_thresh) @ Vt

            # Step 4: E_k = S_{lambda * tau}[Y_{k-1}]
            E_next = soft_threshold(Y, lambda_param * tau)

            # Step 5: Y_k = Y_{k-1} + delta_k * (D - A_k - E_k)
            residual_mat = D - A_next - E_next
            Y_next = Y + delta * residual_mat

            residual = np.linalg.norm(residual_mat, "fro") / D_norm
            progress.set_postfix(residual=f"{residual:.6e}", refresh=False)
            progress.update(1)

            if residual < tol:
                progress.set_postfix_str(f"converged residual={residual:.6e}")
                progress.close()
                A, E = A_next, E_next
                break

            A, E, Y = A_next, E_next, Y_next

    return A, E
