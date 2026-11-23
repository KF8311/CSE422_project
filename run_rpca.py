from datetime import datetime
import logging
from pathlib import Path
import time
import tracemalloc

from matplotlib.animation import FFMpegWriter, FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

from rpca_apg import apg_rpca
from rpca_ialm import ialm_rpca
from rpca_it import it_rpca


def setup_logger(log_path: str = "rpca_run.log") -> logging.Logger:
    logger = logging.getLogger("rpca_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def summarize_matrix(name: str, arr: np.ndarray) -> str:
    return (
        f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={np.min(arr):.6g}, max={np.max(arr):.6g}, "
        f"mean={np.mean(arr):.6g}, std={np.std(arr):.6g}, "
        f"fro_norm={np.linalg.norm(arr, ord='fro'):.6g}"
    )


def main() -> None:
    log_file = "rpca_run.log"
    logger = setup_logger(log_file)

    # --- 1. LOAD AND NORMALIZE DATA ---
    WIDTH = 240
    HEIGHT = 160
    MAX_FRAMES = 200  # Set to an integer for a shorter test run, or None for all frames.
    INPUT_PATH = "video_matrix.npy"
    OUTPUT_MP4 = "rpca_result.mp4"
    FG_CLIP_SCALE = 2.0
    FG_THRESHOLD = 0.2
    FPS = 25

    algo_name = "ialm_rpca"
    algo_kwargs = {}

    logger.info("=== RPCA Experiment Start (%s) ===", datetime.now().isoformat())
    logger.info(
        "Config: WIDTH=%d, HEIGHT=%d, MAX_FRAMES=%s, INPUT_PATH=%s, "
        "OUTPUT_MP4=%s, FG_CLIP_SCALE=%.4f, FG_THRESHOLD=%.4f, FPS=%d, algo=%s, algo_kwargs=%s",
        WIDTH,
        HEIGHT,
        MAX_FRAMES,
        INPUT_PATH,
        OUTPUT_MP4,
        FG_CLIP_SCALE,
        FG_THRESHOLD,
        FPS,
        algo_name,
        algo_kwargs,
    )

    if WIDTH * HEIGHT != 38400:
        logger.warning(
            "WIDTH * HEIGHT (%d) != 38400; verify reshape dimensions match D.shape[0].",
            WIDTH * HEIGHT,
        )

    logger.info("Loading %s...", INPUT_PATH)
    D_raw = np.load(INPUT_PATH)
    logger.info(summarize_matrix("D_raw", D_raw))

    D = D_raw.astype(np.float32) / 255.0
    if MAX_FRAMES is not None:
        D = D[:, :MAX_FRAMES]

    m, n = D.shape
    default_lambda = 1.0 / np.sqrt(max(m, n))
    logger.info(summarize_matrix("D_normalized", D))
    logger.info("Derived m=%d, n=%d, default_lambda=%.6g", m, n, default_lambda)

    # --- 2. RUN SELECTED RPCA ALGORITHM ---
    tracemalloc.start()
    t_start = time.perf_counter()

    # # Algorithm 1: RPCA via Iterative Thresholding from Lin et al. (2010)
    # algo_name = "it_rpca"
    # algo_kwargs = {}
    # logger.info("Running IT-RPCA background subtraction...")
    # A, E = it_rpca(D, **algo_kwargs)

    # # Algorithm 2: RPCA via APG from Lin et al. (2010)
    # algo_name = "apg_rpca"
    # algo_kwargs = {}
    # logger.info("Running APG-RPCA background subtraction...")
    # A, E = apg_rpca(D, **algo_kwargs)

    # Algorithm 5: RPCA via Inexact ALM from Lin et al. (2010)
   
    lambda_param = 1.0 / np.sqrt(max(m, n))
    TEST_KWARGS = {
    "lambda_param": 2.0 * lambda_param ,   
    "mu0": None,               # Uses default heuristic when None
    "rho": 1.5,
    "mu_max": 1e7,
    "tol": 1e-7,
    "max_iter": 500,
    "use_randomized_svd": True,
    "n_components": 10,
    "n_iter_svd": 2,
    "random_state": 42,        # Fixed seed for reproducible comparisons
    }
    algo_name = "ialm_rpca"
    algo_kwargs = {}
    logger.info("Running IALM-RPCA background subtraction...")
    A, E = ialm_rpca(D, **TEST_KWARGS)

    elapsed_rpca = time.perf_counter() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    recon_err = D - A - E
    rel_recon_err = np.linalg.norm(recon_err, ord="fro") / max(
        np.linalg.norm(D, ord="fro"), 1e-12
    )
    a_rank_approx = int(np.linalg.matrix_rank(A))
    e_nonzero_ratio = float(np.mean(np.abs(E) > 1e-6))

    logger.info(
        "RPCA finished in %.3fs (%.4fs/frame) | current_mem=%.2f MB | peak_mem=%.2f MB",
        elapsed_rpca,
        elapsed_rpca / max(n, 1),
        current_mem / (1024**2),
        peak_mem / (1024**2),
    )
    logger.info(summarize_matrix("A (Low-Rank Background)", A))
    logger.info(summarize_matrix("E (Sparse Foreground)", E))
    logger.info(
        "Decomposition metrics: rel_recon_err=%.6e, A_matrix_rank=%d, "
        "E_abs_gt_1e-6_ratio=%.6f (%.2f%%)",
        rel_recon_err,
        a_rank_approx,
        e_nonzero_ratio,
        e_nonzero_ratio * 100.0,
    )

    # --- 3. POST-PROCESS FOREGROUND & SAVE ANIMATION ---
    foreground = np.abs(E)
    fg_max_before = float(foreground.max())
    if fg_max_before > 0.0:
        foreground = np.clip((foreground / fg_max_before) * FG_CLIP_SCALE, 0.0, 1.0)
    else:
        foreground = np.zeros_like(foreground)

    fg_active_before_thresh = int(np.sum(foreground >= FG_THRESHOLD))
    foreground[foreground < FG_THRESHOLD] = 0.0
    fg_active_after_thresh = int(np.sum(foreground > 0.0))

    logger.info(summarize_matrix("foreground (post-processed)", foreground))
    logger.info(
        "Foreground stats: fg_max_before=%.6g, active_pixels_ge_thresh=%d, "
        "active_pixels_after_zeroing=%d / %d (%.2f%%)",
        fg_max_before,
        fg_active_before_thresh,
        fg_active_after_thresh,
        foreground.size,
        (fg_active_after_thresh / max(foreground.size, 1)) * 100.0,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    original_image = axes[0].imshow(
        D[:, 0].reshape(HEIGHT, WIDTH), cmap="gray", vmin=0, vmax=1
    )
    background_image = axes[1].imshow(
        A[:, 0].reshape(HEIGHT, WIDTH), cmap="gray", vmin=0, vmax=1
    )
    foreground_image = axes[2].imshow(
        foreground[:, 0].reshape(HEIGHT, WIDTH),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title("Original")
    axes[1].set_title("Low-Rank Background")
    axes[2].set_title("Sparse Foreground")
    title_text = fig.suptitle(f"Frame 1/{n}")
    for axis in axes:
        axis.axis("off")

    def update(frame_index: int):
        original_image.set_data(D[:, frame_index].reshape(HEIGHT, WIDTH))
        background_image.set_data(A[:, frame_index].reshape(HEIGHT, WIDTH))
        foreground_image.set_data(foreground[:, frame_index].reshape(HEIGHT, WIDTH))
        title_text.set_text(f"Frame {frame_index + 1}/{n}")
        return original_image, background_image, foreground_image, title_text

    # Note: blit=False avoids stale suptitle/artist artifacts when saving MP4
    animation = FuncAnimation(
        fig,
        update,
        frames=n,
        interval=int(1000 / FPS),
        blit=False,
        repeat=True,
    )

    plt.tight_layout()

    t_anim_start = time.perf_counter()
    logger.info("Saving animation to %s...", OUTPUT_MP4)
    if not FFMpegWriter.isAvailable():
        raise RuntimeError(
            "FFmpeg is required to save the animation as MP4. "
            "Install FFmpeg and make sure its executable is on PATH."
        )
    animation.save(OUTPUT_MP4, writer=FFMpegWriter(fps=FPS))
    elapsed_anim = time.perf_counter() - t_anim_start
    mp4_size_mb = (
        Path(OUTPUT_MP4).stat().st_size / (1024**2)
        if Path(OUTPUT_MP4).exists()
        else 0.0
    )
    logger.info(
        "Animation saved in %.3fs | file_size=%.2f MB | log_file=%s",
        elapsed_anim,
        mp4_size_mb,
        log_file,
    )

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
