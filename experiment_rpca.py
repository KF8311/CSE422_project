"""Run named IALM-RPCA parameter experiments on the video matrix.

Edit the TEST_* functions below to add or change parameter configurations, then
run ``python experiment_rpca.py`` (all tests) or ``python experiment_rpca.py test2``.
Each run writes its own log and video under ``experiments/``.
"""

import argparse
import csv
import logging
import tracemalloc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from rpca_ialm import ialm_rpca

# Reference:
# BASE_KWARGS = {
#     "lambda_param": None,   
#     "mu0": None,               # Uses default heuristic when None
#     "rho": 1.5,
#     "mu_max": 1e7,
#     "tol": 1e-7,
#     "max_iter": 500,
#     "use_randomized_svd": True,
#     "n_components": 10,
#     "n_iter_svd": 2,
#     "random_state": 42,        # Fixed seed for reproducible comparisons
#     } # Use for baseline comparisons

# Keep these settings aligned with run_rpca.py.
WIDTH = 240
HEIGHT = 160
MAX_FRAMES = 200  # Use None to process every frame in video_matrix.npy.
INPUT_PATH = Path("video_matrix.npy")
OUTPUT_DIR = Path("experiments")
FG_CLIP_SCALE = 2.0
FG_THRESHOLD = 0.2
FPS = 25


def test1(D: np.ndarray) -> dict:
    """Baseline: use all IALM defaults."""
    test_kwargs = {}
    return run_test(D, "test1", test_kwargs)


def test2(D: np.ndarray) -> dict:
    """Use half of the default lambda, matching the 0.5x-lambda experiment."""
    default_lambda = 1.0 / np.sqrt(max(D.shape))
    test_kwargs = {"lambda_param": 0.5 * default_lambda}
    return run_test(D, "test2", test_kwargs)


def test3(D: np.ndarray) -> dict:
    """Use twice the default lambda."""
    default_lambda = 1.0 / np.sqrt(max(D.shape))
    test_kwargs = {"lambda_param": 2.0 * default_lambda}
    return run_test(D, "test3", test_kwargs)


def test4(D: np.ndarray) -> dict:
    """Use tighter convergence tolerance and allow more iterations."""
    test_kwargs = {"tol": 1e-8, "max_iter": 1000}
    return run_test(D, "test4", test_kwargs)


TESTS = {"test1": test1, "test2": test2, "test3": test3, "test4": test4}


def make_logger(test_name: str, output_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"rpca_experiment.{test_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def run_test(D: np.ndarray, test_name: str, test_kwargs: dict) -> dict:
    """Run one named configuration and save its metrics and visualization."""
    test_dir = OUTPUT_DIR / test_name
    test_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(test_name, test_dir / f"{test_name}.txt")

    logger.info("Starting %s with kwargs=%s", test_name, test_kwargs)
    start = time.perf_counter()
    tracemalloc.start()
    try:
        A, E, solver_info = ialm_rpca(D, return_info=True, **test_kwargs)
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    elapsed = time.perf_counter() - start
    peak_mem_mb = peak_mem_bytes / (1024**2)

    rel_recon_err = np.linalg.norm(D - A - E, ord="fro") / max(
        np.linalg.norm(D, ord="fro"), 1e-12
    )
    rank = int(np.linalg.matrix_rank(A))
    e_nonzero_ratio = float(np.mean(np.abs(E) > 1e-6))

    foreground = np.abs(E)
    fg_max = float(foreground.max())
    if fg_max > 0:
        foreground = np.clip(foreground / fg_max * FG_CLIP_SCALE, 0.0, 1.0)
    else:
        foreground = np.zeros_like(foreground)
    foreground[foreground < FG_THRESHOLD] = 0.0
    active_ratio = float(np.mean(foreground > 0))

    video_path = test_dir / f"{test_name}.mp4"
    save_video(D, A, foreground, video_path)

    logger.info(
        "Finished in %.3fs; iterations_run=%d; converged=%s; "
        "peak_mem_mb=%.2f; rel_recon_err=%.6e; A_rank=%d; "
        "E_nonzero_ratio=%.6f; foreground_active_ratio=%.6f; video=%s",
        elapsed,
        solver_info["iterations_run"],
        solver_info["converged"],
        peak_mem_mb,
        rel_recon_err,
        rank,
        e_nonzero_ratio,
        active_ratio,
        video_path,
    )
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    logger.handlers.clear()

    return {
        "test": test_name,
        "kwargs": repr(test_kwargs),
        "seconds": elapsed,
        "iterations_run": solver_info["iterations_run"],
        "converged": solver_info["converged"],
        "peak_mem_mb": peak_mem_mb,
        "relative_reconstruction_error": rel_recon_err,
        "background_rank": rank,
        "sparse_nonzero_ratio": e_nonzero_ratio,
        "foreground_active_ratio": active_ratio,
        "video": str(video_path),
    }


def save_video(D, A, foreground, output_path: Path) -> None:
    if D.shape[0] != WIDTH * HEIGHT:
        raise ValueError(
            f"D has {D.shape[0]} rows, but WIDTH*HEIGHT is {WIDTH * HEIGHT}. "
            "Update WIDTH and HEIGHT to match the matrix."
        )
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("FFmpeg is required to save experiment videos.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    original_image = axes[0].imshow(D[:, 0].reshape(HEIGHT, WIDTH), cmap="gray", vmin=0, vmax=1)
    background_image = axes[1].imshow(A[:, 0].reshape(HEIGHT, WIDTH), cmap="gray", vmin=0, vmax=1)
    foreground_image = axes[2].imshow(
        foreground[:, 0].reshape(HEIGHT, WIDTH), cmap="gray", vmin=0, vmax=1
    )
    axes[0].set_title("Original")
    axes[1].set_title("Low-Rank Background")
    axes[2].set_title("Sparse Foreground")
    for axis in axes:
        axis.axis("off")
    title = fig.suptitle(f"Frame 1/{D.shape[1]}")

    def update(index):
        original_image.set_data(D[:, index].reshape(HEIGHT, WIDTH))
        background_image.set_data(A[:, index].reshape(HEIGHT, WIDTH))
        foreground_image.set_data(foreground[:, index].reshape(HEIGHT, WIDTH))
        title.set_text(f"Frame {index + 1}/{D.shape[1]}")
        return original_image, background_image, foreground_image, title

    animation = FuncAnimation(fig, update, frames=D.shape[1], interval=1000 / FPS, blit=False)
    fig.tight_layout()
    animation.save(output_path, writer=FFMpegWriter(fps=FPS))
    plt.close(fig)


def load_data() -> np.ndarray:
    raw = np.load(INPUT_PATH)
    D = raw.astype(np.float32) / 255.0
    if MAX_FRAMES is not None:
        D = D[:, :MAX_FRAMES]
    return D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tests",
        nargs="*",
        choices=TESTS.keys(),
        help="Tests to run (default: run all four).",
    )
    args = parser.parse_args()
    selected_tests = args.tests or list(TESTS)

    D = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [TESTS[name](D) for name in selected_tests]

    summary_path = OUTPUT_DIR / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Experiment summary saved to {summary_path}")


if __name__ == "__main__":
    main()
