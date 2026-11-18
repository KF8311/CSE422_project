from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
import numpy as np

from rpca_apg import apg_rpca

# Import your RPCA function (save the apg_rpca function in a file named rpca.py, or paste it above)
# from rpca import apg_rpca


# --- 1. LOAD AND NORMALIZE DATA ---
WIDTH = 240
HEIGHT = 160

print("Loading video_matrix.npy...", flush=True)
D = np.load("video_matrix.npy").astype(np.float32)

# Normalize pixel values from [0, 255] to [0.0, 1.0] for stable matrix convergence
D = D / 255.0
MAX_FRAMES = 200  # Set to an integer for a shorter test run.
if MAX_FRAMES is not None:
    D = D[:, :MAX_FRAMES]
print(f"Data matrix shape (Pixels x Frames): {D.shape}", flush=True)


# --- 2. RUN APG-RPCA ---
print(
    "Running APG-RPCA background subtraction (this may take a minute depending"
    " on frame count)..."
    ,
    flush=True,
)
A, E = apg_rpca(D)

# --- 3. PLAY RESULTS AS ANIMATION ---
# Take absolute value and binarize/threshold out the remaining background noise
foreground = np.abs(E)
foreground = np.clip(foreground / foreground.max() * 2.0, 0, 1)  # Scale and clip intensity
foreground[foreground < 0.2] = 0  # <--- MINIMAL CHANGE: Zero out anything below 20% intensity to kill background gray

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
    vmax=1,  # <--- Change vmax to 1.0 since we normalized it
)

axes[0].set_title("Original")
axes[1].set_title("Low-Rank Background")
axes[2].set_title("Sparse Foreground")
for axis in axes:
    axis.axis("off")

def update(frame_index):
    original_image.set_data(D[:, frame_index].reshape(HEIGHT, WIDTH))
    background_image.set_data(A[:, frame_index].reshape(HEIGHT, WIDTH))
    foreground_image.set_data(foreground[:, frame_index].reshape(HEIGHT, WIDTH))
    fig.suptitle(f"Frame {frame_index + 1}/{D.shape[1]}")
    return original_image, background_image, foreground_image


animation = FuncAnimation(
    fig,
    update,
    frames=D.shape[1],
    interval=40,
    blit=True,
    repeat=True,
)
plt.tight_layout()
plt.show()