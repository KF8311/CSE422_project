import cv2
import glob
import numpy as np

# Load frames into a 3D NumPy array (Height x Width x Frames)
frame_paths = sorted(
    glob.glob("dataset/PETS09-S2L1/img1/*.jpg")
)  # or downsample here
frames = []
WIDTH = 240
HEIGHT = 160
for path in frame_paths:
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  # Downsample for efficiency as noted in your proposal
  img_resized = cv2.resize(img, (WIDTH, HEIGHT))
  frames.append(img_resized)

# Convert to matrix D where each column is a flattened frame: (Pixels x Frames)
video_matrix = np.column_stack([f.flatten() for f in frames])

# 1. Print out matrix details to verify its structure
print("--- Video Matrix Summary ---")
print(f"Shape (Pixels x Frames): {video_matrix.shape}")
print(f"Data Type: {video_matrix.dtype}")
print(f"Memory Size: {video_matrix.nbytes / (1024 * 1024):.2f} MB")
print(
    f"Pixel Value Range: Min={video_matrix.min():.2f}, Max={video_matrix.max():.2f}"
)

# 2. Store the matrix to 'video_matrix.npy'
output_filename = "video_matrix.npy"

print(f"Saving matrix to {output_filename}...")
np.save(output_filename, video_matrix)
print(f"Successfully saved to {output_filename}!")