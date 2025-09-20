import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import svd

gray_photo = cv2.imread("corgi.jpg", cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
print('start')

# SVD
U, S, V_T = svd(gray_photo, full_matrices=False)

# Energy Metrics
total_energy = np.sum(S**2)

r_values = [1, 5, 10, 50, 100, 200]

fig, axes = plt.subplots(len(r_values), 2, figsize=(10, 3*len(r_values)))

for i, r in enumerate(r_values):
    approx = U[:, :r] @ np.diag(S[:r]) @ V_T[:r, :]

    # Metrics
    explained_energy = np.sum(S[:r]**2) / total_energy
    compression_ratio = (U[:, :r].size + S[:r].size + V_T[:r, :].size) / gray_photo.size

    axes[i, 0].imshow(approx, cmap="gray")
    axes[i, 0].set_title(
        f"Approx r={r}\n$\\eta$={explained_energy*100:.2f}% | СR={compression_ratio:.4f}"
    )
    axes[i, 0].axis("off")

    axes[i, 1].imshow(gray_photo, cmap="gray")
    axes[i, 1].set_title("Original")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()
