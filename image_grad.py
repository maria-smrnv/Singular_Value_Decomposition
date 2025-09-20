import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

gray_photo = cv2.imread("corgi.jpg", cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
print('start')

m, n = gray_photo.shape
r = min(m, n)

A = torch.tensor(gray_photo, dtype=torch.double).to(device)
U = torch.nn.Parameter(torch.randn(m, r, dtype=torch.double, device=device))
raw_S = torch.nn.Parameter(torch.randn(r, dtype=torch.double, device=device))
Vh = torch.nn.Parameter(torch.randn(r, n, dtype=torch.double, device=device))

opt = torch.optim.Adam([U, raw_S, Vh], lr=1e-2)

def get_S_from_raw(raw):
    return torch.nn.functional.softplus(raw)

# SVD
epochs = 10000
I_r = torch.eye(r, dtype=A.dtype, device=device)
for epoch in range(epochs):
    opt.zero_grad()
    S = get_S_from_raw(raw_S)
    US = U * S
    A_pred = US @ Vh

    recon_loss = torch.norm(A - A_pred, p='fro')**2
    U_loss = torch.norm(U.T @ U - I_r, p='fro')**2
    V_loss = torch.norm(Vh @ Vh.T - I_r, p='fro')**2

    loss = recon_loss + U_loss + V_loss

    loss.backward()
    opt.step()
    if epoch % 500 == 0:
        with torch.no_grad():
            Q_u, R_u = torch.linalg.qr(U)
            U.data.copy_(Q_u)
            Q_v, R_v = torch.linalg.qr(Vh.T)
            Vh.data.copy_(Q_v.T)
        print(f"Epoch {epoch:4d} | Loss {loss.item():.6f}")


with torch.no_grad():
    S_final = get_S_from_raw(raw_S)
    sorted_indices = torch.argsort(S_final, descending=True)

    U_sorted = U[:, sorted_indices].cpu()
    S_sorted = S_final[sorted_indices].cpu()
    Vh_sorted = Vh[sorted_indices, :].cpu()

# Energy Metrics
total_energy = torch.sum(S_sorted ** 2)
r_values = [1, 5, 10, 50, 100, 200]

fig, axes = plt.subplots(len(r_values), 2, figsize=(10, 3*len(r_values)))

for i, r in enumerate(r_values):
    U_r = U_sorted[:, :r]
    S_r = S_sorted[:r]
    Vh_r = Vh_sorted[:r, :]
    approx_tensor = U_r @ torch.diag(S_r) @ Vh_r
    approx = approx_tensor.numpy()
    approx = np.clip(approx, 0, 1)

    # Metrics
    explained_energy = torch.sum(S_r ** 2) / total_energy
    compression_ratio = (U_r.numel() + S_r.numel() + Vh_r.numel()) / gray_photo.size

    axes[i, 0].imshow(approx, cmap="gray")
    axes[i, 0].set_title(
        f"Approx r={r}\n$\\eta$={explained_energy*100:.2f}% | СR={compression_ratio:.4f}"
    )
    axes[i, 0].axis("off")

    axes[i, 1].imshow(gray_photo, cmap="gray")
    axes[i, 1].set_title("Original")
    axes[i, 1].axis("off")

    fig_single, ax_single = plt.subplots(1, 1, figsize=(6, 6))
    ax_single.imshow(approx, cmap="gray")
    ax_single.set_title(f"Approx r={r}")
    ax_single.axis("off")
    plt.tight_layout()
    plt.savefig(f"compressed_images/approx_r_{r}.png", dpi=150, bbox_inches='tight')
    plt.close(fig_single)

plt.tight_layout()
plt.savefig("compressed_images/comparison.png", dpi=300, bbox_inches='tight')
plt.show()