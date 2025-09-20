import torch
torch.set_default_dtype(torch.double)
torch.manual_seed(0)

# Target matrix to approximate
A = torch.tensor([...])
m, n = A.shape
r = min(m, n)

# Initialize learnable parameters
U = torch.nn.Parameter(torch.randn(m, r))
raw_S = torch.nn.Parameter(torch.randn(r))
Vh = torch.nn.Parameter(torch.randn(r, n))
opt = torch.optim.Adam([U, raw_S, Vh], lr=1e-2)

def get_S_from_raw(raw):
    return torch.nn.functional.softplus(raw)

epochs = 10000
I_r = torch.eye(r, dtype=A.dtype)
for epoch in range(epochs):
    opt.zero_grad()

    # Reconstruction
    S = get_S_from_raw(raw_S)      # (r,)
    US = U * S                     # broadcasting: (m,r)
    A_pred = US @ Vh               # (m,n)

    recon_loss = torch.norm(A - A_pred, p='fro')**2
    U_loss = torch.norm(U.T @ U - I_r, p='fro')**2
    V_loss = torch.norm(Vh @ Vh.T - I_r, p='fro')**2
    loss = recon_loss + U_loss + V_loss

    loss.backward()
    opt.step()
    if epoch % 500 == 0:
        with torch.no_grad():
            Q_u, _ = torch.linalg.qr(U)
            U.data.copy_(Q_u)
            Q_v, _ = torch.linalg.qr(Vh.T)
            Vh.data.copy_(Q_v.T)
        print(f"epoch {epoch:4d} | loss {loss.item():.6f} | recon {recon_loss.item():.6f} | orthoU {U_loss.item():.6e} | orthoV {V_loss.item():.6e}")

# Results
S_final = get_S_from_raw(raw_S).detach()
A_final_pred = (U.detach() * S_final) @ Vh.detach()
final_recon_loss = torch.norm(A - A_final_pred, p='fro')**2

U_ortho_deviation = torch.norm(U.detach().T @ U.detach() - torch.eye(r), p='fro').item()
V = Vh.detach().T
V_ortho_deviation = torch.norm(V.T @ V - torch.eye(r), p='fro').item()

print("||A - A_pred ||_F:", final_recon_loss.item())
print("||U^T U - I||_F:", U_ortho_deviation)
print("||V^T V - I||_F:", V_ortho_deviation)

print("\nA:")
print(A)
print("\nA_pred r=min(m, n):")
print(A_final_pred)