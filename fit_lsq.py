import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, nargs="+", default=[12])
args = parser.parse_args()

Y = torch.load(f"lsq_data/Y.pt").to('cuda')

for layer in args.layer:
    X = torch.load(f"lsq_data/X_l{layer}.pt").to('cuda')
    λ = 1e-4                              # tiny ridge makes inverse stable
    Xt = X.T                              # (d,N)
    W  = (Y.T @ X) @ torch.linalg.inv(Xt @ X + λ * torch.eye(X.shape[1]).to('cuda')) # (V,d)
    mse = torch.mean((X @ W.T - Y) ** 2)
    print(f"layer: {layer}, shape: {W.shape}, mse: {mse}")
    torch.save(W.to(torch.bfloat16), f"lsq_data/weights/ee_head_l{layer}.pt")