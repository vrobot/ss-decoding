import torch
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, nargs="+", default=[12])
args = parser.parse_args()

X = torch.load(f"lsq_data/X_l1.pt").to('cuda')
Y = torch.load(f"lsq_data/Y.pt").to('cuda')
eye_matrix = torch.eye(X.shape[1], device='cuda')

mses = {}

for layer in args.layer:
    X = torch.load(f"lsq_data/X_l{layer}.pt").to('cuda')
    λ = 1e-4                              # tiny ridge makes inverse stable
    Xt = X.T                              # (d,N)
    W  = (Y.T @ X) @ torch.linalg.inv(Xt @ X + λ * eye_matrix) # (V,d)
    mse = torch.mean((X @ W.T - Y) ** 2)
    print(f"layer: {layer}, shape: {W.shape}, mse: {mse}")
    mses[layer] = mse.item()
    torch.save(W.to(torch.bfloat16).cpu(), f"lsq_data/weights/ee_head_l{layer}.pt")
    del X, W
    torch.cuda.empty_cache()

torch.save(mses, f"lsq_data/weights/mses.pth")
plt.plot(list(mses.keys()), list(mses.values()))
plt.savefig(f"lsq_data/weights/mses.png")
