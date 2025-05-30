import torch, time, numpy as np
DUMP = torch.load("lsq_data_l12.pt")
X, Y  = DUMP["X"], DUMP["Y"]          # (N,d)  (N,V)
print("dataset", X.shape, Y.shape)

λ = 1e-4                              # tiny ridge makes inverse stable
t0 = time.time()
Xt = X.T                              # (d,N)
W  = (Y.T @ X) @ torch.linalg.inv(Xt @ X + λ * torch.eye(X.shape[1]))
print("fit done in", time.time()-t0, "s", W.shape)    # (V,d)
torch.save(W.to(torch.bfloat16), "ee_head_l12_l4.pt")
