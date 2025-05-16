# quick_fit_l12.py
import torch, glob, tqdm, argparse, os
p = argparse.ArgumentParser()
p.add_argument('--act_dir', required=True)
p.add_argument('--out', default='h12.pt')
p.add_argument('--layer', type=int, default=12)
p.add_argument('--lam', type=float, default=1e-4)
args = p.parse_args()

shards = sorted(glob.glob(f'{args.act_dir}/b*.pt'))
d   = torch.load(shards[0])[f'h{args.layer}'].shape[-1]
V   = torch.load(shards[0])['logits'].shape[-1]
XtX = torch.zeros(d, d,  device='cuda', dtype=torch.float16)
YtX = torch.zeros(V, d,  device='cuda', dtype=torch.float16)

for path in tqdm.tqdm(shards, desc='accum'):
    blob = torch.load(path, map_location='cpu')
    X = blob[f'h{args.layer}'].to(dtype=torch.float16, device='cuda')  # (B,d)
    Y = blob['logits'].to(dtype=torch.float16, device='cuda')          # (B,V)
    XtX += X.T @ X
    YtX += Y.T @ X

A = (XtX + args.lam * torch.eye(d, device='cuda', dtype=torch.float16)).to(torch.float32)
B = YtX.to(torch.float32).T         # (d, V)

W = torch.linalg.solve(A, B).T      # (V, d)  fp32 solve on GPU
torch.save(W.to(torch.bfloat16).cpu(), args.out)
print('saved →', args.out)

