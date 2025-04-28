import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast

activations = torch.load("activations_399.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
activations = {k: v.to(device) for k, v in activations.items()}

Xs = activations['post_attention_layernorm_46']
Ys = activations['post_attention_layernorm_47']

n = int(Xs.shape[0]*0.8)
Xs_train = Xs[:n]
Ys_train = Ys[:n]
Xs_val = Xs[n:]
Ys_val = Ys[n:]

class ActivationMapping(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(x)

model = ActivationMapping(dim=activations['post_attention_layernorm_46'].shape[1]).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
criterion = nn.MSELoss()

BATCH_SIZE = 8192
NUM_EPOCHS = 30000

for epoch in range(NUM_EPOCHS):
    idx = torch.randint(0, Xs_train.shape[0], (BATCH_SIZE,))
    Xs = Xs_train[idx]
    Ys = Ys_train[idx]
    optimizer.zero_grad()
    with autocast(dtype=torch.bfloat16):
        pred = model(Xs)
        loss = criterion(pred, Ys)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(loss.item())

model.eval()
with torch.no_grad(): 
    with autocast(dtype=torch.bfloat16):
        pred = model(Xs_val)
        loss = criterion(pred, Ys_val)
        print(loss.item())

