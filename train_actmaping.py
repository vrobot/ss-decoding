import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast

activations = torch.load("testactivations_8192.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
activations = {k: v.to(device) for k, v in activations.items()}

class ActivationMapping(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(x)

model = ActivationMapping(dim=activations['post_attention_layernorm_46'].shape[1]).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

# overfit but why not
NUM_EPOCHS = 10000

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    with autocast(dtype=torch.bfloat16):
        pred = model(activations['post_attention_layernorm_46'])
        loss = criterion(pred, activations['post_attention_layernorm_47'])
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(loss.item())
