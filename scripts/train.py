import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
import click

class ActivationMapping(nn.Module):
    def __init__(self, dim, hidden_mult=8):
        super().__init__()
        h = dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, dim),
        )

    def forward(self, x):
        return self.net(x)

def get_train_val_split(Xs, Ys):
    perm = torch.randperm(Xs.shape[0])
    Xs = Xs[perm]
    Ys = Ys[perm]
    n = int(Xs.shape[0]*0.8)
    Xs_train = Xs[:n]
    Ys_train = Ys[:n]
    Xs_val = Xs[n:]
    Ys_val = Ys[n:]
    return Xs_train, Ys_train, Xs_val, Ys_val

def get_val_loss(model, Xs_val, Ys_val, mse, criterion, batch_size, norm=None, lm_head=None):
    idx = torch.randint(0, Xs_val.shape[0], (batch_size,))
    Xs_val = Xs_val[idx]
    Ys_val = Ys_val[idx]
    with torch.no_grad():
        with autocast(dtype=torch.bfloat16):
            outputs = model(Xs_val)
            if mse:
                loss = criterion(outputs, Ys_val)
            else:
                h_norm = norm(Xs_val)
                logits = lm_head(h_norm)
                log_softmax = F.log_softmax(logits, dim=-1)
                loss = criterion(log_softmax, Ys_val)
    return loss.item()

def validate_out_path(ctx, param, value):
    if not ctx.params.get('mse', True) and not value:
        raise click.BadParameter('--out-path is required when using --no-mse')
    return value

@click.command()
@click.option('--mse/--no-mse', default=True, help='Use MSE loss if set, otherwise use different loss')
@click.option('--activations-path', required=True, type=click.Path(exists=True),
              help='Path to the activations file')
@click.option('--out-path', default='', type=click.Path(), callback=validate_out_path,
              help='Output path for saving results (required when using --no-mse)')
@click.option('--layer-idx', required=True, type=int,
              help='Layer index to use for input features')
@click.option('--batch-size', default=8192, type=int,
              help='Batch size')
@click.option('--num-epochs', default=30000, type=int,
              help='Number of epochs')
@click.option('--lr', default=1e-5, type=float,
              help='Learning rate')
@click.option('--save', required=True, type=click.Path(),
              help='Path to save the model')

def main(mse, activations_path, out_path, layer_idx, batch_size, num_epochs, lr, save):
    """Training script for processing activations."""
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load activations
    activations = torch.load(activations_path)

    # Get input features
    Xs = activations[f"layer_{layer_idx}"]
    
    # If using MSE, true examples are layer 48 hidden state
    # If using NLL, true examples are output of the model
    if mse:
        Ys = activations[f"layer_47"]
        criterion = nn.MSELoss()
    else:
        from transformers.models.llama4.modeling_llama4 import Llama4TextRMSNorm
        from transformers import AutoConfig
        ckpt = torch.load("/mnt/ss-decoding/clean_activations/norm_and_head.pth")
        cfg = AutoConfig.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")  
        norm = Llama4TextRMSNorm(cfg.text_config.hidden_size, eps=cfg.text_config.rms_norm_eps).eval()
        norm.load_state_dict(ckpt['norm'])
        norm.to(device)
        lm_head = nn.Linear(cfg.text_config.hidden_size, cfg.text_config.vocab_size, bias=False).to(device).eval()
        lm_head.load_state_dict(ckpt['head'])
        lm_head.to(device)
        Ys = torch.load(out_path)
        criterion = nn.NLLLoss()
    
    # breakpoint()
    Xs_train, Ys_train, Xs_val, Ys_val = get_train_val_split(Xs, Ys)
    Xs_train = Xs_train.to(device)
    Ys_train = Ys_train.to(device)
    Xs_val = Xs_val.to(device)
    Ys_val = Ys_val.to(device)

    # some sanity checks
    # idx = torch.randint(0, Xs_train.shape[0], (batch_size,))
    # Xs = Xs_train[idx].to(device)
    # Ys = Ys_train[idx].to(device)
    # h_norm = norm(Xs)
    # logits = lm_head(h_norm)
    # log_probs = F.log_softmax(logits, dim=-1)
    # nll = torch.sum(log_probs[torch.arange(batch_size), Ys])
    # print(nll/batch_size)
    # breakpoint()

    # Initialize model and optimizer
    model = ActivationMapping(dim=Xs.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(num_epochs):
        if epoch > 500:
            optimizer = optim.Adam(model.parameters(), lr=3e-5)
        idx = torch.randint(0, Xs_train.shape[0], (batch_size,))
        Xs = Xs_train[idx]
        Ys = Ys_train[idx]
        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):
            # Forward pass
            outputs = model(Xs)
            if mse:
                loss = criterion(outputs, Ys)
            else:
                h_norm = norm(Xs)
                logits = lm_head(h_norm)
                log_softmax = F.log_softmax(logits, dim=-1)
                loss = criterion(log_softmax, Ys)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            if mse:
                val_loss = get_val_loss(model, Xs_val, Ys_val, mse, criterion, batch_size)
            else:
                val_loss = get_val_loss(model, Xs_val, Ys_val, mse, criterion, batch_size, norm, lm_head)
            print(f"Epoch {epoch} train loss: {loss.item()} val loss: {val_loss}")

    torch.save(model.state_dict(), save)

if __name__ == "__main__":
    main()



