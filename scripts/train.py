import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import click

class ActivationMapping(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.proj.weight)    # start as identity
        nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        return x + self.proj(x)           # residual

def get_train_val_split(Xs, Ys):
    perm = torch.randperm(Xs.shape[0])
    Xs = Xs[perm]
    Ys = Ys[perm]
    n = int(Xs.shape[0] * 0.8)
    return Xs[:n], Ys[:n], Xs[n:], Ys[n:]

def get_val_loss(model, Xs_val, Ys_val, mse, criterion, batch_size, norm=None, lm_head=None):
    idx = torch.randint(0, Xs_val.shape[0], (batch_size,))
    Xs_batch = Xs_val[idx]
    Ys_batch = Ys_val[idx]
    with torch.no_grad():
        with autocast(dtype=torch.bfloat16):
            outputs = model(Xs_batch)
            if mse:
                loss = criterion(outputs, Ys_batch)
            else:
                h_norm = norm(Xs_batch)
                logits = lm_head(h_norm)
                log_sm = F.log_softmax(logits, dim=-1)
                loss = criterion(log_sm, Ys_batch)
    return loss.item()

def validate_out_path(ctx, param, value):
    if not ctx.params.get('mse', True) and not value:
        raise click.BadParameter('--out-path is required when using --no-mse')
    return value

@click.command()
@click.option('--mse/--no-mse', default=True, help='Use MSE loss if set, otherwise use NLL')
@click.option('--activations-path', required=True, type=click.Path(exists=True),
              help='Path to the activations file')
@click.option('--out-path', default='', type=click.Path(), callback=validate_out_path,
              help='Path to targets when using --no-mse')
@click.option('--layer-source', required=True, type=int,
              help='Layer index to use for input features')
@click.option('--layer-target', required=True, type=int,
              help='Layer index to use for output features')
@click.option('--batch-size', default=8192, type=int, help='Batch size')
@click.option('--num-epochs', default=30000, type=int, help='Number of epochs')
@click.option('--lr', default=1e-5, type=float, help='Learning rate')
@click.option('--save', required=True, type=click.Path(), help='Path to save the model')
def main(mse, activations_path, out_path, layer_source, layer_target,
         batch_size, num_epochs, lr, save):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    activations = torch.load(activations_path)
    Xs = activations[layer_source].squeeze(0)

    if mse:
        Ys = activations[layer_target].squeeze(0)
        criterion = nn.MSELoss()
    else:
        from transformers.models.llama4.modeling_llama4 import Llama4TextRMSNorm
        from transformers import AutoConfig
        ckpt = torch.load("/mnt/ss-decoding/clean_activations/norm_and_head.pth")
        cfg = AutoConfig.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")
        norm = Llama4TextRMSNorm(cfg.text_config.hidden_size,
                                 eps=cfg.text_config.rms_norm_eps).eval().to(device)
        norm.load_state_dict(ckpt['norm'])
        lm_head = nn.Linear(cfg.text_config.hidden_size,
                            cfg.text_config.vocab_size, bias=False).eval().to(device)
        lm_head.load_state_dict(ckpt['head'])
        Ys = torch.load(out_path)
        criterion = nn.NLLLoss()

    Xs_train, Ys_train, Xs_val, Ys_val = get_train_val_split(Xs, Ys)
    Xs_train, Ys_train = Xs_train.to(device), Ys_train.to(device)
    Xs_val, Ys_val = Xs_val.to(device), Ys_val.to(device)

    model = ActivationMapping(dim=Xs.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        idx = torch.randint(0, Xs_train.shape[0], (batch_size,))
        Xb = Xs_train[idx]
        Yb = Ys_train[idx]

        optimizer.zero_grad()
        with autocast(dtype=torch.bfloat16):
            out = model(Xb)
            if mse:
                loss = criterion(out, Yb)
            else:
                h_norm = norm(Xb)
                logits = lm_head(h_norm)
                loss = criterion(F.log_softmax(logits, dim=-1), Yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if epoch % 100 == 0:
            if mse:
                v_loss = get_val_loss(model, Xs_val, Ys_val, True, criterion, batch_size)
            else:
                v_loss = get_val_loss(model, Xs_val, Ys_val, False, criterion,
                                      batch_size, norm, lm_head)
            print(f"Epoch {epoch} — train loss: {loss.item():.6f}, val loss: {v_loss:.6f}")

    torch.save(model.state_dict(), save)

if __name__ == "__main__":
    main()