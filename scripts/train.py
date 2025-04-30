import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
import click

class ActivationMapping(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(x)

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

def get_val_loss(model, Xs_val, Ys_val, criterion):
    with torch.no_grad():
        outputs = model(Xs_val)
        loss = criterion(outputs, Ys_val)
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

def main(mse, activations_path, out_path, layer_idx, batch_size, num_epochs, lr):
    """Training script for processing activations."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        Ys = torch.load(out_path)
        criterion = nn.NLLLoss()
    
    breakpoint()
    Xs_train, Ys_train, Xs_val, Ys_val = get_train_val_split(Xs, Ys)

    

    # Initialize model and optimizer
    model = ActivationMapping(dim=Xs.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(num_epochs):
        idx = torch.randint(0, Xs_train.shape[0], (batch_size,))
        Xs = Xs_train[idx]
        Ys = Ys_train[idx]
        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):
            # Forward pass
            outputs = model(Xs)
            loss = criterion(outputs, Ys)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            val_loss = get_val_loss(model, Xs_val, Ys_val, criterion)
            print(f"Epoch {epoch} train loss: {loss.item()} val loss: {val_loss}")

if __name__ == "__main__":
    main()



