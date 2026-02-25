import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from GraphData import graphDataset
from model import FlexibleGraphEncoder, Ag_Sturction_loss


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Alignment_Encoder_train(data, device, Ag_in_dim, Ag_hidden, Ag_out_dim, Ag_epochs, fold, sim=False, data_name=None):
    """Train the alignment encoder to generate structured node embeddings.

    Args:
        data: Graph data object containing node features, edges, and attributes.
        device: Computation device (CPU/GPU).
        Ag_in_dim: Input feature dimension for the encoder.
        Ag_hidden: Hidden layer dimension.
        Ag_out_dim: Output embedding dimension.
        Ag_epochs: Number of training epochs.
        fold: Current fold index (for logging).
        sim: Whether to compute similarity metrics (unused, kept for compatibility).
        data_name: Dataset name for saving training logs.

    Returns:
        Detached node embeddings.
    """
    set_seed(42)
    print(f"Alignment encoder device: {device}")
    print(f"Alignment encoder input dimension: {Ag_in_dim}")

    max_antigen_distance = data.edge_attr.max().item()
    print(f"Max antigen distance: {max_antigen_distance:.4f}")

    Ag_encoder_type = 'GCN'
    Ag_encoder = FlexibleGraphEncoder(
        encoder_type=Ag_encoder_type,
        in_features=Ag_in_dim,
        hidden_size=Ag_hidden,
        out_features=Ag_out_dim,
        n_layers=2,
        dropout=0
    ).to(device)

    optimizer = torch.optim.AdamW([
        {'params': Ag_encoder.parameters(), 'lr': 3e-4, 'weight_decay': 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.90)

    if data_name is not None:
        all_epochs = []
        all_losses = []

    print("Alignment encoder training started")
    for epoch in range(1, Ag_epochs + 1):
        Ag_encoder.train()
        optimizer.zero_grad()

        embeddings = Ag_encoder(data)
        loss = Ag_Sturction_loss(
            embeddings, data.edge_index, data.edge_attr, max_antigen_distance
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if data_name is not None:
            all_epochs.append(epoch - 1)
            all_losses.append(loss.item())

        if epoch % 50 == 0:
            Ag_encoder.eval()
            with torch.no_grad():
                emb = Ag_encoder(data)
                loss_now = Ag_Sturction_loss(emb, data.edge_index, data.edge_attr, max_antigen_distance)
            Ag_encoder.train()
            print(f"Epoch {epoch:03d}, Train-Loss: {loss.item():.4f}, Eval-Loss: {loss_now.item():.4f}")

    if data_name is not None:
        save_training_loss(data_name, all_epochs, all_losses, interval=10)

    return embeddings.detach()


def save_training_loss(data_name, epochs, losses, save_dir=".", interval=10):
    """Save training loss history to a JSON file.

    Args:
        data_name: Dataset name.
        epochs: List of epoch numbers.
        losses: List of loss values.
        save_dir: Directory to save the JSON file.
        interval: Sampling interval for the data points.
    """
    sampled_epochs = []
    sampled_losses = []

    for i, (epoch, loss) in enumerate(zip(epochs, losses)):
        if i % interval == 0:
            sampled_epochs.append(epoch)
            sampled_losses.append(float(loss))

    data = {
        "dataset": data_name,
        "epochs": sampled_epochs,
        "val_loss": sampled_losses
    }

    filename = f"{data_name}_AG_LOSS.json"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Training loss saved to: {filepath}")
    print(f"Number of sampled data points: {len(sampled_epochs)}")

    return filepath


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = "nature585BVictoria"
    gData = graphDataset(dataset_name)
    data = gData.data.to(device)

    Alignment_Encoder_train(
        data,
        device=device,
        Ag_in_dim=data.num_features,
        Ag_hidden=256,
        Ag_out_dim=128,
        Ag_epochs=1500,
        sim=False,
        fold=1,
        data_name=dataset_name
    )