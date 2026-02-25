import os
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split

import config
from GraphData import graphDataset
from model import FlexibleGraphEncoder, EdgeValueDecoder, CompositeLossMSE
from Alignment_Encoder import Alignment_Encoder_train


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EdgeRegressionModel(nn.Module):
    """Edge regression model combining encoder and decoder."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        """Forward pass.

        Args:
            data: Graph data object containing:
                - x: Node features [num_nodes, in_features]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge attributes [num_edges]

        Returns:
            Predicted edge values [num_edges]
        """
        node_embeddings = self.encoder(data)
        return self.decoder(node_embeddings, data.edge_index)


def train_one_fold(fold, train_val_idx, test_idx, data, ori_data, device, alpha, beta,
                   encoder_type, decoder_mode, in_features, hidden_size, out_features,
                   Ag_hidden, Ag_out_dim, Ag_epochs, epochs=800):
    """Train and evaluate model for one fold.

    Args:
        fold: Current fold index.
        train_val_idx: Indices for training and validation edges.
        test_idx: Indices for test edges.
        data: Complete graph data.
        ori_data: Original graph data (for alignment encoder training).
        device: Computation device.
        alpha: Weight for main loss.
        beta: Weight for auxiliary loss.
        encoder_type: Type of graph encoder.
        decoder_mode: Decoding mode.
        in_features: Input feature dimension.
        hidden_size: Hidden layer dimension.
        out_features: Output embedding dimension.
        Ag_hidden: Alignment encoder hidden dimension.
        Ag_out_dim: Alignment encoder output dimension.
        Ag_epochs: Alignment encoder training epochs.
        epochs: Main model training epochs.

    Returns:
        Tuple of (MSE, MAE, R²) on test set.
    """
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42)

    encoder = FlexibleGraphEncoder(
        encoder_type=encoder_type,
        in_features=in_features,
        hidden_size=hidden_size,
        out_features=out_features,
        n_layers=2
    ).to(device)

    decoder = EdgeValueDecoder(
        mode=decoder_mode,
        in_features=out_features,
        hidden_size=hidden_size,
        n_layers=2
    ).to(device)

    model = EdgeRegressionModel(encoder, decoder).to(device)

    loss_fn = CompositeLossMSE(
        alpha=alpha,
        beta=beta,
        param_type='hyperparam',
        similarity_metric='cosine'
    )

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 3e-3, 'weight_decay': 1e-4},
        {'params': model.decoder.parameters(), 'lr': 3e-3, 'weight_decay': 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    best_val = 1e9
    best_st = None

    train_data = ori_data.clone()
    train_data.edge_index = train_data.edge_index[:, train_idx]
    train_data.edge_attr = train_data.edge_attr[train_idx]

    train_data.x = Alignment_Encoder_train(
        train_data,
        device=device,
        Ag_in_dim=train_data.num_features,
        Ag_hidden=Ag_hidden,
        Ag_out_dim=Ag_out_dim,
        Ag_epochs=Ag_epochs,
        sim=False,
        fold=fold
    )

    data.x = train_data.x
    train_data.x = train_data.x.detach().clone()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(train_data)
        src_nodes = train_data.edge_index[0]
        dst_nodes = train_data.edge_index[1]
        node_embeddings = model.encoder(train_data)
        src_emb = node_embeddings[src_nodes]
        dst_emb = node_embeddings[dst_nodes]

        total_loss, main_loss, aux_loss = loss_fn(
            pred, train_data.edge_attr, src_emb, dst_emb
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_data = data.clone()
            val_data.edge_index = val_data.edge_index[:, val_idx]
            val_data.edge_attr = val_data.edge_attr[val_idx]

            val_pred = model(val_data)
            val_src_nodes = val_data.edge_index[0]
            val_dst_nodes = val_data.edge_index[1]
            val_node_embeddings = model.encoder(val_data)
            val_src_emb = val_node_embeddings[val_src_nodes]
            val_dst_emb = val_node_embeddings[val_dst_nodes]

            val_total_loss, val_main_loss, val_aux_loss = loss_fn(
                val_pred, val_data.edge_attr, val_src_emb, val_dst_emb
            )

        if val_total_loss < best_val:
            best_val = val_total_loss
            best_st = copy.deepcopy(model.state_dict())

        if epoch % 50 == 0:
            print(f'Fold{fold} Epoch{epoch} | '
                  f'Train Loss: {total_loss:.4f} (Main: {main_loss:.4f}, Aux: {aux_loss:.4f}) | '
                  f'Val Loss: {val_total_loss:.4f} (Main: {val_main_loss:.4f}, Aux: {val_aux_loss:.4f})')

    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        test_data = data.clone()
        test_data.edge_index = test_data.edge_index[:, test_idx]
        test_data.edge_attr = test_data.edge_attr[test_idx]

        test_pred = model(test_data)
        test_src_nodes = test_data.edge_index[0]
        test_dst_nodes = test_data.edge_index[1]
        test_node_embeddings = model.encoder(test_data)
        test_src_emb = test_node_embeddings[test_src_nodes]
        test_dst_emb = test_node_embeddings[test_dst_nodes]

        test_total_loss, test_main_loss, test_aux_loss = loss_fn(
            test_pred, test_data.edge_attr, test_src_emb, test_dst_emb
        )

        test_mae = mean_absolute_error(
            test_data.edge_attr.cpu().numpy(),
            test_pred.cpu().numpy()
        )
        test_r2 = r2_score(
            test_data.edge_attr.cpu().numpy(),
            test_pred.cpu().numpy()
        )

    print(f'Fold{fold} TEST => MSE: {test_main_loss:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}')
    return test_main_loss.item(), test_mae, test_r2


def main():
    """Main training pipeline."""
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_name = config.dataset_name
    gData = graphDataset(dataset_name)
    data = gData.data.to(device)
    ori_data = data.clone()

    alpha = config.alpha
    beta = config.beta
    Ag_hidden = config.Ag_hidden
    Ag_out_dim = config.Ag_out_dim
    Ag_epochs = config.Ag_epochs
    in_features = config.in_features
    hidden_size = config.hidden_size
    out_features = config.out_features
    encoder_type = config.encoder_type
    decoder_mode = config.decoder_mode

    epochs = config.epochs

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    all_edge_indices = np.arange(data.num_edges)

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(all_edge_indices)):
        print(f"\n===== Starting Fold {fold + 1}/5 =====")

        mse, mae, r2 = train_one_fold(
            fold=fold,
            train_val_idx=train_val_idx,
            test_idx=test_idx,
            data=data,
            ori_data=ori_data,
            device=device,
            alpha=alpha,
            beta=beta,
            encoder_type=encoder_type,
            decoder_mode=decoder_mode,
            in_features=in_features,
            hidden_size=hidden_size,
            out_features=out_features,
            Ag_hidden=Ag_hidden,
            Ag_out_dim=Ag_out_dim,
            Ag_epochs=Ag_epochs,
            epochs=epochs
        )
        metrics.append([mse, mae, r2])
        torch.cuda.empty_cache()

    metrics = np.array(metrics)
    print('\n===== 5-Fold Cross-Validation Results =====')
    print(f"MSE  = {metrics[:, 0].mean():.4f} ± {metrics[:, 0].std():.4f}")
    print(f"MAE  = {metrics[:, 1].mean():.4f} ± {metrics[:, 1].std():.4f}")
    print(f"R²   = {metrics[:, 2].mean():.4f} ± {metrics[:, 2].std():.4f}")

    results_df = pd.DataFrame({
        'Fold': range(1, 6),
        'MSE': metrics[:, 0],
        'MAE': metrics[:, 1],
        'R2': metrics[:, 2]
    })

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{dataset_name}_{encoder_type}_{decoder_mode}_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()