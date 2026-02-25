import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.utils import softmax


def Ag_Sturction_loss(embeddings, edge_index, edge_attr, max_antigen_distance):
    """Loss function for antigenic distance constraint.

    Args:
        embeddings: Node embeddings [num_nodes, embedding_dim]
        edge_index: Edge indices [2, num_edges]
        edge_attr: Antigenic distances [num_edges]
        max_antigen_distance: Maximum antigenic distance in dataset (scalar)

    Returns:
        loss: Computed loss value
    """
    src_idx, dst_idx = edge_index
    src_emb = embeddings[src_idx]
    dst_emb = embeddings[dst_idx]

    cos_sim = F.cosine_similarity(src_emb, dst_emb, dim=1)
    target_sim = 1 - 2 * (edge_attr / max_antigen_distance)

    loss = F.mse_loss(cos_sim, target_sim)
    return loss

class LinearEncoder(nn.Module):
    """Linear Encoder"""

    def __init__(self, in_features, hidden_size, out_features, n_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            if i == 0:
                self.layers.append(nn.Linear(in_features, hidden_size))
            elif i == n_layers - 1:
                self.layers.append(nn.Linear(hidden_size, out_features))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GCNEncoder(nn.Module):
    """GCN Encoder"""

    def __init__(self, in_features, hidden_size, out_features, n_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            if i == 0:
                self.layers.append(GCNConv(in_features, hidden_size))
            elif i == n_layers - 1:
                self.layers.append(GCNConv(hidden_size, out_features))
            else:
                self.layers.append(GCNConv(hidden_size, hidden_size))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoder(nn.Module):
    """GAT Encoder"""

    def __init__(self, in_features, hidden_size, out_features, n_layers=2, heads=1, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.res3 = LinearEncoder(
            in_features, hidden_size, out_features, n_layers, dropout
        )

        for i in range(n_layers):
            if i == 0:
                self.layers.append(GATConv(in_features, hidden_size, heads=heads, dropout=dropout))
            elif i == n_layers - 1:
                self.layers.append(GATConv(hidden_size * heads, out_features, heads=1, concat=False, dropout=dropout))
            else:
                self.layers.append(GATConv(hidden_size * heads, hidden_size, heads=heads, dropout=dropout))

    def forward(self, x, edge_index):
        x_res3 = self.res3(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x + x_res3

class COSGAT(MessagePassing):
    """Modified COSGAT layer with edge weight gating mechanism"""

    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.3, bias=True, **kwargs):
        super(COSGAT, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.fusion_beta = nn.Parameter(torch.Tensor(1, heads, 1))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.constant_(self.fusion_beta, 0.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index, edge_weight=None, return_attention_weights=False):
        x_orig = x
        H = self.lin(x).view(-1, self.heads, self.out_channels)

        out = self.propagate(
            edge_index,
            H=H,
            x_orig=x_orig,
            edge_weight=edge_weight,
            size=None
        )

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        out = F.elu(out)

        if return_attention_weights:
            return out, self.attention_weights
        else:
            return out

    def message(self, H_i, H_j, x_orig_i, x_orig_j, edge_weight, index, ptr, size_i):
        E = H_i.size(0)

        # GAT attention computation
        H_cat = torch.cat([H_i, H_j], dim=-1)
        gat_logits = (H_cat * self.att).sum(dim=-1)
        gat_logits = F.leaky_relu(gat_logits, self.negative_slope)
        gat_logits = gat_logits.view(E, self.heads)

        # Cosine similarity computation
        cos_sim = F.cosine_similarity(x_orig_i, x_orig_j, dim=-1)
        cos_sim = cos_sim.view(E, 1).expand(-1, self.heads)

        # Normalization
        gat_alpha = softmax(gat_logits, index, num_nodes=size_i)
        cos_alpha = softmax(cos_sim, index, num_nodes=size_i)

        # Fusion of attention mechanisms
        beta = torch.sigmoid(self.fusion_beta)
        beta = beta.view(1, self.heads)
        fused_attention = (1 - beta) * gat_alpha + beta * cos_alpha

        # Edge weight gating
        if edge_weight is not None:
            w = edge_weight.reshape(E, 1)
            w_clamped = torch.clamp(w, max=4.0)
            gate_factor = torch.clamp(1.0 - (w_clamped / 4.0), min=0.0, max=1.0)
            gated_attention = fused_attention * gate_factor
        else:
            gated_attention = fused_attention

        gated_attention = gated_attention.view(E, self.heads)
        final_attention = softmax(gated_attention, index, num_nodes=size_i)

        self.attention_weights = (index, final_attention)
        final_attention = F.dropout(final_attention, p=self.dropout, training=self.training)

        return H_j * final_attention.unsqueeze(-1)


class COSGATEncoder(nn.Module):
    """COSGAT Encoder"""

    def __init__(self, in_features, hidden_size, out_features, n_layers=2, dropout=0):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.res3 = LinearEncoder(
            in_features, hidden_size, out_features, n_layers, dropout
        )

        for i in range(n_layers):
            if i == 0:
                self.layers.append(COSGAT(in_features, hidden_size, dropout=dropout))
            elif i == n_layers - 1:
                self.layers.append(COSGAT(hidden_size, out_features, dropout=dropout))
            else:
                self.layers.append(COSGAT(hidden_size, hidden_size, dropout=dropout))

    def forward(self, x, edge_index, edge_attr):
        x_res3 = self.res3(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x + x_res3


class FlexibleGraphEncoder(nn.Module):
    """Flexible graph encoder supporting multiple encoder types."""

    def __init__(self, encoder_type, in_features, hidden_size, out_features,
                 n_layers=2, heads=1, dropout=0.3):
        super().__init__()
        self.encoder_type = encoder_type.upper()
        self.res1 = nn.Linear(in_features, hidden_size)
        self.res2 = nn.Linear(hidden_size, out_features)
        self.res4 = nn.Linear(in_features, out_features)
        self.res3 = LinearEncoder(
            in_features, hidden_size, out_features, n_layers, dropout
        )

        if n_layers < 1:
            raise ValueError("Number of layers must be at least 1")

        if self.encoder_type == 'GCN':
            self.encoder = GCNEncoder(
                in_features, hidden_size, out_features, n_layers, dropout
            )
        elif self.encoder_type == 'GAT':
            self.encoder = GATEncoder(
                in_features, hidden_size, out_features, n_layers, heads, dropout
            )
        elif self.encoder_type == 'LINEAR':
            self.encoder = LinearEncoder(
                in_features, hidden_size, out_features, n_layers, dropout
            )
        elif self.encoder_type == 'COSGAT':
            self.encoder = COSGATEncoder(
                in_features, hidden_size, out_features, n_layers, dropout
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}. Supported: GCN, GAT, LINEAR, COSGAT")

    def forward(self, data):
        if self.encoder_type == 'GCN':
            x_res3 = self.res3(data.x)
            x_res4 = self.res4(data.x)
            return self.encoder(data.x, data.edge_index) + x_res3
        elif self.encoder_type == 'GAT':
            x_res3 = self.res3(data.x)
            x_res4 = self.res4(data.x)
            return self.encoder(data.x, data.edge_index) + x_res4
        elif self.encoder_type == 'LINEAR':
            x_res3 = self.res3(data.x)
            return self.encoder(data.x)
        elif self.encoder_type == 'COSGAT':
            x_res3 = self.res3(data.x)
            x_res4 = self.res4(data.x)
            return self.encoder(data.x, data.edge_index, data.edge_attr) + x_res4


class EdgeValueDecoder(nn.Module):
    """Edge value decoder for predicting edge attributes from node features.

    Args:
        mode: Decoding mode ('concat', 'add', 'dot', 'euclidean')
        in_features: Feature dimension per node
        hidden_size: Hidden layer dimension (required for 'concat' and 'add' modes)
        n_layers: Number of MLP layers (required for 'concat' and 'add' modes)
        dropout: Dropout probability (required for 'concat' and 'add' modes)
    """

    def __init__(self, mode, in_features, hidden_size=None, n_layers=2, dropout=0.3):
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in ['concat', 'add', 'dot', 'euclidean']:
            raise ValueError(f"Unsupported decoding mode: {mode}. Supported: 'concat', 'add', 'dot', 'euclidean'")

        if self.mode in ['concat', 'add']:
            if hidden_size is None:
                hidden_size = in_features * 2 if self.mode == 'concat' else in_features

            layers = []
            if self.mode == 'concat':
                input_dim = in_features * 2
            else:
                input_dim = in_features

            for i in range(n_layers - 1):
                layers.append(nn.Linear(input_dim, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_size

            layers.append(nn.Linear(hidden_size, 1))
            self.mlp = nn.Sequential(*layers)

        elif self.mode == 'dot':
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index):
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        x_src = x[src_nodes]
        x_dst = x[dst_nodes]

        if self.mode == 'concat':
            combined = torch.cat([x_src, x_dst], dim=1)
            return self.mlp(combined).squeeze(-1)
        elif self.mode == 'add':
            combined = x_src + x_dst
            return self.mlp(combined).squeeze(-1)
        elif self.mode == 'dot':
            dot_product = torch.sum(x_src * x_dst, dim=1)
            return self.scale * dot_product + self.bias
        elif self.mode == 'euclidean':
            distances = torch.norm(x_src - x_dst, dim=1)
            return distances


class GCNDecoder(torch.nn.Module):
    """GCN-based decoder for edge prediction."""

    def __init__(self, out_feats):
        super().__init__()
        self.fc1 = nn.Linear(2 * out_feats, 2 * out_feats)
        self.bn1 = DyT(2 * out_feats)
        self.fc2 = nn.Linear(2 * out_feats, 2 * out_feats)
        self.bn2 = DyT(2 * out_feats)
        self.fc3 = nn.Linear(out_feats, out_feats)
        self.bn3 = DyT(out_feats)
        self.fc4 = nn.Linear(2 * out_feats, 2 * out_feats)
        self.bn4 = DyT(2 * out_feats)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.final = nn.Linear(2 * out_feats, 1)
        self.l1_lambda = 0.001

        self.reduce_dim1 = nn.Linear(2 * out_feats, out_feats)
        self.reduce_bn1 = DyT(out_feats)
        self.reduce_dim2 = nn.Linear(out_feats, out_feats // 2)
        self.reduce_bn2 = DyT(out_feats // 2)
        self.reduce_dim3 = nn.Linear(out_feats // 2, 1)

        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        x_src = x[edge_index[0]]
        x_dst = x[edge_index[1]]
        edge_x = torch.cat((x_src, x_dst), dim=1)

        edge_x = self.fc4(edge_x)
        edge_x = self.bn4(edge_x)
        edge_x = self.relu(edge_x)
        edge_x = self.dropout1(edge_x)

        out1 = self.fc1(edge_x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.dropout2(out1)
        out1 = edge_x + out1

        out2 = self.fc2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)
        out2 = self.dropout3(out2)
        out2 += out1

        out = self.final(out2)
        out = torch.flatten(out)
        return out


class DyT(nn.Module):
    """Dynamic Tanh activation module."""

    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class CompositeLossMSE(nn.Module):
    """Composite MSE loss with similarity constraint.

    Args:
        alpha: Weight for main loss (default: 1.0)
        beta: Weight for auxiliary loss (default: 1.0)
        param_type: 'hyperparam' (fixed) or 'learnable' parameters
        similarity_metric: Similarity metric ('cosine', 'euclidean', 'dot')
    """

    def __init__(self, alpha=1.0, beta=1.0, param_type='hyperparam', similarity_metric='cosine'):
        super().__init__()
        self.similarity_metric = similarity_metric

        if param_type == 'hyperparam':
            self.alpha = alpha
            self.beta = beta
        elif param_type == 'learnable':
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}. Supported: 'hyperparam', 'learnable'")

    def forward(self, pred_edge_attr, true_edge_attr, src_emb, dst_emb):
        main_loss = F.mse_loss(pred_edge_attr, true_edge_attr)

        if self.similarity_metric == 'cosine':
            similarities = F.cosine_similarity(src_emb, dst_emb, dim=1)
            max_antigen_distance = true_edge_attr.max().item()
            target_sim = 1 - 2 * (true_edge_attr / max_antigen_distance)
            aux_loss = F.mse_loss(similarities, target_sim)
        elif self.similarity_metric == 'euclidean':
            distances = torch.norm(src_emb - dst_emb, dim=1)
            aux_loss = F.mse_loss(true_edge_attr, distances)
        elif self.similarity_metric == 'dot':
            dot_product = torch.sum(src_emb * dst_emb, dim=1)
            processed_edge_attr = true_edge_attr.clone()
            processed_edge_attr[true_edge_attr < 0] = 0
            aux_loss = F.mse_loss(processed_edge_attr, dot_product)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

        total_loss = self.alpha * main_loss + self.beta * aux_loss
        if self.beta == 0:
            total_loss = main_loss
        return total_loss, main_loss, aux_loss


if __name__ == '__main__':
    from GraphData import graphDataset

    gData = graphDataset("nature566H1N1")
    print(gData.data.edge_attr.shape)
    print(gData.data.x.shape)

    encoder = FlexibleGraphEncoder("GCN", gData.data.x.size()[1], 1024, 256)
    x = encoder(gData.data)
    print(x.size())

    decoder = EdgeValueDecoder("dot", gData.data.x.size()[1], 64, 2)
    x = decoder(x, gData.data.edge_index)
    print(x)
    print(x.size())