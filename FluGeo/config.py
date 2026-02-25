import os
import torch
from GraphData import graphDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset configuration
dataset_name = "nature566H1N1"
gData = graphDataset(dataset_name)
data = gData.data.to(device)
ori_data = data.clone()

# Model hyperparameters
alpha = 1
beta = 5
Ag_hidden = 256
Ag_out_dim = 128
Ag_epochs = 1000
in_features = 128
hidden_size = 512
out_features = 256
encoder_type = 'COSGAT'  # Encoder type: 'GCN', 'GAT', 'LINEAR', 'COSGAT'
decoder_mode = 'concat'  # Decoder mode: 'concat', 'add', 'dot', 'euclidean'

# Training hyperparameters
epochs = 1000