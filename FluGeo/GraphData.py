from sklearn import preprocessing
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader

class graphDataset(Dataset):
    """
    Convert virus strains into trigram vector representations and build a graph.
    """
    def __init__(self, subtype):
        """
        Args:
            subtype: Name of the influenza subtype dataset.
        """
        max_seq_len = 600
        path = f"data/ori_data/{subtype}_antigenicDistance.xlsx"
        strain = pd.read_excel(path)

        # Store original strain names and create lowercase versions
        strain["strain1_original"] = strain["strain1"].copy()
        strain["strain2_original"] = strain["strain2"].copy()
        strain["strain1"] = strain["strain1"].str.lower()
        strain["strain2"] = strain["strain2"].str.lower()

        # Deduplicate strain names
        unique_names = pd.concat([
            strain[["strain1", "strain1_original"]].rename(
                columns={"strain1": "lower", "strain1_original": "original"}),
            strain[["strain2", "strain2_original"]].rename(
                columns={"strain2": "lower", "strain2_original": "original"})
        ]).drop_duplicates()

        # Mapping from lowercase to original names
        self.lower_to_original = pd.Series(unique_names['original'].values,
                                           index=unique_names['lower']).to_dict()

        labels = preprocessing.LabelEncoder()
        unique_labels = np.unique(list(strain["strain1"]) + list(strain["strain2"]))
        labels.fit(unique_labels)

        strain["id1"] = labels.transform(list(strain["strain1"].values))
        strain["id2"] = labels.transform(list(strain["strain2"].values))
        strain.sort_values(by=['id1', 'id2'], inplace=True)

        # Edge construction
        oriN = strain["id1"].values
        endN = strain["id2"].values
        edge_index = torch.tensor(np.array([oriN, endN]), dtype=torch.long)
        edge_attr = torch.tensor(strain["distance"].values, dtype=torch.float)

        # Node feature construction using trigram vectors
        provect = pd.read_csv("data/protVec_100d_3grams.csv", delimiter='\t')
        trigrams = list(provect['words'])
        trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
        trigram_vecs = provect.loc[:, provect.columns != 'words'].values

        x = []
        virus_names = []
        for strain_name in labels.classes_:
            if len(strain[strain["strain1"] == strain_name]["seq1"]) > 0:
                seq = strain[strain["strain1"] == strain_name]["seq1"].values[0]
            else:
                seq = strain[strain["strain2"] == strain_name]["seq2"].values[0]

            strain_embedding = []
            current_seq_len = len(seq) - 2
            if current_seq_len > max_seq_len:
                max_seq_len = current_seq_len

            for i in range(0, len(seq) - 2):
                trigram = seq[i:i + 3]
                if "-" in trigram:
                    tri_embedding = trigram_vecs[trigram_to_idx['<unk>']]
                else:
                    try:
                        tri_embedding = trigram_vecs[trigram_to_idx[trigram]]
                    except KeyError:
                        tri_embedding = trigram_vecs[trigram_to_idx['<unk>']]
                strain_embedding.append(tri_embedding)

            strain_embedding_array = np.array(strain_embedding)
            strain_embedding_tensor = torch.tensor(strain_embedding_array, dtype=torch.float)
            x.append(strain_embedding_tensor)

            original_name = self.lower_to_original.get(strain_name, strain_name)
            virus_names.append(original_name)

        x = torch.stack(x)
        x = x.detach().clone()
        x = x.view(x.size(0), -1)

        self.data = Data(
            x=x,                    # Node feature matrix
            edge_index=edge_index,  # Edge connections
            edge_attr=edge_attr,    # Antigenic distances
            virus_names=virus_names # Original strain names
        )

if __name__ == '__main__':
    gData = graphDataset("nature566H1N1")
    print(gData.data.edge_attr.shape)
    print(gData.data.x.shape)