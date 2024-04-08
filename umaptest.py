import umap.umap_ as umap
import plotly.express as px
import pandas as pd
import random
import viz_utils
import torch

import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
import yaml



data = torch.load("./PyGdata.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



movies_df = pd.read_csv("./sampled_movie_dataset/movies_metadata.csv")

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # these convolutions have been replicated to match the number of edge types
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        # concat user and movie embeddings
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)
        # concatenated embeddings passed to linear layer
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)
    
class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # z_dict contains dictionary of movie and user embeddings returned from GraphSage
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
    
model = Model(hidden_channels=32).to(device)
model2 = Model(hidden_channels=32).to(device)
model.load_state_dict(torch.load("PyGTrainedModelState.pt"))
model.eval()

total_users = data['user'].num_nodes 
total_movies = data['movie'].num_nodes

print("total users =", total_users)
print("total movies =", total_movies)



with torch.no_grad():
    a = model.encoder(data.x_dict,data.edge_index_dict)
    user = pd.DataFrame(a['user'].detach().cpu())
    movie = pd.DataFrame(a['movie'].detach().cpu())
    embedding_df = pd.concat([user, movie], axis=0)


movie_index = 20
title = movies_df.iloc[movie_index]['title']
print(title) 


fig_umap = viz_utils.visualize_embeddings_umap(embedding_df) 
viz_utils.save_visualization(fig_umap, "umap_visualization")

fig_tsne = viz_utils.visualize_embeddings_tsne(embedding_df) 
viz_utils.save_visualization(fig_tsne, "tsne_visualization")

fig_pca = viz_utils.visualize_embeddings_pca(embedding_df) 
viz_utils.save_visualization(fig_pca, "pca_visualization")
