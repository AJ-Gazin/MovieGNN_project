


import sys

sys.path.insert(0, "./interactive_tutorials")

import pandas as pd
from arango import ArangoClient
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import itertools
import requests
import sys
import oasis
from arango import ArangoClient
from pyArango.connection import Connection

import torch
import torch.nn.functional as F
from torch.nn import Linear
from arango import ArangoClient
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
import yaml

print(torch.__version__)


metadata_path = './sampled_movie_dataset/movies_metadata.csv'
df = pd.read_csv(metadata_path)


df.head()


df.columns



df = df.drop([19730, 29503, 35587])



links_small = pd.read_csv('./sampled_movie_dataset/links_small.csv')


links_small.head()



links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')



df['id'] = df['id'].astype('int')


sampled_md = df[df['id'].isin(links_small)]
sampled_md.shape


sampled_md['tagline'] = sampled_md['tagline'].fillna('')
sampled_md['description'] = sampled_md['overview'] + sampled_md['tagline']
sampled_md['description'] = sampled_md['description'].fillna('')


sampled_md = sampled_md.reset_index()


sampled_md.head()


indices = pd.Series(sampled_md.index, index=sampled_md['title'])


ind_gen = pd.Series(sampled_md.index, index=sampled_md['genres'])








ratings_path = './sampled_movie_dataset/ratings_small.csv'


ratings_df = pd.read_csv(ratings_path)
ratings_df.head()



def node_mappings(path, index_col):
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    return mapping


user_mapping = node_mappings(ratings_path, index_col='userId')


movie_mapping = node_mappings(ratings_path, index_col='movieId')


m_id = ratings_df['movieId'].tolist()




m_id = list(dict.fromkeys(m_id))
len(m_id)


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


id_map = pd.read_csv('./sampled_movie_dataset/links_small.csv')[['movieId', 'tmdbId']]



id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)


id_map.columns = ['movieId', 'id']



id_map.head()



id_map = id_map.merge(sampled_md[['title', 'id']], on='id').set_index('title')


indices_map = id_map.set_index('id')









login = oasis.getTempCredentials(tutorialName="MovieRecommendations", credentialProvider="https://tutorials.arangodb.cloud:8529/_db/_system/tutorialDB/tutorialDB")



movie_rec_db = oasis.connect_python_arango(login)






print("https://"+login["hostname"]+":"+str(login["port"]))
print("Username: " + login["username"])
print("Password: " + login["password"])
print("Database: " + login["dbName"])



list(movie_mapping.items())[:5]


print("%d number of unique movie ids" %len(m_id))




def remove_movies(m_id):
    no_metadata = []
    for idx in range(len(m_id)):
        tmdb_id = id_map.loc[id_map['movieId'] == m_id[idx]]

        if tmdb_id.size == 0:
            no_metadata.append(m_id[idx])
            
    return no_metadata


no_metadata = remove_movies(m_id)



for element in no_metadata:
    if element in m_id:
        print("ids with no metadata information:",element)
        m_id.remove(element)


print("Number of movies with metadata information:", len(m_id))



movie_mappings = {}
for idx, m in enumerate(m_id):
    movie_mappings[m] = idx










if not movie_rec_db.has_collection("Movie"):
    movie_rec_db.create_collection("Movie", replication_factor=3)


batch = []
BATCH_SIZE = 128
batch_idx = 1
index = 0
movie_collection = movie_rec_db["Movie"]



for idx in tqdm(range(len(m_id))):
    insert_doc = {}
    tmdb_id = id_map.loc[id_map['movieId'] == m_id[idx]]

    if tmdb_id.size == 0:
        print('No Meta data information at:', m_id[idx])


    else:
        tmdb_id = int(tmdb_id.iloc[:,1][0])
        emb_id = "Movie/" + str(movie_mappings[m_id[idx]])
        insert_doc["_id"] = emb_id
        m_meta = sampled_md.loc[sampled_md['id'] == tmdb_id]
        
        m_title = m_meta.iloc[0]['title']
        m_poster = m_meta.iloc[0]['poster_path']
        m_description = m_meta.iloc[0]['description']
        m_language = m_meta.iloc[0]['original_language']
        m_genre = m_meta.iloc[0]['genres']
        m_genre = yaml.load(m_genre, Loader=yaml.BaseLoader)
        genres = [g['name'] for g in m_genre]

        insert_doc["movieId"] = m_id[idx]
        insert_doc["mapped_movieId"] = movie_mappings[m_id[idx]]
        insert_doc["tmdbId"] = tmdb_id
        insert_doc['movie_title'] = m_title

        insert_doc['description'] = m_description
        insert_doc['genres'] = genres
        insert_doc['language'] = m_language

        if str(m_poster) == "nan":
            insert_doc['poster_path'] = "No poster path available"
        else:
            insert_doc['poster_path'] = m_poster

        batch.append(insert_doc)
        index +=1
        last_record = (idx == (len(m_id) - 1))
        if index % BATCH_SIZE == 0:
            
            batch_idx += 1
            movie_collection.import_bulk(batch)
            batch = []
        if last_record and len(batch) > 0:
            print("Inserting batch the last batch!")
            movie_collection.import_bulk(batch)










if not movie_rec_db.has_collection("Users"):
    movie_rec_db.create_collection("Users", replication_factor=3)



total_users = np.unique(ratings_df[['userId']].values.flatten()).shape[0]
print("Total number of Users:", total_users)


def populate_user_collection(total_users):
    batch = []
    BATCH_SIZE = 50
    batch_idx = 1
    index = 0
    user_ids = list(user_mapping.keys())
    user_collection = movie_rec_db["Users"]
    for idx in tqdm(range(total_users)):
        insert_doc = {}

        insert_doc["_id"] = "Users/" + str(user_mapping[user_ids[idx]])
        insert_doc["original_id"] = str(user_ids[idx])

        batch.append(insert_doc)
        index +=1
        last_record = (idx == (total_users - 1))
        if index % BATCH_SIZE == 0:
            
            batch_idx += 1
            user_collection.import_bulk(batch)
            batch = []
        if last_record and len(batch) > 0:
            print("Inserting batch the last batch!")
            user_collection.import_bulk(batch)


populate_user_collection(total_users)










if not movie_rec_db.has_collection("Ratings"):
    movie_rec_db.create_collection("Ratings", edge=True, replication_factor=3)





if not movie_rec_db.has_graph("movie_rating_graph"):
    movie_rec_db.create_graph('movie_rating_graph')


movie_rating_graph = movie_rec_db.graph("movie_rating_graph")



if not movie_rating_graph.has_vertex_collection("Users"):
    movie_rating_graph.vertex_collection("Users")



if not movie_rating_graph.has_vertex_collection("Movie"):
    movie_rating_graph.vertex_collection("Movie")




if not movie_rating_graph.has_edge_definition("Ratings"):
    Ratings = movie_rating_graph.create_edge_definition(
        edge_collection='Ratings',
        from_vertex_collections=['Users'],
        to_vertex_collections=['Movie']
    )


user_id, movie_id, ratings = ratings_df[['userId']].values.flatten(), ratings_df[['movieId']].values.flatten() , ratings_df[['rating']].values.flatten()


def create_ratings_graph(user_id, movie_id, ratings):
    batch = []
    BATCH_SIZE = 100
    batch_idx = 1
    index = 0
    edge_collection = movie_rec_db["Ratings"]
    for idx in tqdm(range(ratings.shape[0])):

        
        if movie_id[idx] in no_metadata:
            print('Removing edges with no metadata', movie_id[idx])

        else:
            insert_doc = {}
            insert_doc = {"_id":    "Ratings" + "/" + 'user-' + str(user_mapping[user_id[idx]]) + "-r-" + "movie-" + str(movie_mappings[movie_id[idx]]),
                          "_from":  ("Users" + "/" + str(user_mapping[user_id[idx]])),
                          "_to":    ("Movie" + "/" + str(movie_mappings[movie_id[idx]])),
                          "_rating": float(ratings[idx])}

            batch.append(insert_doc)
            index += 1
            last_record = (idx == (ratings.shape[0] - 1))

            if index % BATCH_SIZE == 0:
                
                batch_idx += 1
                edge_collection.import_bulk(batch)
                batch = []
            if last_record and len(batch) > 0:
                print("Inserting batch the last batch!")
                edge_collection.import_bulk(batch)


create_ratings_graph(user_id, movie_id, ratings)




















users = movie_rec_db.collection('Users')
movies = movie_rec_db.collection('Movie')
ratings_graph = movie_rec_db.collection('Ratings')


len(users), len(movies), len(ratings_graph)


print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)










def create_pyg_edges(rating_docs):
    src = []
    dst = []
    ratings = []
    for doc in rating_docs:
        _from = int(doc['_from'].split('/')[1])
        _to   = int(doc['_to'].split('/')[1])

        src.append(_from)
        dst.append(_to)
        ratings.append(int(doc['_rating']))

    edge_index = torch.tensor([src, dst])
    edge_attr = torch.tensor(ratings)

    return edge_index, edge_attr


edge_index, edge_label = create_pyg_edges(movie_rec_db.aql.execute('FOR doc IN Ratings RETURN doc'))


print(edge_index.shape)
print(edge_label.shape)

















def SequenceEncoder(movie_docs , model_name=None):
    
    
    movie_descriptions = [doc['description'] for doc in movie_docs]  
    model = SentenceTransformer(model_name, device=device)
    description_embeddings = model.encode(movie_descriptions, show_progress_bar=True,
                              convert_to_tensor=True, device=device)

    return description_embeddings 


def GenresEncoder(movie_docs):
    gen = []
    
    for doc in movie_docs:
        gen.append(doc['genres'])
        
        

    
    unique_gen = set(list(itertools.chain(*gen)))
    print("Number of unqiue genres we have:", unique_gen)

    mapping = {g: i for i, g in enumerate(unique_gen)}
    x = torch.zeros(len(gen), len(mapping))
    for i, m_gen in enumerate(gen):
        for genre in m_gen:
            x[i, mapping[genre]] = 1
    return x.to(device)


title_emb = SequenceEncoder(movie_rec_db.aql.execute('FOR doc IN Movie RETURN doc'), model_name='all-MiniLM-L6-v2')
encoded_genres = GenresEncoder(movie_rec_db.aql.execute('FOR doc IN Movie RETURN doc'))
print('Title Embeddings shape:', title_emb.shape)
print("Encoded Genres shape:", encoded_genres.shape)



movie_x = torch.cat((title_emb, encoded_genres), dim=-1)
print("Shape of the concatenated features:", movie_x.shape)











data = HeteroData()


data['user'].num_nodes = len(users)  
data['movie'].x = movie_x
data['user', 'rates', 'movie'].edge_index = edge_index
data['user', 'rates', 'movie'].edge_label = edge_label



data['user'].x = torch.eye(data['user'].num_nodes, device=device)
del data['user'].num_nodes








data = ToUndirected()(data)
del data['movie', 'rev_rates', 'user'].edge_label  


data = data.to(device)



train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'movie')],
    rev_edge_types=[('movie', 'rev_rates', 'user')],
)(data)



print('Train data:', train_data)
print('Val data:', val_data)
print('Test data', test_data)






train_data['user', 'movie'].edge_label_index



data.x_dict


data.x_dict['user'].shape, data.x_dict['movie'].shape



data.to_dict()


data.edge_index_dict


data.edge_label_dict



node_types, edge_types = data.metadata()
print('Different types of nodes in graph:',node_types)
print('Different types of edges in graph:',edge_types)




weight = torch.bincount(train_data['user', 'movie'].edge_label)
weight = weight.max() / weight



def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()










class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        
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
        
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)
        
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
        
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


model = Model(hidden_channels=32).to(device)




with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


train_losses = []
val_losses = []


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data['user', 'movie'].edge_label_index)
    target = train_data['user', 'movie'].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()

    
    train_losses.append(float(loss))

    return float(loss)




@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['user', 'movie'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['user', 'movie'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()

    
    val_losses.append(float(rmse))

    return float(rmse)


for epoch in range(1, 3000):
    
    loss = train()
    train_rmse = test(train_data)
    val_rmse = test(val_data)
    test_rmse = test(test_data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
          f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')



plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim(0, 3000)
plt.ylim(0,10)
plt.legend()
plt.title('Training and Validation Loss')
plt.show()








total_users = len(users)
total_movies = len(movies)
movie_recs = []
for user_id in tqdm(range(0, total_users)):
    user_row = torch.tensor([user_id] * total_movies)
    all_movie_ids = torch.arange(total_movies)
    edge_label_index = torch.stack([user_row, all_movie_ids], dim=0)
    pred = model(data.x_dict, data.edge_index_dict,
             edge_label_index)
    pred = pred.clamp(min=0, max=5)
    
    rec_movie_ids = (pred == 5).nonzero(as_tuple=True)
    top_ten_recs = [rec_movies for rec_movies in rec_movie_ids[0].tolist()[:10]]
    movie_recs.append({'user': user_id, 'rec_movies': top_ten_recs})










if not movie_rec_db.has_collection("Recommendation_Inferences"):
    movie_rec_db.create_collection("Recommendation_Inferences", edge=True, replication_factor=3)




def populate_movies_recommendations(movie_recs):
    batch = []

    BATCH_SIZE = 100
    batch_idx = 1
    index = 0
    rec_collection = movie_rec_db["Recommendation_Inferences"]
    for idx in tqdm(range(total_users)):
        insert_doc = {}
        to_insert = []
        user_id = movie_recs[idx]['user']
        movie_ids = movie_recs[idx]['rec_movies']

        for m_id in movie_ids:
            insert_doc = {
                           "_from":  ("Users" + "/" + str(user_id)),
                           "_to":    ("Movie" + "/" + str(m_id)),
                           "_rating": 5}
            to_insert.append(insert_doc)

        batch.extend(to_insert)
        index +=1
        last_record = (idx == (total_users - 1))
        if len(batch) > BATCH_SIZE:
          rec_collection.import_bulk(batch)
          batch = []
        if last_record and len(batch) > 0:
          print("Inserting batch the last batch!")
          rec_collection.import_bulk(batch)



populate_movies_recommendations(movie_recs)










