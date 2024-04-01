
import streamlit as st
import pandas as pd
import torch
import requests
import random
from io import BytesIO
from PIL import Image
from torch_geometric.nn import SAGEConv, to_hetero, Linear
from dotenv import load_dotenv
import os

load_dotenv() #load environment variables from .env file


API_KEY = os.getenv("HUGGINGFACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

# --- MODEL DEFINITION --- (Provided by you) 
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


# --- LOAD DATA AND MODEL ---
movies_df = pd.read_csv("./sampled_movie_dataset/movies_metadata.csv")  # Load your movie data
data = torch.load("./PyGdata.pt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(hidden_channels=32).to(device) 
model.load_state_dict(torch.load("PyGTrainedModelState.pt"))
model.eval()

# --- STREAMLIT APP ---
st.title("Movie Recommendation App")
user_id = st.number_input("Enter the User ID:", min_value=0)




def get_movie_recommendations(model, data, user_id, total_movies):
    user_row = torch.tensor([user_id] * total_movies).to(device)
    all_movie_ids = torch.arange(total_movies).to(device)
    edge_label_index = torch.stack([user_row, all_movie_ids], dim=0)

    pred = model(data.x_dict, data.edge_index_dict, edge_label_index).to('cpu')
    top_five_indices = pred.topk(5).indices

    recommended_movies = movies_df.iloc[top_five_indices]
    return recommended_movies

def generate_poster(movie_title):
    headers = {"Authorization": f"Bearer {API_KEY}"}

    #creates random seed so movie poster changes on refresh even if same title. 
    seed = random.randint(0, 2**32 - 1)
    payload = {
        "inputs": movie_title,
        # "parameters": {
        #     "seed": seed
        # }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error if the request fails

        # Display the generated image
        image = Image.open(BytesIO(response.content))
        st.image(image, caption=movie_title)

    except requests.exceptions.HTTPError as err:
        st.error(f"Image generation failed: {err}")


if st.button("Get Recommendations"):
    st.write("Top 5 Recommendations:")
    try:
        total_movies = data['movie'].num_nodes  # Adjust if necessary
        recommended_movies = get_movie_recommendations(model, data, user_id, total_movies)
        cols = st.columns(3)  

       
        for i, row in recommended_movies.iterrows():
            with cols[i % 3]: 
                ##st.write(f"{i+1}. {row['title']}")
                try:
                    image = generate_poster(row['title'])
                except requests.exceptions.HTTPError as err:
                    st.error(f"Image generation failed for {row['title']}: {err}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
