
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

import viz_utils
import model_def

load_dotenv() #load environment variables from .env file


API_KEY = os.getenv("HUGGINGFACE_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

# --- LOAD DATA AND MODEL ---
movies_df = pd.read_csv("./sampled_movie_dataset/movies_metadata.csv")  # Load your movie data
data = torch.load("./PyGdata.pt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model_def.Model(hidden_channels=32).to(device) 
model.load_state_dict(torch.load("PyGTrainedModelState.pt"))
model.eval()

# --- STREAMLIT APP ---
st.title("Movie Recommendation App")
user_id = st.number_input("Enter the User ID:", min_value=0)

with torch.no_grad():
    a = model.encoder(data.x_dict,data.edge_index_dict)
    user = pd.DataFrame(a['user'].detach().cpu())
    movie = pd.DataFrame(a['movie'].detach().cpu())
    embedding_df = pd.concat([user, movie], axis=0)

st.subheader('UMAP Visualization')
umap_fig = viz_utils.visualize_embeddings_umap(embedding_df)
st.plotly_chart(umap_fig)

st.subheader('TSNE Visualization')
tsne_fig = viz_utils.visualize_embeddings_tsne(embedding_df)
st.plotly_chart(tsne_fig)

st.subheader('PCA Visualization')
pca_fig = viz_utils.visualize_embeddings_pca(embedding_df)
st.plotly_chart(pca_fig)



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
        total_movies = data['movie'].num_nodes  
        recommended_movies = get_movie_recommendations(model, data, user_id, total_movies)
        cols = st.columns(3)  

       
        for i, row in recommended_movies.iterrows():
            with cols[i % 3]: 
                #st.write(f"{i+1}. {row['title']}") 
                try:
                    image = generate_poster(row['title'])
                except requests.exceptions.HTTPError as err:
                    st.error(f"Image generation failed for {row['title']}: {err}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
