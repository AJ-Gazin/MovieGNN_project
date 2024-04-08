import umap.umap_ as umap
import plotly.express as px
import pandas as pd
import random
import numpy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


movies_df = pd.read_csv("./sampled_movie_dataset/movies_metadata.csv")


##all_genres = movies_df['genres'].unique().tolist()  # Adjust the column name if needed
genres = movies_df['genres'].tolist()[671:] # Offset to start at movies



##can't get to work for coloring by genre
def get_genre_for_movie(movie_index):
    genres_str = movies_df.iloc[movie_index]['genres']
    # You might need to parse genres_str if it's not a simple list
    return genres_str  # Or a list of genres 

print(get_genre_for_movie(20))



def visualize_embeddings_umap(embedding_df, n_neighbors=15, min_dist=0.1, n_components=3):
    # Convert Series to DataFrame
    #embedding_df = pd.DataFrame(embedding_series.tolist(), columns=[f'dim_{i+1}' for i in range(len(embedding_series[0]))])
    # Perform UMAP dimensionality reduction
    umap_embedded = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
    ).fit_transform(embedding_df.values)


    # Plot the UMAP embedding
    umap_df = pd.DataFrame(umap_embedded, columns=['UMAP Dimension 1', 'UMAP Dimension 2', 'UMAP Dimension 3'])
    umap_df['Label'] = embedding_df.index


    color = [0]*671 + [1]*9025
    umap_df['color'] = color

    # Plot the UMAP embedding using Plotly Express
    fig = px.scatter_3d(umap_df, x='UMAP Dimension 1', y='UMAP Dimension 2',z='UMAP Dimension 3',color='color',hover_data=['Label'], title='UMAP Visualization of Embeddings')
    return fig

def visualize_embeddings_tsne(embedding_df, n_components=3, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0):
    # Perform t-SNE dimensionality reduction
    tsne_embedded = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        random_state=42,
    ).fit_transform(embedding_df.values)

    # Plot the t-SNE embedding
    tsne_df = pd.DataFrame(tsne_embedded, columns=[f't-SNE Dimension {i+1}' for i in range(n_components)])
    tsne_df['Label'] = embedding_df.index

    # Add color column (adjust how colors are applied based on your data)
    tsne_df['color'] = [0]*671 + [1]*9025 

    fig = px.scatter_3d(tsne_df, x='t-SNE Dimension 1', y='t-SNE Dimension 2', z='t-SNE Dimension 3', color='color', hover_data=['Label'], title='t-SNE Visualization of Embeddings')
    return fig


def visualize_embeddings_pca(embedding_df, n_components=3):
    # Perform PCA 
    pca = PCA(n_components=n_components, random_state=42)
    pca_embedded = pca.fit_transform(embedding_df.values)

    # Plot the PCA embedding
    pca_df = pd.DataFrame(pca_embedded, columns=[f'PCA Dimension {i+1}' for i in range(n_components)])
    pca_df['Label'] = embedding_df.index

    # Add color column (adjust how colors are applied based on your data)
    pca_df['color'] = [0]*671 + [1]*9025 

    fig = px.scatter_3d(pca_df, x='PCA Dimension 1', y='PCA Dimension 2', z='PCA Dimension 3', color='color', hover_data=['Label'], title='PCA Visualization of Embeddings')
    return fig




def save_visualization(fig, filename):
    fig.write_html(f"{filename}.html")


