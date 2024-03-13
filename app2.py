#ATTEMPT AT A FANCIER STREAMLIT -- NOT YET WORKING



import streamlit as st
from arango import ArangoClient
import huggingface_hub
from PIL import Image
import requests
from io import BytesIO
from streamlit import experimental_memo
import streamlit_modal as modal
import random
from st_clickable_images import clickable_images
import base64

# StableDiffusion
API_KEY = "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" 
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"



# ArangoDB connection details
ARANGO_URL = "https://tutorials.arangodb.cloud:8529"
ARANGO_USER = "TUThe5pvbu55xsalxczfo1dip"
ARANGO_PASSWORD = "TUTkxua8mas5yphb2jhl4lh7p"
ARANGO_DATABASE = "TUT8zj5z8o7wxqtndgnevej"

# Connect to ArangoDB
client = ArangoClient(hosts=ARANGO_URL)
db = client.db(ARANGO_DATABASE, username=ARANGO_USER, password=ARANGO_PASSWORD)


# def get_movie_title(movie_id):
#     """Fetches movie title from the 'Movie' collection."""
#     aql = f"FOR doc IN Movie FILTER doc._id == 'Movie/{movie_id}' RETURN doc"
#     cursor = db.aql.execute(aql)
#     for movie_doc in cursor:  # Iterate over the cursor
#         return movie_doc['movie_title']  # Return the title
def get_movie_title_and_desc(movie_id):
    """Fetches movie title from the 'Movie' collection."""
    aql = f"FOR doc IN Movie FILTER doc._id == 'Movie/{movie_id}' RETURN doc"
    cursor = db.aql.execute(aql)
    for movie_doc in cursor:  # Iterate over the cursor
        return movie_doc['movie_title'], movie_doc['description']  # Return the title
    
def get_recommendations(user_id):
    """Queries recommendations from the 'Recommendation_Inferences' collection."""
    aql = f"FOR v, e IN 1..1 OUTBOUND 'Users/{user_id}' Recommendation_Inferences RETURN e._to"
    cursor = db.aql.execute(aql)
    return [movie_id.split('/')[1] for movie_id in cursor]

def generate_poster(movie_title):
    headers = {"Authorization": f"Bearer {API_KEY}"}

    #keeps same seed
    seed = 1
    #creates random seed so movie poster changes on refresh even if same title. 
    #seed = random.randint(0, 2**32 - 1)
    
    seed = 1
    payload = {
        "inputs": movie_title,
        "parameters": {
            "seed": seed
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error if the request fails
        # ... 
        image = Image.open(BytesIO(response.content))

        # Convert to base64
        with BytesIO() as buffer:
            image.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/jpeg;base64,{img_str}"  # Return base64-encoded string


    except requests.exceptions.HTTPError as err:
        st.error(f"Image generation failed: {err}")
        return "0"
   
   
    return image


    
# New session state management
if 'current_movie' not in st.session_state:
    st.session_state['current_movie'] = None

# Streamlit app structure
st.title("Movie Recommendation App")
user_id = st.number_input("Enter your user ID:", min_value=0)

def show_movie_details(movie_id):
    """Displays a detail page for the selected movie"""
    movie_title, movie_description = get_movie_title_and_desc(movie_id)
    st.header(movie_title)
    st.image(generate_poster(movie_title), caption=movie_title)
    st.write(movie_description)
    st.button("Back to Recommendations", on_click=st.session_state.pop, args=('current_movie',))

if st.button("Get Recommendations"):
    recommended_movie_ids = get_recommendations(user_id)

    if not recommended_movie_ids:
        st.warning("No recommendations found for this user.")
    else:
        st.subheader("Recommended Movies:")

        cols = st.columns(3)
        posters = []  # Store generated posters

        for i, movie_id in enumerate(recommended_movie_ids):
            with cols[i % 3]:
                movie_title, _ = get_movie_title_and_desc(movie_id)   

                try:
                    poster = generate_poster(movie_title) 
                    posters.append(poster)  # Add to the list of posters

                except requests.exceptions.HTTPError as err:
                    st.error(f"Image generation failed for {movie_title}: {err}")

        # Display clickable posters
        clicked_image_idx = clickable_images(
            posters,  
            titles=[get_movie_title_and_desc(id_)[0] for id_ in recommended_movie_ids],  # Extract titles 
            div_style={"display": "flex", "justify-content": "center", "flex-flow": "column wrap"},
            img_style={"margin": "5px", "height": "200px"},
        )

        if clicked_image_idx >= 0:  
            st.session_state['current_movie'] = recommended_movie_ids[clicked_image_idx]
            show_movie_details(st.session_state['current_movie'])

# Handle detail page display based on session state
if st.session_state['current_movie']:
    show_movie_details(st.session_state['current_movie'])
                
            