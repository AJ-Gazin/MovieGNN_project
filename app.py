import streamlit as st
from streamlit import experimental_memo
from arango import ArangoClient
from PIL import Image
import requests
import random
from io import BytesIO
import streamlit_modal as modal


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

    #creates random seed so movie poster changes on refresh even if same title. 
    seed = random.randint(0, 2**32 - 1)
    payload = {
        "inputs": movie_title,
        "parameters": {
            "seed": seed
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error if the request fails

        # Display the generated image
        image = Image.open(BytesIO(response.content))
        st.image(image, caption=movie_title)

    except requests.exceptions.HTTPError as err:
        st.error(f"Image generation failed: {err}")


    
# Streamlit app structure
st.title("Movie Recommendation App")
user_id = st.number_input("Enter your user ID:", min_value=0)

if st.button("Get Recommendations"):
    recommended_movie_ids = get_recommendations(user_id)

    if not recommended_movie_ids:
        st.warning("No recommendations found for this user.")
    else:
        st.subheader("Recommended Movies:")

        # Create columns for the movie grid
        cols = st.columns(3)  

        for i, movie_id in enumerate(recommended_movie_ids):
            with cols[i % 3]:  # Distribute movies evenly across columns
                movie_title, movie_description = get_movie_title_and_desc(movie_id)
                try:
                    image = generate_poster(movie_title)  # Generate poster
                    st.markdown(movie_description)
                    #st.image(image, caption=movie_title, width=200)  # Set width for half-size
                    # if st.button(f"Show Description for {movie_title}", key=f"modal_{i}"):
                    #     modal.open_modal(f"Description: {movie_title}", movie_description, key=f"modal_{i}_content")
                except requests.exceptions.HTTPError as err:
                    st.error(f"Image generation failed for {movie_title}: {err}")
            st.markdown(" ") 

                
            