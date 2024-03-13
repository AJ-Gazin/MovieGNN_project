

import streamlit as st
import requests
from io import BytesIO
from PIL import Image

# Your Hugging Face API Key (keep it secure!)   
API_KEY = "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" 
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"



def generate_image(caption):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "inputs": caption
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error if the request fails

        # Display the generated image
        image = Image.open(BytesIO(response.content))
        st.image(image, caption=caption)

    except requests.exceptions.HTTPError as err:
        st.error(f"Image generation failed: {err}")





st.title("AI Image Generator with Stable Diffusion")
caption = st.text_input("Describe the image you wish to generate:", "photo of a rocket launching into space")

if st.button("Generate Image"):
    # We'll handle image generation in a function below 
    generate_image(caption)



