import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load API URL from environment
API_URL = os.getenv("API_KEY") + "/recommend"

st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter a movie name")
num_recommendations = st.slider("Number of recommendations", 1, 20, 10)

if st.button("Get Recommendations"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        response = requests.post(
            API_URL,
            json={
                "title": movie_name,
                "num_recommendations": num_recommendations
            }
        )

        if response.status_code == 200:
            data = response.json()

            if "error" in data:
                st.error(data["error"])
            else:
                st.subheader("Recommended Movies:")
                for movie in data["recommendations"]:
                    st.write("•", movie)
        else:
            st.error("API connection failed.")
