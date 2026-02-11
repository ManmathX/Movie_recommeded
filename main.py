import pandas as pd
import string
import re
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("NetflixDataset.csv", encoding="latin-1", index_col="Title")
    df = df.drop_duplicates()

    df["Genre"] = df["Genre"].astype(str)
    df["Tags"] = df["Tags"].astype(str)
    df["Summary"] = df["Summary"].astype(str)

    return df

netflix_data = load_data()

# ---------------- Preprocessing ----------------
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text

netflix_data["combined_features"] = (
    netflix_data["Genre"] + " " +
    netflix_data["Tags"] + " " +
    netflix_data["Summary"]
)

netflix_data["combined_features"] = netflix_data["combined_features"].apply(preprocess_text)

# ---------------- Vectorization ----------------
@st.cache_resource
def build_model(data):
    vectorizer = CountVectorizer(stop_words="english")
    count_matrix = vectorizer.fit_transform(data["combined_features"])
    cosine_sim = cosine_similarity(count_matrix)
    return cosine_sim

cosine_sim = build_model(netflix_data)

titles = netflix_data.index.tolist()
title_to_idx = {title: idx for idx, title in enumerate(titles)}

# ---------------- Recommendation Function ----------------
def get_recommendations(title, num_recommendations=10):
    if title not in title_to_idx:
        return None

    idx = title_to_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]

    movie_indices = [i[0] for i in sim_scores]
    return netflix_data.iloc[movie_indices]

# ---------------- Streamlit UI ----------------
st.title("🎬 Netflix Movie Recommendation System")

movie_name = st.text_input("Enter a movie name:")

if st.button("Get Recommendations"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        recommendations = get_recommendations(movie_name)

        if recommendations is None:
            st.error("Movie not found. Try another title.")
        else:
            st.subheader(f"Recommended movies for '{movie_name}'")
            for index, row in recommendations.iterrows():
                st.markdown(f"**{index}**")
                st.write(row['Summary'])
                st.markdown("---")
