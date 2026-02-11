import pandas as pd
import string
import re
import streamlit as st
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

@st.cache_data(ttl=600)
def load_data():
    sheet_id = "1sIjsvU75DSvjNgGuOc5kXlHv9Q0xpN5HcRbiTcci47c"
    # Use the gid from the URL provided by user
    gid = "1435065871"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    
    try:
        df = pd.read_csv(url)
        df = df.drop_duplicates()
        # Verify required columns exist
        required_columns = ["Title", "Genre", "Tags", "Summary"]
        if not all(col in df.columns for col in required_columns):
            st.error(f"Error: The Google Sheet must contain these columns: {', '.join(required_columns)}")
            return pd.DataFrame()

        df["Title"] = df["Title"].astype(str)
        df["Genre"] = df["Genre"].astype(str)
        df["Tags"] = df["Tags"].astype(str)
        df["Summary"] = df["Summary"].astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        st.warning("Please ensure the Google Sheet is shared with 'Anyone with the link can view'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        st.warning("Please ensure the Google Sheet is shared with 'Anyone with the link can view'.")
        return pd.DataFrame()

netflix_data = load_data()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

netflix_data["combined_features"] = (
    netflix_data["Genre"] + " " +
    netflix_data["Tags"] + " " +
    netflix_data["Summary"]
)

netflix_data["combined_features"] = netflix_data["combined_features"].apply(preprocess_text)

@st.cache_resource
def build_model(data):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(data["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

cosine_sim = build_model(netflix_data)

titles = netflix_data["Title"].tolist()
title_to_idx = {title: idx for idx, title in enumerate(titles)}

def get_recommendations(title, num_recommendations=10):
    if title not in title_to_idx:
        return None
    idx = title_to_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    return netflix_data.iloc[movie_indices]

st.set_page_config(page_title="Netflix Recommender", layout="wide")

st.title("🎬 Netflix Movie Recommendation System")
st.write("Content-Based Filtering using TF-IDF and Cosine Similarity")

movie_name = st.text_input("Enter a movie name:")

if st.button("Get Recommendations"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        recommendations = get_recommendations(movie_name)
        if recommendations is None:
            st.error("Movie not found. Please check spelling.")
        else:
            st.subheader(f"Top Recommendations for '{movie_name}'")
            for _, row in recommendations.iterrows():
                st.markdown(f"## 🎥 {row['Title']}")
                st.write(f"**Genre:** {row['Genre']}")
                st.write(f"**Tags:** {row['Tags']}")
                st.write("**Description:**")
                st.info(row["Summary"])
                st.write("---")
