# app.py
import streamlit as st
import pandas as pd
from recommender import recommend_movies

# Load dataset
movies = pd.read_csv("movies.csv")

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.write("Get similar movie recommendations based on genres using TF-IDF & Cosine Similarity.")

# Select movie
movie_list = movies['title'].values
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie, num_recommendations=5)
    st.subheader("Recommended Movies:")
    for i, rec in enumerate(recommendations, start=1):
        st.write(f"{i}. {rec}")
