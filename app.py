import streamlit as st
from recommender import MovieRecommender

# Load recommender
recommender = MovieRecommender("data/movies.csv")

st.title("🎬 Movie Recommendation System")
st.write("Get similar movie suggestions based on genres using content-based filtering.")

movie_name = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    if movie_name:
        recommendations = recommender.recommend(movie_name, top_n=5)
        st.subheader("Recommended Movies:")
        for r in recommendations:
            st.write(f"- {r}")
    else:
        st.write("⚠️ Please enter a movie name.")
