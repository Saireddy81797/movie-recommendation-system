# recommender.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Fill NaN with empty string
movies['genres'] = movies['genres'].fillna('')

# TF-IDF Vectorizer on genres
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Reset index
movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend_movies(title, num_recommendations=5):
    if title not in indices:
        return ["Movie not found in dataset. Try another title."]
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

if __name__ == "__main__":
    print("Recommended movies for 'Toy Story (1995)':")
    print(recommend_movies("Toy Story (1995)"))
