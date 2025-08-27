import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class MovieRecommender:
    def __init__(self, movies_path="data/movies.csv"):
        self.movies = pd.read_csv(movies_path)
        self.movies["genres"] = self.movies["genres"].fillna("")
        
        # TF-IDF vectorization on genres
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies["genres"])
        
        # Compute cosine similarity
        self.similarity_matrix = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def recommend(self, movie_title, top_n=5):
        if movie_title not in self.movies["title"].values:
            return [f"Movie '{movie_title}' not found in database."]
        
        idx = self.movies.index[self.movies["title"] == movie_title][0]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[1:top_n+1]  # skip the first (itself)
        
        recommended = [self.movies.iloc[i[0]]["title"] for i in scores]
        return recommended
