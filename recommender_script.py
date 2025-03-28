import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def get_movie_recommendations_content(movie_title, metadata, latent_matrix_1, nn, n_recommendations=10):
    idx = metadata[metadata['title'] == movie_title].index[0]
    

    
    distances, indices = nn.kneighbors([latent_matrix_1[idx]], n_neighbors=n_recommendations + 1)

    recommended_movies = [
        metadata.iloc[i]['title'] for i in indices[0] if i != movie_index
    ]
    
    return recommended_movies[:n_recommendations]



def calc_metadata():
    print("Loading data...")
    movies = pd.read_csv('Data/movies.csv')
    ratings = pd.read_csv('Data/ratings.csv')
    tags = pd.read_csv('Data/tags.csv')

    print("Preprocessing data...")
    # Merging movies and tags DataFrames on 'movieId'
    metadata = pd.merge(movies, tags, on='movieId')
    # Combine all tags for each movie into a single string
    metadata['all_tags'] = metadata.groupby('movieId')['tag'].transform(lambda x: ' '.join(x.astype(str)))
    return metadata


def calc_latent_matrix(metadata):
    print("Initializing TfidfVectorizer...")
    # Initialize TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english')

    print("Fit and transform data...")
    # Fit and transform the data
    tfidf_matrix = tfidf.fit_transform(metadata['all_tags'])

    print("Converting to sparse matrix format...")
    tfidf_matrix_sparse = csr_matrix(tfidf_matrix)

    print("Initializing TruncatedSVD...")
    # Reduce dimensions
    svd = TruncatedSVD(n_components=50)
    latent_matrix_1 = svd.fit_transform(tfidf_matrix_sparse)
    return latent_matrix_1

def calc_nn(latent_matrix_1):
    print("Initializing Nearest Neighbors model...")
    # Use NearestNeighbors to find similar movies
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(latent_matrix_1)
    return nn


def get_movie(movie_title):
    metadata = calc_metadata
    latent_matrix_1 = calc_latent_matrix(metadata)
    nn = calc_nn(latent_matrix_1)
    output_movies = get_movie_recommendations_content(movie_title, metadata, latent_matrix_1, nn)
    return output_movies

if __name__ == "__main__":
    metadata = calc_metadata()
    calc_latent_matrix(metadata)        
    print("asking for recommendation...")
    try:
        recommendations = get_movie_recommendations_content('Toy Story (1995)',metadata)
        print(recommendations)
    except IndexError as e:
        print(e)