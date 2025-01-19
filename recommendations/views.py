from django.shortcuts import render

# Create your views here.
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from django.http import JsonResponse

# Assuming you have a DataFrame 'movies' loaded with relevant data
movies = pd.read_csv('recommendations/movies.csv')  # Load your dataset

# Fill NaN values in genres
movies['genres'] = movies['genres'].fillna('')

# Create a simplified title column for case-insensitive and year-agnostic matching
movies['simplified_title'] = movies['title'].apply(
    lambda x: re.sub(r'\s*\(\d{4}\)', '', x).strip().lower()
)

# Vectorize genres using CountVectorizer
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = vectorizer.fit_transform(movies['genres'])

# Compute cosine similarity between movies
cosine_sim = cosine_similarity(genre_matrix)

def recommend_movies(movie_title, num_recommendations=5):
    simplified_input = re.sub(r'\s*\(\d{4}\)', '', movie_title).strip().lower()
    
    if simplified_input not in movies['simplified_title'].values:
        return {"error": f"'{movie_title}' not found in the dataset!"}

    # Get the index of the movie
    idx = movies[movies['simplified_title'] == simplified_input].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top recommendations (excluding the input movie itself)
    sim_scores = sim_scores[1:num_recommendations + 1]
    
    # Fetch movie titles
    recommendations = [movies.iloc[i[0]]['title'] for i in sim_scores]
    return {"recommendations": recommendations}

def recommend_view(request):
    movie_title = request.GET.get('movie')
    num_recommendations = int(request.GET.get('num_recommendations', 5))
    
    # Call the recommend_movies function
    recommendations = recommend_movies(movie_title, num_recommendations)
    
    # Return recommendations as JSON response
    return JsonResponse(recommendations)
