import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv(
    r'C:/Users/aarya/OneDrive/Pictures/Desktop/movie-recommender-system/data/movies.csv',
    encoding='latin-1',
    engine='python',
    on_bad_lines='skip'
)

movies.head()
movies['Genre'] = movies['Genre'].str.replace('|', ' ')
movies['MovieName'] = movies['MovieName'].str.lower()

tfidf = TfidfVectorizer(stop_words='english')
vectors = tfidf.fit_transform(movies['Genre']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie_name):
    movie_name = movie_name.lower()
    
    if movie_name not in movies['MovieName'].values:
        return ["Movie not found!"]
    
    movie_index = movies[movies['MovieName'] == movie_name].index[0]
    
    distances = similarity[movie_index]
    
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    return [movies.iloc[i[0]].MovieName for i in movie_list]

st.title("🎬 Movie Recommender System")

movie_input = st.text_input("Enter Movie Name")

if st.button("Recommend"):
    results = recommend(movie_input)
    
    for movie in results:
        st.write(movie)