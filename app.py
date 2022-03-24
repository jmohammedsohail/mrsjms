
import pickle
import pandas as pd
import streamlit as st
import requests
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# data = pickle.load(open('data.pkl','rb'))

df1 = pd.read_csv('./model/netflix_titles.csv')

df1['combined'] = df1['description']+df1['cast']+df1['director']+df1['listed_in']+df1['country']
# df1['combined'].head(5)

from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(min_df = 3,max_features = None,analyzer = 'word',token_pattern = 'r\w{1,}', ngram_range = (1,3), stop_words = 'english')

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df1['combined'] = df1['combined'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df1['combined'])

#Output the shape of tfidf_matrix
# tfidf_matrix.shape


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)



def get_recommendations(movie):
    idx = movies[movie]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    
    recommended_movie_names=df1['title'].iloc[movie_indices].values
    
    return  recommended_movie_names
# load the movie_list model from disk
movies = pickle.load(open('movie_list.pkl','rb'))


st.header('Netfilx Movie Recommender System')
st.subheader('By Mohammed Sohail')

movie_list = movies.index
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

   
if st.button('Show Recommendation'):
    recommended_movie_names = get_recommendations(selected_movie)

    st.title("Top 10 Most Similar Movies From Netflix!")
    
    for i in range(0, 10):
        cols = st.columns(1)
        cols[0].write(recommended_movie_names[i])  
    st.success('Done!')  
    



