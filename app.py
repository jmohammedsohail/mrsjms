
import pickle
import streamlit as st
import requests


def get_recommendations(movie):
    idx = movies[movie]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    
    recommended_movie_names=data['title'].iloc[movie_indices].values
    
    return  recommended_movie_names
# load the movie_list model,data, and tfidf vectorizer from disk
movies = pickle.load(open('movie_list.pkl','rb'))
cosine_sim = pickle.load(open('cosine_sim.pkl','rb'))
data = pickle.load(open('data.pkl','rb'))
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
    



