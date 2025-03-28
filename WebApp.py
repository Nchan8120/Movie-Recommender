import streamlit as st
import pandas as pd
import os
from recommender_script import get_movie_recommendations_content  # Import your function
from recommender_script import calc_metadata 

st.title("Movie Recommender")
if st.button("Train Model"):

    
user_input = st.text_input("Enter a movie you like:")
if st.button("Get Recommendations"):
    recommendations = get_movie_recommendations_content(user_input)
    st.write(recommendations)

if st.button("Quit App"):
    st.write("Shutting down...")
    os._exit(0)  # Forcefully exits the app
