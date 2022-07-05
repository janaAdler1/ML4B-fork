import pandas as pd
import streamlit as st
import numpy as np
# insert
from streamlit_option_menu import option_menu

st.set_page_config(page_icon="⭐", page_title="Political Party Tweet Classification", layout="wide")

selected = option_menu(None, ["Home", "Dataset",  "Process", 'Live Demo', 'Blub'], 
    icons=['house', 'file-earmark-text', "cpu", 'collection-play', "cpu"], 
    menu_icon="cast", default_index=0, orientation="horizontal",
)

if selected=="Home":
    st.markdown("<h1 style='text-align: center'>Political Party Classification</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center'>Our Goal</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center'>The goal of our project is to classify Tweets of german politicans by the political party of the author. However, we don't just want to research the politicians and cathegorize them manually, we want to use Machine Learning algorithms.</h4>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center'>Our Team</h2>", unsafe_allow_html=True)

    #present your team
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image_jan.jpeg", caption = 'Jan Amend')
        st.text("4th semester Wirtschaftsinformatik")
        st.text("TEXT")
    with col2:
        st.image("image.jpeg", caption = "Jana Adler")
        st.text("4th semester Wirtschaftsinformatik")
        st.text("Hi, my name is Jana and I'm currently part of a dual studies programm at DATEV where im focusing on internet security. Since ML and AI is a pretty huge deal in web security I'm very interested in this topic. In my spare time I like to dance and go for a ride on my motorcycle.")
    with col3:
        st.image("image_ser.jpg", caption = 'Sergei Mezhonnov')
        st.text("4th semester Wirtschaftsinformatik")
        st.text("TEXT")

if selected=="Dataset":
    st.text('Our dataset is 8GB of JL-Data...')
    st.image("https://i.kym-cdn.com/photos/images/newsfeed/000/173/576/Wat8.jpg?1315930535", caption = "My Notebook with 4GB RAM")
    if st.checkbox("Show me example"):
        data = json.load(open('data.json'))
        st.write(data)   

if selected=="Process":
    with st.expander("Business Understanding"):
        st.text("Everybody knows Tweets")
    with st.expander("Data Understanding"):
        st.text("blub")
    with st.expander("Data Preparation"):
        st.text("Before Analyse to start we need to prepare our dataframe.")
        st.text("To do this, we use several functions:")
        if st.checkbox("Count of Tweets"):
            fig1 = plt.figure(figsize=(8,6))
            df.groupby('party').tweet_prep.count().plot.bar(ylim=0)
            st.pyplot(fig1)
    with st.expander("Modeling"):
        st.text("blub")
    with st.expander("Evaluation"):
        st.text("blub")
    with st.expander("Deployment"):
        st.text("blub")

if selected=="Live Demo":  
    if st.button("Prepare"):
        st.write("prepare")
    
    option = st.selectbox('ML Model', 
        ["Naive Bayes",
         "Linear Support Vector Machine", 
         "Logistic Regression","Test"])
    
    if option == 'Naive Bayes':
        st.write("nb")

        if st.button("Predict"):
            st.write("nb_pred")
        
        if st.button("Evaluation"):
            st.text('Model Report:\n')
            
    
    elif option == 'Linear Support Vector Machine':
        
        if st.button("Predict"):
            st.write("sgd_pred")
            
        if st.button("Evaluation"):
            st.write("sgd_pred")
    
    elif option == 'Logistic Regression':
       
        if st.button("Predict"):
            st.write("sgd_pred")
            
        if st.button("Evaluation"):
            st.write("sgd_pred")

