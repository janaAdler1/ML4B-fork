import pandas as pd
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(page_icon="🕊️", page_title="Political Party Tweet Classification")
"""selected = option_menu(None, ["Home", "Dataset", "Data Preparation", 'Live Demo'], 
    icons=['house', 'file-earmark-text', "cpu", 'collection-play'], 
    menu_icon="cast", default_index=0, orientation="horizontal"
    styles={
        "container": {"padding": "50px", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
        }
    )"""
selected = option_menu(None, ["Home", "Dataset",  "Data Preparation", 'Live Demo'], 
    icons=['house', 'file-earmark-text', "cpu", 'collection-play'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "50px", "background-color": "#F5F5F5"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#ff751a"},
        "nav-link-selected": {"background-color": "#990033"},
    }
)
selected
st.title('Political Party Classification')
#present your project
st.subheader("Our Goal")
st.text('The goal of our project is to classify Tweets by the political party of the author')

# 1. as sidebar menu


#present your team
with st.expander("Our Team:"):
    col1, col2, col3 = st.columns(3)
with col1:
    st.image("image_jan.jpeg", caption = 'Jan Amend')
with col2:
    st.image("image.jpeg", caption = "Jana Adler")

with col3:
    st.image("image_ser.jpg", caption = 'Sergei Mezhonnov')

with st.expander('Example of dataset'):
    st.text('Our dataset is 8GB of JL-Data...')
    st.image("https://i.kym-cdn.com/photos/images/newsfeed/000/173/576/Wat8.jpg?1315930535", caption = "My Notebook with 4GB RAM")
    if st.checkbox("Show me example"):
        data = json.load(open('data.json'))
        st.write(data)   

with st.expander('Data Preparation'):
    st.text("Before Analyse to start we need to prepare our dataframe.")
    st.text("To do this, we use several functions:")
    if st.checkbox("Count of Tweets"):
        fig1 = plt.figure(figsize=(8,6))
        df.groupby('party').tweet_prep.count().plot.bar(ylim=0)
        st.pyplot(fig1)
    

with st.expander("Prediction"):
   
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

