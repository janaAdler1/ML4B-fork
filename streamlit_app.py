import pandas as pd
import streamlit as st
import numpy as np
# insert
from streamlit_option_menu import option_menu

st.set_page_config(page_icon="⭐", page_title="Political Party Tweet Classification", layout="wide")

selected = option_menu(None, ["Home", "Dataset",  "Process", 'Live Demo', 'Other'], 
    icons=['house', 'file-earmark-text', "cpu", 'collection-play', "cpu"], 
    menu_icon="cast", default_index=0, orientation="horizontal",
)

if selected=="Home":
    st.markdown("<h1 style='text-align: center'>Political Party Classification</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center'>Our Goal</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center'>The goal of our project is to classify Tweets of german politicans by the political party of the author. However, we don't just want to research the politicians and cathegorize them manually, we want to use Machine Learning algorithms.</p>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center'>Our Team</h2>", unsafe_allow_html=True)

    #present your team
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("image_jan.jpeg")
        st.markdown("<h5 style='text-align: center'>Jan Amend </h5>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center'>4th semester Wirtschaftsinformatik</h6>", unsafe_allow_html=True)
        st.text("TEXT")

    with col2:
        st.image("image.jpeg")
        st.markdown("<h5 style='text-align: center'>Jana Adler </h5>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center'>4th semester Wirtschaftsinformatik</h6>", unsafe_allow_html=True)
        st.text("Hi, my name is Jana and I'm currently part of a dual studies programm \n at DATEV where im focusing on internet security. \n Since ML and AI is a pretty huge deal in web security \n I'm very invested in this topic. \n In my spare time I like to dance and \n go for a ride on my motorcycle.")
    with col3:
        st.image("image_ser.jpg")
        st.markdown("<h5 style='text-align: center'>Sergei Mezhonnov </h5>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center'>4th semester Wirtschaftsinformatik</h6>", unsafe_allow_html=True)
        st.text("TEXT")

if selected=="Dataset":
    st.markdown("<h1 style='text-align: center'>Twitter Dataset</h1>", unsafe_allow_html=True)
    st.text('Our dataset is a JSON file consisting of official tweets from members of the german parliament as of march 2021. Thus it includes tweets from CDU/CSU, SPD, Die Gruenen, Die Linken, AFD, etc. \n The main problem one will soon discover is...')
    
    st.markdown("<h5>...our dataset is 8GB of JL-Data...</h5>", unsafe_allow_html=True)

    st.image("https://i.kym-cdn.com/photos/images/newsfeed/000/173/576/Wat8.jpg?1315930535", caption = "My Notebook with 4GB RAM")
    if st.checkbox("Show me example"):
        data = json.load(open('data.json'))
        st.write(data)

if selected=="Process":
    with st.expander("Business Understanding"):
        st.text("Everybody knows Tweets. You can retweet a tweet or you can create a new one completly on your own. There are almost no limitis to what you can include in your tweet. You can use text, numbers and emojicons.")
        st.text("Despite the almost unlimited possibilites to write a tweet one might use same patterns - like special emojis or syntax - over and over again. Furthermore members of some political parties tend to write more about special topics like 'football' and less about other topics like 'gardening'")
        st.text("The interesting part is to find exactly these patterns. Some are quite obvious and others are rather inconspicuous. However, we do not need to find those patterns on our own and read all of the 5000 tweets, we will use KI-algorithms for this!")

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
if selected=="Other":
    st.download_button('Download some text', text_contents)
