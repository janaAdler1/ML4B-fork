import streamlit as st
import pandas as pd
import numpy as np
import json
import linecache

st.title('Political Party Classification')

#present your project
st.subheader("Our Goal")
st.text('The goal of our project is to classify Tweets by the political party of the author')

#present your team
with st.expander("Our Team:"):
    col1, col2, col3 = st.columns(3)
with col1:
    #st.header("Jan Amend")
    st.markdown(f'<h1 style="color:#454545;font-size:16px;">{"Jan Amend"}</h1>', unsafe_allow_html=True)
    st.image("image_jan.jpeg")

with col2:
    #st.header("Jana Adler")
    st.markdown(f'<h1 style="color:#454545;font-size:16px;">{"Jana Adler"}</h1>', unsafe_allow_html=True)
    st.image("image.jpeg")

with col3:
    #st.header("Sergei Mezhonnov")
    st.markdown(f'<h1 style="color:#454545;font-size:16px;">{"Sergei Mezhonnov"}</h1>', unsafe_allow_html=True)
    st.image("https://scontent-frt3-2.xx.fbcdn.net/v/t31.18172-8/20451827_1559088087491677_5562632512013699296_o.jpg?_nc_cat=103&ccb=1-5&_nc_sid=09cbfe&_nc_eui2=AeEdsPUBJzyPOZFs7ylgC0hgD7d7OvNp0kMPt3s682nSQ3wLFAZVTOrmLSOyHzD2mTt9LJKE7h7ZjiAONvU_-HJN&_nc_ohc=6ihYdGJVRsYAX_5hGRe&_nc_ht=scontent-frt3-2.xx&oh=00_AT893UIInBTyBmNpf071lnSEZBuEZS3igGFkVQEABTd9rg&oe=6294469E")

    
#show one element of your dataset
#with st.expander('Example of dataset'):
    #st.header("Party of Schulz Anja is...")
    #data = json.load(open('data.json'))
    #st.text(linecache.getline('data.json',6))
   # st.text('**Tweet of Anja Schulz:** "Das Spiel macht wirklich Freude. Ein Traeumchen"')
    #if st.button("Party of tweeting Member of Parliament is..."):
      #  time.sleep(3)
     #   data = json.load(open('data.json'))
     #   st.markdown(data['account_data']['Partei'])
with st.expander('Example of dataset'):
    st.text('**Tweet of Anja Schulz:** "Das Spiel macht wirklich Freude. Ein Traeumchen"')
    if st.button("Party of tweeting Member of Parliament is..."):
        #time.sleep(3)
        data = json.load(open('data.json'))
        st.markdown(data['account_data']['Partei'])

if st.button("What is a classificator?"):
    placeholder = st.image("https://miro.medium.com/max/1400/1*R6Rbcks-pGO0SkhCINrP0g.png")
    but = st.button("I already know")
    if but:
        placeholer = st.remove()
