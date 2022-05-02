import streamlit as st
import pandas as pd
import numpy as np
st.title('Political Party Classification Group 2')

with st.expander("Our Team"):
  st.write("TEAM ")
with st.expander("Our Goal"):
  st.write("We want to classify tweets.")
with st.expander("Our Data"):
  st.write("Datasetexamples")
  col1, col2, col3 = st.columns(3)

with col1:
    st.header("Chrysler Logo")

with col2:
    st.header("NIKE Shoe")

with col3:
    st.header("Girl in a white dress")
