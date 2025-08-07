import streamlit as st
import seaborn as sns
import pandas as pd
st.header("This header is from Khhorem")
st.text("Kaisa diya? acha nahi diya")
st.header("PAta nahi kiya chal raha hai")
df = sns.load_dataset('iris')
st.write(df.head(10))