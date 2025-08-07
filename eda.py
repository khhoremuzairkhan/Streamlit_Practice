import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import profile_report
from streamlit_pandas_profiling import st_profile_report

# Title
st.markdown('''
#  **Exploratory Data Analysis**
Build by Khhorem
            ''')


with st.sidebar.header("Upload your csv dataset"):
    uploaded_file = st.sidebar.file_uploader("Upload your file ",type=['csv'])
    df=sns.load_dataset('titanic') 
    st.sidebar.markdown("[Example CSV File](df)")