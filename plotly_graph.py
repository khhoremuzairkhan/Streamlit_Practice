import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


# import datasets
st.title("Plotly and Streamlit Combined App")
df = px.data.gapminder()
st.write(df)

st.subheader("list of columns")
st.write(df.columns)


# data management
year_option = df['year'].unique().tolist()

year=st.selectbox("which year should we plot?",year_option,0)
#df = df[df['year']==year]

# now coming to plotting the basic based on the year selected
fig = px.scatter(df,x='gdpPercap',y='lifeExp',size='pop',
                 color='country',hover_name='continent',log_x =True, size_max=55,range_x=[100,1000],
                 range_y=[20,90],animation_frame='year',animation_group='country')
st.write(fig)
