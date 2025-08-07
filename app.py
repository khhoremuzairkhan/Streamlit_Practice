import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# make containers
header = st.container()
datasets = st.container()
features = st.container()
model_training = st.container()
df = sns.load_dataset("titanic")

with header:
    st.title("Kashti ki app")
    st.text("In this project we will work with python")

with datasets:
    st.header("Kashti doob gaye")
    df.dropna(inplace=True)
    st.write(df.head())
    st.subheader("Plot for Gender ratio")
    st.bar_chart(df['sex'].value_counts())
    
    
    st.subheader("Plot for Passenger Class ratio")
    st.bar_chart(df['pclass'].value_counts())
    
with features:
    st.header("These are our app features")
    
with model_training:
    st.header("Kashti walon ka kya bana")
    # creating different columns
    features, display = st.columns(2)
    
    # feature selection
    max_depth = features.slider("How Many People",min_value=10,max_value=100,step=5,value=20)
    n_estimators = features.selectbox("How many trees in random forest",options=[50,100,500,1000,'No Limit'])
    features.write(df.columns)
    input_features = features.text_input("Which feature we should use?")
    
# Machine Learning column
    model = RandomForestRegressor(max_depth=max_depth, 
                                  n_estimators=n_estimators)
# define X and y
    X = df[[input_features]]
    y = df[['fare']]
    model.fit(X,y)
    pred = model.predict(X)


#Display metrics
    display.subheader("Mean Squared error is", )
    display.write(mean_squared_error(y,pred))
    display.subheader("Mean Absolute error is", )
    display.write(mean_absolute_error(y,pred))
    display.subheader("R Squared of this model is", )
    display.write(r2_score(y,pred))