from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from explainerdashboard import ClassifierExplainer,ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, feature_descriptions
import streamlit as st


X_train,y_train,X_test,y_test = titanic_survive()
st.write(X_train.head())
st.write(X_train.shape)
st.write(X_test.head())
st.write(X_test.shape)
st.write(y_train.head())
st.write(y_train.shape)
st.write(y_test.head())
st.write(y_test.shape)
#feature_descriptions_dict = dict(zip(feature_descriptions.feature, feature_descriptions.description))

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = RandomForestClassifier()
model.fit(X_train,y_train)

# now using explainer for complete dashbaord
#
explainer = ClassifierExplainer(model, X_test,y_test,
                                cats=['Sex','Deck','Embarked'],
                                labels=['survived','not survived'])

ExplainerDashboard(explainer).run()