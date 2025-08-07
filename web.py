import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------- Login UI ----------
st.title("Login Required")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if username == "admin" and password == "admin":
    st.success("Logged in as admin")

    # ---------- App development starting ----------
    st.write('''
    # Explore different ML Models and Datasets
    Attempting to see what is the best among them        
    ''')

    dataset_name = st.sidebar.selectbox('Select the Dataset', ('Iris', 'Breast Cancer', 'Wine'))
    classifier_name = st.sidebar.selectbox('Select the ML Algo', ('KNN', 'SVM', 'Random Forest'))

    def get_dataset(dataset_name):
        if dataset_name == "Iris":
            data = datasets.load_iris()
        elif dataset_name == "Wine":
            data = datasets.load_wine()
        else:
            data = datasets.load_breast_cancer()
        return data.data, data.target

    X, y = get_dataset(dataset_name)

    st.write("Shape of Dataset is:", X.shape)
    st.write("Number of classes:", len(np.unique(y)))

    def add_param_ui(classifier_name):
        params = {}
        if classifier_name == "SVM":
            C = st.sidebar.slider('C', 0.01, 10.0)
            params['C'] = C
        elif classifier_name == "KNN":
            K = st.sidebar.slider('K', 1, 15)
            params['K'] = K
        else:
            max_depth = st.sidebar.slider('Max Depth', 2, 15)
            n_estimators = st.sidebar.slider('n_estimators', 1, 100)
            params['max_depth'] = max_depth
            params['n_estimators'] = n_estimators
        return params

    params = add_param_ui(classifier_name)

    def get_classifier(classifier_name, params):
        if classifier_name == 'SVM':
            return SVC(C=params['C'])
        elif classifier_name == "KNN":
            return KNeighborsClassifier(n_neighbors=params['K'])
        else:
            return RandomForestClassifier(max_depth=params['max_depth'],
                                          n_estimators=params['n_estimators'], random_state=1234)

    clf = get_classifier(classifier_name, params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    st.write(f'Classifier = {classifier_name}')
    st.write(f'Accuracy = {accuracy:.2f}')

    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    st.pyplot(fig)

else:
    pass


