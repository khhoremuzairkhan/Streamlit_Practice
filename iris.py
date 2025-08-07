elif option == "Predict":
    st.subheader("🌼 Predict Iris Species")

    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.0)

    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                               columns=iris.feature_names)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.iloc[:, :-1])
    model = LogisticRegression(max_iter=200)
    model.fit(X_scaled, df['species'])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"The predicted Iris species is: **{prediction.capitalize()}**")
