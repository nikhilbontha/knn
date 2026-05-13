import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Title
st.title(" KNN Regression using Streamlit")
st.write("Predict California housing prices using KNN Regression.")

# Load Dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target
feature_names = housing.feature_names

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Select K Value
st.subheader("Model Settings")
k_value = st.slider("Select K Value", 1, 20, 5)

# Train Model
model = KNeighborsRegressor(n_neighbors=k_value)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display Metrics
st.subheader(" Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MSE", f"{mse:.4f}")

with col2:
    st.metric("MAE", f"{mae:.4f}")

with col3:
    st.metric("R² Score", f"{r2:.4f}")

# Prediction Section
st.subheader(" Predict Housing Price")

user_input = []

for feature in feature_names:
    value = st.number_input(
        f"Enter {feature}",
        value=0.0,
        format="%.2f"
    )
    user_input.append(value)

# Prediction Button
if st.button("Predict Price"):
    prediction = model.predict([user_input])

    st.success(
        f"Predicted House Price: ${prediction[0] * 100000:.2f}"
    )

# Dataset Preview
if st.checkbox("Show Dataset Sample"):
    df = pd.DataFrame(X, columns=feature_names)
    df["Target"] = y

    st.dataframe(df.head())

# Footer
st.write("---")
st.write("Developed using Streamlit & Scikit-learn")