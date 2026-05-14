import streamlit as st
import math
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


st.title("KNN Classifier with Iris Dataset")

st.header("1. Euclidean Distance Example")

x1 = st.number_input("x1", value=2.0)
y1 = st.number_input("y1", value=3.0)
x2 = st.number_input("x2", value=6.0)
y2 = st.number_input("y2", value=7.0)

distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

st.write(f"Euclidean Distance: {distance:.2f}")

st.header("2. Load Iris Dataset")

data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

st.dataframe(df.head())

st.header("3. Train KNN Model")

k_value = st.slider("Select K value", min_value=1, max_value=15, value=3)

metric = st.selectbox(
    "Select Distance Metric",
    ["minkowski", "euclidean", "manhattan"]
)

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

if metric == "minkowski":
    knn = KNeighborsClassifier(
        n_neighbors=k_value,
        metric="minkowski",
        p=2
    )
else:
    knn = KNeighborsClassifier(
        n_neighbors=k_value,
        metric=metric
    )

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

st.subheader("Model Evaluation")

st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Precision: {precision:.4f}")
st.write(f"Recall: {recall:.4f}")
st.write(f"F1 Score: {f1:.4f}")

st.subheader("Classification Report")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    columns=data.target_names,
    index=data.target_names
)

st.dataframe(cm_df)

st.success("KNN Classification Completed Successfully!")
