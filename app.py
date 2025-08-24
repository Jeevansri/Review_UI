import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("student-scores.csv")

# Subjects to predict
subjects = [
    "math_score", "history_score", "physics_score",
    "chemistry_score", "biology_score", "english_score", "geography_score"
]

# Define class label function
def score_class_label(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

# Simple model: fit linear regression for each subject
models = {}
for subject in subjects:
    X = df[["weekly_self_study_hours"]]
    y = df[subject]
    model = LinearRegression().fit(X, y)
    models[subject] = model

st.title("Student Score Class Predictor")

st.write("Enter weekly self-study hours to predict your class label for each subject:")

# Add custom CSS for colors and style
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f4f8;
    }
    .stApp {
        background-color: #f0f4f8;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2a5298;
    }
    .stButton>button {
        background-color: #2a5298;
        color: white;
        border-radius: 8px;
        padding: 0.5em 2em;
        font-weight: bold;
    }
    .stTable th {
        background-color: #2a5298;
        color: white;
    }
    .stTable td {
        background-color: #e3eafc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

hours = st.number_input("Weekly Self-Study Hours", min_value=0, max_value=60, value=10)

if st.button("Predict"):
    results = {}
    for subject in subjects:
        pred_score = models[subject].predict([[hours]])[0]
        label = score_class_label(pred_score)
        results[subject.replace("_score", "").capitalize()] = f"{label} ({pred_score:.1f})"
    st.write("### Predicted Class Labels:")
    st.table(pd.DataFrame(results.items(), columns=["Subject", "Class Label (Predicted Score)"]))
