import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import contractions

# Set page config
st.set_page_config(page_title="Movie Popularity Prediction", layout="wide")

# Custom CSS to inject for styling
st.markdown("""
<style>
body {
    background-color: #F8F0E3;
    color: #000000;
}
h1 {
    color: #d2042d;
}
.stTextInput>label, .stNumberInput>label, .stTextarea>label {
    color: #000000;
}
.stButton>button {
    background-color: #d2042d;
    color: #ffffff;
}
/* Styling sliders to match the color palette */
.stSlider .stThumb {
    background-color: #d2042d;
}
.stSlider .stTrack {
    background-color: #d2042d;
}
</style>
""", unsafe_allow_html=True)

# Function to remove special characters from text
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Function to remove emojis from text
def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Data preprocessing function
def preprocess_data(df):
    df['YEAR'] = df['YEAR'].astype(str)
    df['GENRE'] = df['GENRE'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
    df['MOVIES'] = df['MOVIES'].str.lower().str.strip().apply(remove_special_characters).apply(lambda x: ' '.join(x.split()))
    df['GENRE'] = df['GENRE'].str.lower().str.strip().apply(remove_special_characters).apply(lambda x: ' '.join(x.split()))
    df['ONE-LINE'] = df['ONE-LINE'].str.lower().str.strip().apply(remove_special_characters).apply(lambda x: ' '.join(x.split()))
    df['STARS'] = df['STARS'].str.lower().str.strip().apply(remove_special_characters).apply(lambda x: ' '.join(x.split()))
    df['VOTES'] = df['VOTES'].replace(',', '').astype(float)
    df['YEAR'] = df['YEAR'].str.extract(r'(\d{4})').astype(float)
    df['MOVIES'] = df['MOVIES'].apply(remove_emojis).apply(contractions.fix)
    df['GENRE'] = df['GENRE'].apply(remove_emojis).apply(contractions.fix)
    df['ONE-LINE'] = df['ONE-LINE'].apply(remove_emojis).apply(contractions.fix)
    df['STARS'] = df['STARS'].apply(remove_emojis).apply(contractions.fix)
    return df

# Display a custom styled header
st.markdown('<h1>Welcome here where you can predict how popular your movie is going to be or it would have been in another era </h1>', unsafe_allow_html=True)
st.markdown('<h2> Enter your movie details here : </h2>', unsafe_allow_html=True)
# Creating two columns for input and output
col1, col2 = st.columns((2, 3))  # Adjust the ratio if needed

with col1:
    with st.form("input_form"):
        movie = st.text_input("Movie name", placeholder="Name your movie however but not needed for analysis",disabled=True)
        year = st.slider("Year", min_value=1930, max_value=2024, value=2005, step=1, format="%d")
        genre = st.text_input("Genre (comma-separated)", placeholder="e.g., Action, Thriller")
        rating = st.slider("Rating", min_value=0.0, max_value=10.0, value=7.5, step=0.1, format="%f",disabled=True)
        one_line = st.text_area("One-line description", placeholder="Enter a brief description of the movie")
        stars = st.text_area("Stars", placeholder="e.g., Leonardo DiCaprio, Joseph Gordon-Levitt")
        votes = st.slider("Votes", min_value=0, max_value=2000000, value=50000, step=1000, format="%d")
        runtime = st.slider("Runtime in minutes", min_value=0, max_value=500, value=120, step=1, format="%d")
        submitted = st.form_submit_button("Submit and Predict")

if submitted:
    data = {'MOVIES': [movie], 'YEAR': [year], 'GENRE': [[g.strip() for g in genre.split(',')]],
            'RATING': [rating], 'ONE-LINE': [one_line], 'STARS': [stars], 'VOTES': [votes], 'RunTime': [runtime]}
    df = pd.DataFrame(data)
    df = preprocess_data(df)
    X = df.drop(['MOVIES', 'RATING'], axis=1)

with col2:
    if submitted:
        st.write("## Prediction Summary")
        classifier_results = {}
        classifier_model_files = [
            'decision_tree_classifier.pkl', 'gradient_booster_classifier.pkl', 'KNN_classifier.pkl', 
            'logistic_regression_classifier.pkl', 'random_forest_classifier.pkl', 'single_vector_classifier.pkl'
        ]
        # Collect predictions
        for model_file in classifier_model_files:
            with open(model_file, 'rb') as file:
                model = pickle.load(file)
            y_pred = model.predict(X)[0]  # Assume y_pred is an array with a single element
            model_name = model_file.replace('_classifier.pkl', '').replace('_', ' ').title()
            classifier_results[model_name] = y_pred

        # Counting occurrences of each prediction category
        prediction_counts = {'Hit': 0, 'Average': 0, 'Flop': 0}
        for result in classifier_results.values():
            if result in prediction_counts:
                prediction_counts[result] += 1

        # Displaying the ratio of Hit : Average : Flop
        summary_display = f"<h3 style='text-align: center; font-weight: bold; font-size: 48px; color: #d2042d;'>Hit : Average : Flop</h3>"
        summary_display += f"<h3 style='text-align: center; font-weight: bold; font-size: 48px; color: #505160;'>{prediction_counts['Hit']} : {prediction_counts['Average']} : {prediction_counts['Flop']}</h3>"
        st.markdown(summary_display, unsafe_allow_html=True)

        # Displaying detailed classifier predictions
        st.write("### Detailed Predictions from Classifiers")
        for model_name, prediction in classifier_results.items():
            st.markdown(f"Our {model_name} predicts that the movie would be a: <span style='color: #d2042d;'>{prediction}</span>", unsafe_allow_html=True)

        # Regressor predictions - Calculating average rating
        st.write("### Predictions from Regressors")
        regression_model_files = [
            'gradient_booster_regressor.pkl', 'knn_regressor.pkl', 'linear_regressor.pkl', 
            'random_forest_regressor.pkl', 'svm_regressor.pkl'
        ]
        ratings = []
        for model_file in regression_model_files:
            with open(model_file, 'rb') as file:
                model = pickle.load(file)
            y_pred = model.predict(X)[0]  # Assume y_pred is an array with a single element
            if y_pred <= 10 and y_pred >= 1:  # Only consider ratings that are 10 or below
                ratings.append(y_pred)

        # Calculate and display the average rating
        if ratings:
            average_rating = sum(ratings) / len(ratings)
            average_rating_display = f"<h3 style='text-align: center; font-weight: bold; font-size: 48px;'>The average rating predicted by all our models is</h3>"
            average_rating_display += f"<h3 style='text-align: center; font-weight: bold; font-size: 48px; color: #d2042d;'>{average_rating:.1f}</h3>"
            st.markdown(average_rating_display, unsafe_allow_html=True)
        
        # Displaying individual regressor predictions
        for model_file in regression_model_files:
            with open(model_file, 'rb') as file:
                model = pickle.load(file)
            y_pred = model.predict(X)[0]
            model_name = model_file.replace('_regressor.pkl', '').replace('_', ' ').title()
            st.markdown(f"**{model_name} predicts the rating to be**: <span style='color: #d2042d;'>{y_pred:.1f}</span>", unsafe_allow_html=True)