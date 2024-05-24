import numpy as np
import pandas as pd
import joblib
import re
import contractions

# Apply the same preprocessing to the new data
def preprocess_data(df):
    # Converting YEAR to string temporarily for string-specific operations
    df['YEAR'] = df['YEAR'].astype(str)

    # Data cleaning methods
    df['GENRE'] = df['GENRE'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
    df['MOVIES'] = df['MOVIES'].str.lower().str.strip().apply(remove_special_characters).apply(lambda x: ' '.join(x.split()))
    df['GENRE'] = df['GENRE'].str.lower().str.strip().apply(remove_special_characters).apply(lambda x: ' '.join(x.split()))
    df['ONE-LINE'] = df['ONE-LINE'].str.lower().str.strip().apply(remove_special_characters).apply(lambda x: ' '.join(x.split()))
    df['STARS'] = df['STARS'].str.lower().str.strip().apply(remove_special_characters).apply(lambda x: ' '.join(x.split()))
    df['VOTES'] = df['VOTES'].replace(',', '').astype(float)

    # Extracting four-digit year and converting back to float
    df['YEAR'] = df['YEAR'].str.extract(r'(\d{4})').astype(float)

    # Removing emojis and expanding contractions
    df['MOVIES'] = df['MOVIES'].apply(remove_emojis).apply(contractions.fix)
    df['GENRE'] = df['GENRE'].apply(remove_emojis).apply(contractions.fix)
    df['ONE-LINE'] = df['ONE-LINE'].apply(remove_emojis).apply(contractions.fix)
    df['STARS'] = df['STARS'].apply(remove_emojis).apply(contractions.fix)

    return df


def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

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

data = {
    'MOVIES': ['blood red sky'],
    'YEAR': [2013.00],
    'GENRE': [['history', 'horror', 'comedy']],
    'RATING': [6.10],
    'ONE-LINE': ['a woman with a mysterious illness is forced in...'],
    'STARS': ['director peter thorwarth stars peri baumeister...'],
    'VOTES': [210620.00],
    'RunTime': [64.00]
}

# Create DataFrame
df = pd.DataFrame(data)
df = preprocess_data(df)

# Assuming 'MOVIES' and 'RATING' are not part of features
X = df.drop(['MOVIES', 'RATING'], axis=1)

# Classifier model predictions
classifier_model_files = [
    'decision_tree_classifier.pkl', 'gradient_booster_classifier.pkl', 'KNN_classifier.pkl', 
    'logistic_regression_classifier.pkl', 'random_forest_classifier.pkl', 'single_vector_classifier.pkl'
]

for model_file in classifier_model_files:
    model = joblib.load(model_file)
    y_pred = model.predict(X)
    print(f'Predictions from {model_file}:', y_pred)

# Regression model predictions
regression_model_files = [
    'gradient_booster_regressor.pkl', 'knn_regressor.pkl', 'linear_regressor.pkl', 'random_forest_regressor.pkl','svm_regressor.pkl'
]

for model_file in regression_model_files:
    model = joblib.load(model_file)
    y_pred = model.predict(X)
    print(f'Prediction values from {model_file}:', y_pred)
