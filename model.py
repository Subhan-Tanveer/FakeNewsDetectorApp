# model.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np

# Load and preprocess the data
news_dataset = pd.read_csv("train.csv")
news_dataset = news_dataset.fillna('')
news_dataset['content'] = news_dataset['title'] + " " + news_dataset['author']
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

# Initialize stemmer
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Vectorize the data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# Split data and train the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)

# Define a function for making predictions
def predict_news(input_text):
    input_data = vectorizer.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]
