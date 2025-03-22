import pickle
import string

import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

import nltk
nltk.download('punkt')
nltk.download('stopwords')


# Load CountVectorizer and TfidfTransformer
with open('count_vectorizer.pkl', 'rb') as file:
    count_vectorizer = pickle.load(file)

with open('tfidf_transformer.pkl', 'rb') as file:
    tfidf_transformer = pickle.load(file)

# Load Logistic Regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    logistic_regression_model = pickle.load(file)

# Function to remove punctuation
def remove_punc(text):
    new_text = [x for x in text if x not in string.punctuation]
    new_text = ''.join(new_text)
    return new_text

# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_words = ' '.join(filtered_words)
    return filtered_words

# Preprocess text data
def preprocess_text(text):
    # Remove punctuation
    text = remove_punc(text)
    # Remove stopwords
    text = remove_stopwords(text)
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        preprocessed_text = preprocess_text(text)
        
        # Vectorize and transform the input text
        text_counts = count_vectorizer.transform([preprocessed_text])
        text_tfidf = tfidf_transformer.transform(text_counts)
        
        # Make prediction
        prediction = logistic_regression_model.predict(text_tfidf)
        
        # Convert prediction to readable format
        if prediction[0] == 1:
            result = "True News"
        else:
            result = "Fake News"
        
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
