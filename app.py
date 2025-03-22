import pickle
import string

import nltk
from flask import Flask, render_template, request
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)


# Load CountVectorizer and TfidfTransformer
with open('count_vectorizer.pkl', 'rb') as file:
    count_vectorizer = pickle.load(file)

with open('tfidf_transformer.pkl', 'rb') as file:
    tfidf_transformer = pickle.load(file)

# Load Logistic Regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    logistic_regression_model = pickle.load(file)



# Define a basic set of English stopwords manually
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being", "i", "me", "my", 
    "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", 
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", 
    "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", 
    "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", 
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", 
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", 
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", 
    "don", "should", "now"
}

# Function to remove punctuation
def remove_punc(text):
    return ''.join(char for char in text if char not in string.punctuation)

# Function to remove stopwords manually
def remove_stopwords(text):
    words = text.split()  # Manual tokenization (split by spaces)
    filtered_words = [word for word in words if word.lower() not in STOPWORDS]
    return ' '.join(filtered_words)

# Preprocess text data
def preprocess_text(text):
    text = remove_punc(text)  # Remove punctuation
    text = remove_stopwords(text)  # Remove stopwords
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
