from pickle import load
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.metrics import classification_report
from numpy import transpose


def tester(input_text, sender_domain):
    with open("model.pkl", "rb") as f:
        clf = load(f)

    with open("count_vectorizer.pkl", "rb") as f:
        count_vectorizer = load(f)

    with open("hash_vectorizer.pkl", "rb") as f:
        hash_vectorizer = load(f)
    

    sentiment = TextBlob(input_text).sentiment.polarity
    hashed_domain = hash_vectorizer.transform([sender_domain])
    word_count = count_vectorizer.transform([input_text])
    X_user = hstack([word_count, [sentiment], hashed_domain])
    prediction = clf.predict(X_user)
    return prediction
    

if __name__ == '__main__':
    tester()