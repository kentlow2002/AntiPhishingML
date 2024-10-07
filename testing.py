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


def tester():
    with open("model.pkl", "rb") as f:
        clf = load(f)

    with open("count_vectorizer.pkl", "rb") as f:
        vect = load(f)

    dataset = pd.read_csv(r'emails_with_features.csv').head(1000)
    X_train, X_test, y_train, y_test = train_test_split(dataset[['Body', 'Sentiment']], dataset[['Label']], test_size=0.1, random_state=42, shuffle=True)
    
    test_vect = vect.transform(X_test.Body)
    print(test_vect.shape, X_test.Sentiment.shape)
    new_X_test = hstack([test_vect, transpose([X_test.Sentiment])])
    predictions = clf.predict(new_X_test)
    print(classification_report(y_test, predictions))

if __name__ == '__main__':
    tester()