from itertools import count
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from vectorization import generate_split_data

def supportVectorClassifier(X_train, X_test, y_train):
    # C is regularization parameter >0, gamma is kernel coeff, random state control rng, tol is tolerance
    svm = SVC(C=1.0, kernel='linear', gamma=0.1, random_state=42)
    #fit trains the model using training data, predict function uses model to generate predictions
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    return predictions
    


def mnbClassifier(X_train, X_test, y_train):
    #alpha is smoothing parameter
    mnb = MultinomialNB(alpha=0.5)
    mnb.fit(X_train, y_train)
    predictions = mnb.predict(X_test)
    return predictions


def randomForestClassifier(X_train, X_test, y_train):
    #n_estimators is number of trees, max_features is number of features, max_depth is max depth per tree
    rfc = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=20, random_state=42)
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)

def plotResults(predictions, y_test):
    #todo: represent results in graph/plots
    print(f"Accuracy: {accuracy_score(y_test, predictions)} \n")
    print(f"Classification Report: {classification_report(y_test, predictions)}")
# i have no clue what im doing bruh

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = generate_split_data(r".\emails_with_features.csv", test_pct=0.50 , head=1000)
    svc_Predictions = supportVectorClassifier(X_train, X_test, y_train)
    plotResults(svc_Predictions, y_test)