from cgi import test
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def supportVectorClassifier(trainingSet, trainingLabels, testSet):
    # C is regularization parameter >0, gamma is kernel coeff, random state control rng, tol is tolerance
    svm = SVC(C=1.0, kernel='linear', gamma=0.1, random_state=42)
    #fit trains the model using training data, predict function uses model to generate predictions
    svm.fit(trainingSet, trainingLabels)
    predictions = svm.predict(testSet)
    print(f"Accuracy: {accuracy_score(testSet, predictions)} \n")
    print(f"Classification Report: {classification_report(testSet, predictions)}")
    


def mnbClassifier(trainingSet, trainingLabels, testSet):
    #alpha is smoothing parameter
    mnb = MultinomialNB(alpha=0.5)
    mnb.fit(trainingSet, trainingLabels)
    print(mnb.predict(testSet))


def randomForestClassifier(trainingSet, trainingLabels, testSet):
    #n_estimators is number of trees, max_features is number of features, max_depth is max depth per tree
    clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=20, random_state=42)
    clf = clf.fit(trainingSet, trainingLabels)


# i have no clue what im doing bruh