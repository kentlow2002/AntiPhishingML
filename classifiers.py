from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, classification_report
from vectorization import generate_split_data
from pickle import dump
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Support Vector Classifier
def supportVectorClassifier(X_train, X_test, y_train):
    # C is regularization parameter >0, gamma is kernel coeff, random state control rng, tol is tolerance
    svm = SVC(C=1.0, kernel='linear', gamma=0.1, random_state=42, class_weight='balanced')
    #svm = LinearSVC(C=1.0, random_state=42)
    #fit trains the model using training data, predict function uses model to generate predictions
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    return predictions

#Multiple Support Vector Classifiers in 1 'bag'
def BaggingSVC(X_train, X_test, y_train):
    # C is regularization parameter >0, gamma is kernel coeff, random state control rng, tol is tolerance
    svm = SVC(C=1.0, kernel='linear', gamma=0.1, random_state=42, class_weight='balanced')
    #Put base estimator into Bagging Classifier
    clf = BaggingClassifier(estimator=svm, n_estimators=5, n_jobs=-1, random_state=42)
    #fit to training data
    clf.fit(X_train, y_train)
    #dump model for reuse
    with open("model.pkl", "wb") as f:
        dump(clf, f, protocol=5)
    #predict on test data
    predictions = clf.predict(X_test)
    return predictions
  
#Random Forest Classifier
def randomForestClassifier(X_train, X_test, y_train):
    #n_estimators is number of trees, max_features is number of features, max_depth is max depth per tree
    clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=20, random_state=42, class_weight={0: 1, 1: 3})
    clf.fit(X_train, y_train)

    #dump model into pkl file for reuse
    with open("model.pkl", "wb") as f:
        dump(clf, f, protocol=5)

    #generate prediction on test data
    predictions = clf.predict(X_test)
    return predictions

#Random Forest Classifier with Hyperparameter Searching to tune for precision, recall etc
def rfcSearch(X_train, X_test, y_train):
    #n_estimators is number of trees, max_features is number of features, max_depth is max depth per tree
    parameters = {
        'n_estimators': [20, 50, 100],
        'max_depth': [20, 30],
        'class_weight': [{0: 1, 1: 3}, {0: 1, 1: 2}, 'balanced'],
        }
    scoring = {'precision':'precision'}
    clf = GridSearchCV(
        RandomForestClassifier(
            random_state=42,
            max_features='sqrt',
            class_weight={0: 1, 1: 3},
                    
        ), 
        param_grid=parameters, 
        n_jobs=-1, 
        scoring=scoring, 
        refit='precision'
    )

    clf.fit(X_train, y_train)

    print(clf.best_params_)
    pickle_name = 'randomForestClassifier'+str(clf.best_params_).replace(': ', '').replace('\'', '').replace(' ','')
    with open(pickle_name, "wb") as f:
        dump(clf, f, protocol=5)
    y_pred = clf.predict(X_test)
    return y_pred

# i have no clue what im doing bruh

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = generate_split_data(r".\emails_with_features.csv", test_pct=0.40)
    predictions = rfcSearch(X_train, X_test, y_train.squeeze())
    print(classification_report(y_test, predictions))
    '''precision, recall, _ = precision_recall_curve(y_test, predictions)
    print(precision, recall)
    print(classification_report(y_test, predictions))'''