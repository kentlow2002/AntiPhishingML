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
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_features='sqrt', 
        max_depth=40, 
        random_state=42, 
        class_weight='balanced'
        )
    clf.fit(X_train, y_train)

    #dump model into pkl file for reuse
    with open("model.pkl", "wb") as f:
        dump(clf, f, protocol=5)

    #generate prediction on test data
    predictions = clf.predict(X_test)
    return predictions

#Random Forest Classifier with Hyperparameter Searching to tune for precision, recall etc
def rfcSearch(X_train, X_test, y_train):
    #set parameters to search best value for
    parameters = {
        'max_depth': list(range(10, 100, 10))
        }
    
    #set scoring metrics
    scoring = {'precision':'precision', 'recall':'recall'}

    #Set search for random forest with set parameters
    clf = GridSearchCV(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_features='sqrt',
            class_weight='balanced'
        ), 
        param_grid=parameters, 
        n_jobs=-1, 
        scoring=scoring, 
        refit='precision',
        return_train_score=True,
    )

    #fit on training data
    clf.fit(X_train, y_train)
    #results = clf.cv_results_
    print(clf.best_params_)

    #plot metric graph
    plotMetrics(clf.cv_results_, scoring)

    #make sure file name has no forbidden characters (for windows filenames)
    #pickle_name = 'randomForestClassifier'+str(clf.best_params_).replace(': ', '').replace('\'', '').replace(' ','')
    pickle_name = 'model.pkl'

    #export model to file
    with open(pickle_name, "wb") as f:
        dump(clf, f, protocol=5)

    #predict on test data
    y_pred = clf.predict(X_test)

    return y_pred


def plotMetrics(results, scoring):
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating Random Forest for Precision", fontsize=16)

    plt.xlabel("max_depth")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(0, 100)
    ax.set_ylim(0.00, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results["param_max_depth"].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ["g", "k"]):
        for sample, style in (("train", "--"), ("test", "-")):
            sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
            sample_score_std = results["std_%s_%s" % (sample, scorer)]
            ax.fill_between(
                X_axis,
                sample_score_mean - sample_score_std,
                sample_score_mean + sample_score_std,
                alpha=0.1 if sample == "test" else 0,
                color=color,
            )
            ax.plot(
                X_axis,
                sample_score_mean,
                style,
                color=color,
                alpha=1 if sample == "test" else 0.7,
                label="%s (%s)" % (scorer, sample),
            )

        best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
        best_score = results["mean_test_%s" % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot(
            [
                X_axis[best_index],
            ]
            * 2,
            [0, best_score],
            linestyle="-.",
            color=color,
            marker="x",
            markeredgewidth=3,
            ms=8,
        )

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()
# i have no clue what im doing bruh

if __name__ == '__main__':
    #get training and testing data
    X_train, X_test, y_train, y_test = generate_split_data(r".\emails_with_features.csv", test_pct=0.40)

    #run classifier to get predictions
    predictions = rfcSearch(X_train, X_test, y_train.squeeze())

    #print metrics report
    print(classification_report(y_test, predictions))
    '''precision, recall, _ = precision_recall_curve(y_test, predictions)
    print(precision, recall)
    print(classification_report(y_test, predictions))'''