from sklearn.metrics import precision_recall_curve, classification_report
from vectorization import generate_split_data
from pickle import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


with open('randomForestClassifier{max_depth10,n_estimators1000}', 'rb') as f:
    model1 = load(f)

with open('randomForestClassifier{class_weightbalanced,n_estimators1000}', 'rb') as f:
    model2 = load(f)

with open('randomForestClassifier{class_weightbalanced,max_depth20,n_estimators100}recall', 'rb') as f:
    model3 = load(f)

#get training and testing data
X_train, X_test, y_train, y_test = generate_split_data(r".\emails_with_features.csv", test_pct=0.20)

#run classifier to get predictions
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)

plt.figure(figsize=(13, 13))
plt.title("GridSearchCV recall evaluation", fontsize=16)

plt.xlabel("n_estimators")
plt.ylabel("Recall")

ax = plt.gca()
ax.set_xlim(50, 1050)
ax.set_ylim(0.00, 1)

#print metrics report
print(classification_report(y_test, pred1))
print(classification_report(y_test, pred2))