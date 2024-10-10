import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import hstack, coo_matrix
from pickle import load, dump
from os.path import isfile

# Function to load data from CSV, perform TF-IDF, and split training and testing data
def generate_split_data(csv_file, test_pct=0.40, head=0):

    #read data from csv. if head is > 0, read the first (head) addresses, else read the whole file
    if head == 0:
        dataset = pd.read_csv(csv_file, low_memory=False)
    else:
        dataset = pd.read_csv(csv_file).head(head)

    #load word count vectorizer from pkl file
    with open('count_vectorizer.pkl', 'rb') as f:
        count_vectorizer = load(f)

    

    #hash_vectorizer = HashingVectorizer(n_features=2**3)

    #split dataset into test and training
    X_train, X_test, y_train, y_test = train_test_split(dataset[['Body', 'Sentiment', 'Sender_Domain']], dataset[['Label']], test_size=test_pct, shuffle=True)
    
    if isfile('hash_vectorizer.pkl'):
        with open('hash_vectorizer.pkl', 'rb') as f:
            hash_vectorizer = load(f)
    else:
        hash_vectorizer = HashingVectorizer(n_features=2**4)

    #transform text features into word counts/hashes with vectorizer
    #lambda used here ensures all values become string, including empty values e.g. np.NaN, which will throw error in vectorizer
    X_train_vect = count_vectorizer.transform(X_train.Body.astype('U').values)
    X_train_sender_vect = hash_vectorizer.fit_transform(X_train.Sender_Domain.apply(lambda x: np.str_(x)))

    if not isfile('hash_vectorizer.pkl'):
        with open('hash_vectorizer.pkl', 'wb') as f:
            dump(hash_vectorizer, f)

    X_test_vect = count_vectorizer.transform(X_test.Body.astype('U').values)
    X_test_sender_vect = hash_vectorizer.transform(X_test.Sender_Domain.apply(lambda x: np.str_(x)))

    print(X_train_vect.shape, coo_matrix(X_train.Sentiment.astype(float)).T.shape, X_train_sender_vect.shape)

    #concat sentiment back to data
    transpose_train_sentiment = coo_matrix(X_train.Sentiment.astype(float)).T
    new_X_train = hstack([X_train_vect, X_train_sender_vect, transpose_train_sentiment])
    transpose_test_sentiment = coo_matrix(X_test.Sentiment.astype(float)).T
    new_X_test = hstack([X_test_vect, X_test_sender_vect, transpose_test_sentiment])

    return new_X_train, new_X_test, y_train, y_test



if __name__ == '__main__':
    generate_split_data('emails_with_features.csv')