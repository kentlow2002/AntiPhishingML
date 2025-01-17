import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import hstack, coo_matrix
from pickle import load, dump
from os.path import isfile

# Function to load data from CSV, perform TF-IDF, and split training and testing data
def generate_split_data(csv_file, test_pct=0.40, head=0):
    """
    Reads a CSV file containing the dataset, splits the dataset into training and testing, 
    performs any vectorization required for input

    Parameters
    ----------
    csv_file : str
        CSV file to read
    test_pct : float
        Percentage of the dataset to put into testing
    head : int
        Read first N rows of the dataset.
        Used for testing the function only.
        Set to 0 to read whole dataset.

    Returns
    -------
    new_X_train : sparse matrix
        Inputs for training set
    new_X_test : sparse matrix
        Inputs for testing set
    y_train : sparse matrix
        Expected output for training set
    y_test : sparse matrix
        Expected output for testing set
    """

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
    X_train, X_test, y_train, y_test = train_test_split(dataset[['Body', 'Sentiment', 'Sender_Domain', 'Caps_Ratio']], dataset[['Label']], test_size=test_pct, shuffle=True)
    
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
    #print(X_train_vect.shape, coo_matrix(X_train.Sentiment.astype(float)).T.shape, X_train_sender_vect.shape)

    #concat sentiment back to data
    #for sentiment and caps ratio, they are column vectors with shape (1,)
    #creating a new matrix and transposing is a trick to turn a column vector
    #into a 2d array
    new_X_train = hstack(
        [
            X_train_vect, 
            X_train_sender_vect, 
            coo_matrix(X_train.Sentiment.astype(float)).T, 
            coo_matrix(X_train.Caps_Ratio.astype(float)).T
        ]
    )
    new_X_test = hstack(
        [
            X_test_vect, 
            X_test_sender_vect, 
            coo_matrix(X_test.Sentiment.astype(float)).T, 
            coo_matrix(X_test.Caps_Ratio.astype(float)).T
        ]
    )

    return new_X_train, new_X_test, y_train, y_test



if __name__ == '__main__':
    generate_split_data('emails_with_features.csv')