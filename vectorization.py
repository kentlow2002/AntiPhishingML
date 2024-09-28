import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from numpy import hstack, transpose

# Function to load data from CSV, perform TF-IDF, and split training and testing data
def generate_split_data(csv_file, test_pct=0.40, head=0):

    #read data from csv. if head is > 0, read the first (head) addresses, else read the whole file
    if head == 0:
        dataset = pd.read_csv(csv_file)
    else:
        dataset = pd.read_csv(csv_file).head(head)

    #initialize TF-IDF vectorizer
    tfidf_vectorize = TfidfVectorizer(use_idf=True)

    #split dataset into test and training
    X_train, X_test, y_train, y_test = train_test_split(dataset[['Body', 'Sentiment']], dataset[['Label']], test_size=test_pct, random_state=42)
    
    #fit and transform TF_IDF vectorizer using training data first
    X_train_tdidf = tfidf_vectorize.fit_transform(X_train.Body).toarray()

    #transform testing data using fitted TD-IDF vectorizer
    X_test_tdidf = tfidf_vectorize.transform(X_test.Body).toarray()

    #concat sentiment to training TD-IDF
    new_X_train = hstack([X_train_tdidf, transpose([X_train.Sentiment])])
    print(X_train_tdidf.shape, new_X_train.shape)
    
    new_X_test = hstack([X_test_tdidf, transpose([X_test.Sentiment])])
    print(X_test_tdidf.shape, new_X_test.shape)

    return new_X_train, new_X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = generate_split_data(r".\emails_with_features.csv", head=200)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)