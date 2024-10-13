from pickle import load
from textblob import TextBlob
from scipy.sparse import hstack


def tester(input_text, sender_domain):
    with open("model.pkl", "rb") as f:
        clf = load(f)

    with open("count_vectorizer.pkl", "rb") as f:
        count_vectorizer = load(f)

    with open("hash_vectorizer.pkl", "rb") as f:
        hash_vectorizer = load(f)
    
    # generate sentiment for email body
    sentiment = TextBlob(input_text).sentiment.polarity

    # vectorize sender domain
    hashed_domain = hash_vectorizer.transform([sender_domain])

    # perform word count vectorization on email body
    word_count = count_vectorizer.transform([input_text])

    caps_ratio = round(sum(1 for c in input_text if c.isupper()) / len(input_text), 5) if len(input_text) > 0 else 0

    # concat all of the above together
    # sentiment is in a array to make it a 2d array like the other 2 variables
    X_user = hstack([word_count, hashed_domain, [sentiment], [caps_ratio]])

    # perform prediction
    prediction = clf.predict(X_user)
    return prediction
    

if __name__ == '__main__':
    tester()