import pandas as pd
import re
import string
import pickle
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance
from spellchecker import SpellChecker
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from distutils.command import clean
from os.path import isfile


# Define lists for suspicious domains and TLDs
known_legit_domains = ['google.com', 'facebook.com', 'bankofamerica.com', 'microsoft.com', 'paypal.com']
suspicious_tlds = ['.xyz', '.info', '.click', '.top', '.icu', '.buzz', '.ru']
non_suspicious_tlds = ['.com', '.org', '.net', '.gov', '.edu', '.uk', '.ca', '.de']
suspicious_domain_pattern = r'\d+|[-]{2,}'

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize spell checker
spell = SpellChecker()


def feature_extraction(filename):

    # Load the email data
    df = pd.read_csv(filename)

    # Remove any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Ensure the 'Body' column is treated as a string and replace NaN with an empty string
    df['Body'] = df['Body'].fillna('').astype(str)

    # 1. Extract the sender's domain from the 'From' field
    df['Sender_Domain'] = df['From'].apply(lambda x: x.split('@')[-1] if isinstance(x, str) and '@' in x else '')



    # Add domain-related features
    df['Sender_Domain_Length'] = df['Sender_Domain'].apply(domain_length)
    df['Suspicious_Domain_Pattern'] = df['Sender_Domain'].apply(is_suspicious_domain)
    df['Suspicious_TLD_Flag'] = df['Sender_Domain'].apply(has_suspicious_tld)
    df['Typo_Domain_Flag'] = df['Sender_Domain'].apply(lambda x: is_typo_domain(x, known_legit_domains))
    df['Suspicious_Domain'] = df.apply(lambda row: row['Suspicious_Domain_Pattern'] or row['Suspicious_TLD_Flag'] or row['Typo_Domain_Flag'], axis=1)

    # 2. Measure the length of the subject and body
    df['Subject_Length'] = df['Subject'].apply(lambda x: len(x) if pd.notna(x) else 0)
    df['Body_Length'] = df['Body'].apply(lambda x: len(x) if pd.notna(x) else 0)

    # 3. Identify phishing-related keywords in the email body
    phishing_keywords = ['urgent', 'password', 'click', 'account', 'verify', 'bank', 'login']
    df['Keyword_Flag'] = df['Body'].apply(lambda x: any(keyword in x.lower() for keyword in phishing_keywords))

    # 4. Calculate the ratio of capital letters in the body
    df['Caps_Ratio'] = df['Body'].apply(lambda x: round(sum(1 for c in x if c.isupper()) / len(x), 5) if len(x) > 0 else 0)

    # 5. Count misspelled words using SpellChecker
    df['Num_Misspelled_Words'] = df['Body'].apply(lambda x: count_misspelled_words(x) if pd.notna(x) else 0)

    # 6. Count the number of special characters in the body
    df['Special_Char_Count'] = df['Body'].apply(lambda x: sum(1 for char in x if not char.isalnum()) if pd.notna(x) else 0)

    # 7. Perform sentiment analysis on the body using TextBlob
    df['Sentiment'] = df['Body'].apply(lambda x: TextBlob(x).sentiment.polarity if pd.notna(x) else 0)

    # Handle missing values if any
    df.fillna({'Body': '', 'Subject': 'No Subject'}, inplace=True)

    # Save the feature-engineered dataset to a new CSV file
    # if the file already exists (with content), dont put column header into file again
    if not isfile('emails_with_features.csv'):
        df.to_csv('emails_with_features.csv', index=False, mode='a')
    else:
        df.to_csv('emails_with_features.csv', index=False, header=False, mode='a')

    # Display the first few rows of the dataset
    print(df.head())

    # --- Spam Keywords and Trends Analysis ---
    df_spam = df[df.iloc[:, -1] == 1]  # Assuming last column is label (1 for spam)

    # Combine spam emails into one large text for analysis
    spam_texts = " ".join(df_spam['Body'].astype(str))

    cleaned_spam_texts = preprocess_text(spam_texts)

    return cleaned_spam_texts+" "

# Function to check if the domain contains suspicious patterns
def is_suspicious_domain(domain):
    return bool(re.search(suspicious_domain_pattern, domain))

# Function to check if the domain has an unusual or suspicious TLD
def has_suspicious_tld(domain):
    return any(domain.endswith(tld) for tld in suspicious_tlds)

# Function to calculate the length of the domain
def domain_length(domain):
    return len(domain)

# Function to check typo in domains using Levenshtein distance
def is_typo_domain(domain, known_legit_domains):
    domain = domain.lower().strip()
    if domain in known_legit_domains:
        return False
    return any(levenshtein_distance(domain, legit_domain) <= 1 for legit_domain in known_legit_domains)


def count_misspelled_words(text):
    # 6. Count misspelled words using SpellChecker
    text_cleaned = re.sub(r'[^\w\s]', '', text)
    words = text_cleaned.split()
    misspelled = spell.unknown(words)
    return len(misspelled)



# Preprocess the spam email texts
def preprocess_text(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)


def wordcloud_generate(cleaned_spam_texts):
    # Step 1: Generate a WordCloud for spam emails
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_spam_texts)

    # Display the WordCloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("WordCloud of Common Keywords in Spam Emails", fontsize=15)
    plt.show()

    # Step 2: Find the top 20 most common words in spam emails using CountVectorizer
    vectorizer = CountVectorizer(max_features=20, stop_words='english')
    spam_word_counts = vectorizer.fit_transform([cleaned_spam_texts]).toarray()
    spam_keywords = vectorizer.get_feature_names_out()

    # Save the CountVectorizer model using pickle
    with open('count_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Create a DataFrame for the most frequent words
    word_counts = pd.DataFrame(spam_word_counts.T, index=spam_keywords, columns=['Count'])

    # Sort the words by their frequency
    word_counts = word_counts.sort_values(by='Count', ascending=False)

    # Display the top 20 words as a bar chart
    plt.figure(figsize=(10, 6))
    word_counts['Count'].plot(kind='barh', color='orange')
    plt.title("Top 20 Most Common Keywords in Spam Emails")
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.gca().invert_yaxis()  # Invert the Y-axis to have the highest frequency at the top
    plt.show()


if __name__ == '__main__':
    datasets = ['clean_enron.csv', 'clean_spam.csv']
    texts = ""
    for i in datasets:
        texts += feature_extraction(i)

    wordcloud_generate(texts)
