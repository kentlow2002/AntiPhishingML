import pandas as pd
import re
from Levenshtein import distance as levenshtein_distance
from spellchecker import SpellChecker
from textblob import TextBlob

# Load email data
df = pd.read_csv('enron_data_fraud_labeled.csv', low_memory=False)

# Remove any unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Ensure the 'Body' column is treated as a string and replace NaN with an empty string
df['Body'] = df['Body'].fillna('').astype(str)

# 1. Extract the sender's domain from the 'From' field
# df['Sender_Domain'] = df['From'].apply(lambda x: x.split('@')[-1] if '@' in x else '')
# List of known trusted domains for typo check
known_legit_domains = ['google.com', 'facebook.com', 'bankofamerica.com', 'microsoft.com', 'paypal.com']

#  List of non-suspicious (whitelist) and suspicious  TLDs (Top-Level Domains)
suspicious_tlds = ['.xyz', '.info', '.click', '.top', '.icu', '.buzz','.ru']
non_suspicious_tlds = ['.com', '.org', '.net', '.gov', '.edu', '.uk', '.ca', '.de']

# Regex to detect multiple hyphens or numbers in domain names
suspicious_domain_pattern = r'\d+|[-]{2,}'

# Function to check if the domain contains suspicious patterns
def is_suspicious_domain(domain):
    return bool(re.search(suspicious_domain_pattern, domain))

# Function to check if the domain has an unusual or suspicious TLD
def has_suspicious_tld(domain):
    for tld in suspicious_tlds:
        if domain.endswith(tld):
            return True
    return False

# Function to calculate the length of the domain
def domain_length(domain):
    return len(domain)

# Function to check typo in domains using Levenshtein distance
def is_typo_domain(domain, known_legit_domains):
    domain = domain.lower().strip()

    if domain in known_legit_domains:
        return False

    for legit_domain in known_legit_domains:
        if levenshtein_distance(domain, legit_domain) <= 1:  # Small edit distance suggests typo
            return True
    return False

df['Sender_Domain'] = df['From'].apply(lambda x: x.split('@')[-1] if isinstance(x, str) and '@' in x else '')
df['Sender_Domain_Length'] = df['Sender_Domain'].apply(domain_length)
df['Suspicious_Domain_Pattern'] = df['Sender_Domain'].apply(is_suspicious_domain)
df['Suspicious_TLD_Flag'] = df['Sender_Domain'].apply(has_suspicious_tld)
df['Typo_Domain_Flag'] = df['Sender_Domain'].apply(lambda x: is_typo_domain(x, known_legit_domains))
df['Suspicious_Domain'] = df.apply(lambda row: row['Suspicious_Domain_Pattern'] or row['Suspicious_TLD_Flag'] or row['Typo_Domain_Flag'], axis=1)


# 2. Count the number of recipients in the 'To' field
df['Num_Recipients'] = df['To'].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0)

# 3. Measure the length of the subject and body
df['Subject_Length'] = df['Subject'].apply(lambda x: len(x) if pd.notna(x) else 0)
df['Body_Length'] = df['Body'].apply(lambda x: len(x) if pd.notna(x) else 0)

# 4. Identify phishing-related keywords in the email body
phishing_keywords = ['urgent', 'password', 'click', 'account', 'verify', 'bank','login']
df['Keyword_Flag'] = df['Body'].apply(lambda x: any(keyword in x.lower() for keyword in phishing_keywords))

# 5. Calculate the ratio of capital letters in the body
df['Caps_Ratio'] = df['Body'].apply(lambda x: round(sum(1 for c in x if c.isupper()) / len(x),5) if pd.notna(x) and len(x) > 0 else 0)

# 6. Count the number of links in the email body using regex
##### no links stored in dataset/unable to identify embedded links
# def find_links(text):
#     return re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
# df['Num_Links'] = df['Body'].apply(lambda x: len(find_links(x)) if pd.notna(x) else 0)

# 7. Count misspelled words using SpellChecker
spell = SpellChecker()

def count_misspelled_words(text):
    # Remove punctuation from the text using regex
    text_cleaned = re.sub(r'[^\w\s]', '', text)
    words = text_cleaned.split()
    misspelled = spell.unknown(words)
    return len(misspelled)

df['Num_Misspelled_Words'] = df['Body'].apply(lambda x: count_misspelled_words(x) if pd.notna(x) else 0)

# 8. Detect if the body mentions an attachment
df['Has_Attachment'] = df['Body'].apply(lambda x: 'attachment' in x.lower() if pd.notna(x) else False)

# 9. Count the number of special characters in the body
df['Special_Char_Count'] = df['Body'].apply(lambda x: sum(1 for char in x if not char.isalnum()) if pd.notna(x) else 0)

# 10. Perform sentiment analysis on the body using TextBlob
df['Sentiment'] = df['Body'].apply(lambda x: TextBlob(x).sentiment.polarity if pd.notna(x) else 0)

# Handle missing values if any
df.fillna({'Body': '', 'Subject': 'No Subject'}, inplace=True)

# Save the feature-engineered dataset to a new CSV file
df.to_csv('emails_with_features.csv', index=False)

# Display the first few rows of the dataset
print(df.head())