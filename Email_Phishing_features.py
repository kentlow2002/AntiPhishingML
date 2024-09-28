import pandas as pd
import re
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
df['Sender_Domain'] = df['From'].apply(lambda x: x.split('@')[-1] if isinstance(x, str) and '@' in x else '')


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
# no links stored in dataset/unable to identify embedded links
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

# 9. Extract the hour the email was sent from the 'Date' field
# df['Hour_Sent'] = pd.to_datetime(df['Date'], errors='coerce').dt.hour

# 10. Use the existing 'Unique-Mails-From-Sender' column as 'Sender_Frequency'
df['Sender_Frequency'] = df['Unique-Mails-From-Sender']

# 11. Count the number of special characters in the body
df['Special_Char_Count'] = df['Body'].apply(lambda x: sum(1 for char in x if not char.isalnum()) if pd.notna(x) else 0)

# 12. Perform sentiment analysis on the body using TextBlob
df['Sentiment'] = df['Body'].apply(lambda x: TextBlob(x).sentiment.polarity if pd.notna(x) else 0)

# Handle missing values if any
df.fillna({'Body': '', 'Subject': 'No Subject'}, inplace=True)

# Save the feature-engineered dataset to a new CSV file
df.to_csv('emails_with_features.csv', index=False)

# Display the first few rows of the dataset
print(df.head())