import pandas as pd
import re
from spellchecker import SpellChecker
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the email data
df = pd.read_csv('enron_data_fraud_labeled.csv', low_memory=False)

# Remove any unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Ensure the 'Body' column is treated as a string and replace NaN with an empty string
df['Body'] = df['Body'].fillna('').astype(str)

# 1. Extract the sender's domain from the 'From' field
df['Sender_Domain'] = df['From'].apply(lambda x: x.split('@')[-1] if isinstance(x, str) and '@' in x else '')

# 2. Count the number of recipients in the 'To' field
df['Num_Recipients'] = df['To'].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0)

# 3. Measure the length of the subject and body
df['Subject_Length'] = df['Subject'].apply(lambda x: len(x) if pd.notna(x) else 0)
df['Body_Length'] = df['Body'].apply(lambda x: len(x) if pd.notna(x) else 0)

# 4. Identify phishing-related keywords in the email body
phishing_keywords = ['urgent', 'password', 'click', 'account', 'verify', 'bank', 'login']
df['Keyword_Flag'] = df['Body'].apply(lambda x: any(keyword in x.lower() for keyword in phishing_keywords))

# 5. Calculate the ratio of capital letters in the body
df['Caps_Ratio'] = df['Body'].apply(lambda x: round(sum(1 for c in x if c.isupper()) / len(x), 5) if pd.notna(x) and len(x) > 0 else 0)

# 6. Count misspelled words using SpellChecker
spell = SpellChecker()
def count_misspelled_words(text):
    text_cleaned = re.sub(r'[^\w\s]', '', text)
    words = text_cleaned.split()
    misspelled = spell.unknown(words)
    return len(misspelled)

df['Num_Misspelled_Words'] = df['Body'].apply(lambda x: count_misspelled_words(x) if pd.notna(x) else 0)

# 7. Detect if the body mentions an attachment
df['Has_Attachment'] = df['Body'].apply(lambda x: 'attachment' in x.lower() if pd.notna(x) else False)

# 8. Perform sentiment analysis on the body using TextBlob
df['Sentiment'] = df['Body'].apply(lambda x: TextBlob(x).sentiment.polarity if pd.notna(x) else 0)

# Handle missing values
df.fillna({'Body': '', 'Subject': 'No Subject'}, inplace=True)

# Save the feature-engineered dataset to a new CSV file
df.to_csv('emails_with_features.csv', index=False)

# Display the first few rows of the dataset
print(df.head())

# --- Spam Keywords and Trends Analysis ---

# Combine all the email text in the spam dataset (assuming label column is the 10th one, as per your earlier description)
df_spam = df[df.iloc[:, -1] == 1]  # Assuming last column is label (1 for spam)

# Combine spam emails into one large text for analysis
spam_texts = " ".join(df_spam['Body'].astype(str))

# Preprocess the spam email texts
def preprocess_text(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

cleaned_spam_texts = preprocess_text(spam_texts)

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
