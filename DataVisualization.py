# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset (you can replace this with the actual path of your dataset)
# Assuming the dataset is in CSV format and has features like 'Sender_Info', 'Content', 'Links', etc.
df = pd.read_csv("emails_with_features.csv")

# Check the first few rows of the dataset
print(df.head())

# Basic information about the dataset
print(df.info())

# Summary statistics for numerical columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Check the distribution of the target variable (e.g., 'Label' or 'Class')
# Assuming 1 is for phishing and 0 is for legitimate
plt.figure(figsize=(6, 4))
sns.countplot(x='Label', data=df, palette='Set2')
plt.title('Distribution of Phishing vs Legitimate Emails')
plt.xlabel('Email Type (0 = Legitimate, 1 = Phishing)')
plt.ylabel('Count')
plt.show()

# Explore categorical features (e.g., Sender_Info, Domain, etc.)
plt.figure(figsize=(10, 6))
sns.countplot(y='Sender_Info', data=df, order=df['Sender_Info'].value_counts().index, palette='coolwarm')
plt.title('Sender Information Frequency')
plt.show()

# Explore email content length distribution (assuming 'Content_Length' is a feature)
plt.figure(figsize=(10, 6))
sns.histplot(df['Content_Length'], bins=30, kde=True)
plt.title('Distribution of Email Content Length')
plt.xlabel('Content Length')
plt.ylabel('Frequency')
plt.show()

# Visualize correlation matrix to detect relationships between features
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

# Visualize relationship between email links and the target label (assuming 'Links' is a feature)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Label', y='Links', data=df, palette='Set1')
plt.title('Email Links vs Label (Phishing or Legitimate)')
plt.xlabel('Email Type (0 = Legitimate, 1 = Phishing)')
plt.ylabel('Number of Links')
plt.show()

# Visualize the relationship between sender info and phishing probability
plt.figure(figsize=(12, 8))
sns.barplot(x='Sender_Info', y='Label', data=df, estimator=np.mean, palette='viridis')
plt.title('Sender Info vs Probability of Phishing Email')
plt.xlabel('Sender Info')
plt.ylabel('Phishing Probability')
plt.xticks(rotation=45)
plt.show()

# Explore the top words in email content based on phishing and legitimate emails
from wordcloud import WordCloud

# Separate phishing and legitimate emails
phishing_emails = df[df['Label'] == 1]['Content'].values
legit_emails = df[df['Label'] == 0]['Content'].values

# Create WordCloud for phishing emails
plt.figure(figsize=(10, 6))
wordcloud_phishing = WordCloud(width=800, height=400, background_color='white').generate(' '.join(phishing_emails))
plt.imshow(wordcloud_phishing, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of Phishing Emails')
plt.show()

# Create WordCloud for legitimate emails
plt.figure(figsize=(10, 6))
wordcloud_legit = WordCloud(width=800, height=400, background_color='white').generate(' '.join(legit_emails))
plt.imshow(wordcloud_legit, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of Legitimate Emails')
plt.show()

# Feature Engineering: Create new features based on existing data (e.g., count links, special characters)
df['Link_Count'] = df['Content'].apply(lambda x: x.count('http'))
df['Special_Char_Count'] = df['Content'].apply(lambda x: sum([1 for char in x if char in "!@#$%^&*()"]))

# Visualize new features
plt.figure(figsize=(10, 6))
sns.histplot(df['Link_Count'], bins=30, kde=True, color='blue')
plt.title('Distribution of Link Count in Emails')
plt.xlabel('Link Count')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Special_Char_Count'], bins=30, kde=True, color='red')
plt.title('Distribution of Special Character Count in Emails')
plt.xlabel('Special Character Count')
plt.ylabel('Frequency')
plt.show()

# Data preprocessing (Encoding categorical variables if necessary)
# Assuming 'Sender_Info' needs to be encoded
le = LabelEncoder()
df['Sender_Info'] = le.fit_transform(df['Sender_Info'])

# Split the data into train and test sets (Assuming 'Label' is the target variable)
X = df.drop('Label', axis=1)  # Features
y = df['Label']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")


#Detect keywords from phising emails and create a keyword list to pass forward