import os
import re
import pandas as pd
from sklearn.impute import KNNImputer

# Define the folder paths
archive_folder = r"C:\PhishingEmail\spam"
subfolders = ["easy_ham", "hard_ham", "spam_2"]

# Regex patterns for extracting fields
patterns = {
    'Message-ID': re.compile(r"^Message-Id: <(.+)>", re.IGNORECASE),
    'Date': re.compile(r"^Date: (.+)", re.IGNORECASE),
    'From': re.compile(r"^From: (.+)", re.IGNORECASE),
    'To': re.compile(r"^To: (.+)", re.IGNORECASE),
    'Subject': re.compile(r"^Subject: (.+)", re.IGNORECASE),
}

# Dictionary to track how many times each sender email has been seen
from_count = {}

# Function to extract fields and body from an email file
def extract_email_fields(file_content):
    email_data = {
        'Message-ID': None,
        'Date': None,
        'From': None,
        'To': None,
        'Subject': None,
        'Body': None,
        'Mail-ID': None,
        'Sender-Type': None,
        'Unique-Mails-From-Sender': None,
        'Label': None
    }

    in_body = False
    body_content = []

    # Process each line to extract header fields and body
    for line in file_content.splitlines():
        if not in_body:
            # Try to match headers
            header_found = False
            for field, pattern in patterns.items():
                match = pattern.match(line)
                if match:
                    email_data[field] = match.group(1)
                    header_found = True
                    break

            # If we reach a line that isn't any known header, it's the start of the body
            if not header_found and 'X-Spam-Level' in line:
                in_body = True
        else:
            # Collect body content
            body_content.append(line)

    email_data['Body'] = "\n".join(body_content).strip() if body_content else None
    return email_data

# Function to label based on folder
def get_label(folder_name):
    if "spam" in folder_name:
        return 1  # Spam
    else:
        return 0  # Ham

# List to hold all extracted data
all_emails = []

# Traverse the folders and process each file
for folder in subfolders:
    folder_path = os.path.join(archive_folder, folder)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the current path is a file (skip directories)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                file_content = file.read()
                email_data = extract_email_fields(file_content)

                # Track the sender (From field) and count occurrences
                sender = email_data['From']
                if sender:
                    if sender not in from_count:
                        from_count[sender] = 0  # Initialize if sender not in the dictionary
                    from_count[sender] += 1  # Increment the count
                    email_data['Unique-Mails-From-Sender'] = from_count[sender]  # Assign the count to the email
                
                email_data['Mail-ID'] = filename  # Use filename as Mail-ID
                email_data['Sender-Type'] = "External"  # Example, adjust as needed
                email_data['Label'] = get_label(folder)  # Label based on folder
                
                all_emails.append(email_data)

# Convert to DataFrame
df = pd.DataFrame(all_emails)

# Remove rows where only "Message-ID" is present and all other fields are missing
df_cleaned = df.dropna(subset=[col for col in df.columns if col != 'Message-ID'], how='all')

# Set missing values in the "Body" column to NaN (if not already NaN)
df_cleaned['Body'] = df_cleaned['Body'].fillna(pd.NA)

# Fill missing values in text columns using mode (most common value) - Exclude 'Message-ID' and 'Subject'
impute_columns = ['Date', 'From', 'To', 'Mail-ID', 'Sender-Type']
for column in impute_columns:
    df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mode()[0])  # Fill missing with mode

# "Message-ID" and "Subject" columns are left unchanged, no imputation or KNN is applied to them

# Select only the numerical columns for KNN imputation
numerical_columns = ['Unique-Mails-From-Sender']
df_numerical = df_cleaned[numerical_columns]

# Apply KNN imputation on numerical columns only
imputer = KNNImputer(n_neighbors=3)
df_numerical_imputed = pd.DataFrame(imputer.fit_transform(df_numerical), columns=df_numerical.columns)

# Replace the imputed numerical columns in the original DataFrame
df_cleaned[numerical_columns] = df_numerical_imputed

# Shift all data up to remove gaps using the stack/unstack method
df_shifted = df_cleaned.stack().unstack().reset_index(drop=True)

# Remove any rows that still have missing data after shifting
df_shifted.dropna(inplace=True)

# Write the final cleaned DataFrame to a CSV file
df_shifted.to_csv('clean_spam.csv', index=False)
