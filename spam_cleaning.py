import os
import re
import pandas as pd
import numpy as np
from faker import Faker  # To generate random email addresses and dates

# Initialize Faker instance for generating random data
fake = Faker()

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
            # Extract Date from the content
            if line.lower().startswith("date:"):
                date_match = patterns['Date'].match(line)
                if date_match:
                    email_data['Date'] = date_match.group(1)

            # Extract other headers
            header_found = False
            for field, pattern in patterns.items():
                if field == 'Date':
                    continue  # Skip Date since it's already handled
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

# Fill missing values with random data
def fill_missing_with_random(df):
    for col in df.columns:
        if col == 'Message-ID':
            # Replace missing Message-ID with random message IDs
            df[col] = df[col].apply(lambda x: f"<{fake.uuid4()}@{fake.domain_name()}>" if pd.isna(x) else x)
        elif col == 'Date':
            # Replace missing dates with random dates
            df[col] = df[col].apply(lambda x: fake.date_time_this_decade() if pd.isna(x) else x)
        elif col in ['From', 'To', 'Mail-ID']:
            # Replace missing email addresses with random emails
            df[col] = df[col].apply(lambda x: fake.email() if pd.isna(x) else x)
        elif col == 'Sender-Type':
            # Replace missing Sender-Type with random choice between 'External' and 'Internal'
            df[col] = df[col].apply(lambda x: fake.random_element(elements=('External', 'Internal')) if pd.isna(x) else x)
    return df

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

# Fill missing categorical values with random data, including Message-ID
df_filled = fill_missing_with_random(df)

# Filter rows based on whether Message-ID contains an '@' symbol
df_filtered_message_id = df_filled[df_filled['Message-ID'].str.contains("@", na=False)]

# Write the filtered data (based on Message-ID) to CSV
df_filtered_message_id.to_csv('clean_spam.csv', index=False)
