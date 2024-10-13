import pandas as pd

# Configuration
input_file = './enron_data_fraud_labeled.csv'
output_file = './clean_enron.csv'
columns_to_remove = [
    'Folder-User', 'Folder-Name', 'Mime-Version', 'Content-Type',
    'Content-Transfer-Encoding', 'X-From', 'X-To', 'X-cc', 'X-bcc',
    'X-Folder', 'X-Origin', 'X-FileName', 'Cc',
    'Bcc', 'Time', 'Attendees', 'Re', 'Source', 'POI-Present',
    'Suspicious-Folders', 'Low-Comm', 'Contains-Reply-Forwards'
]
# Step 1: Load the original CSV file
df = pd.read_csv(input_file)

# # Step 2: Remove unwanted columns
if columns_to_remove:
    df = df.drop(columns=columns_to_remove)
    #df.to_csv(output_file, index=False)
    #print(f"Unwanted columns removed and initial data saved to '{output_file}'")
else:
    print("No columns to remove.")

if 'Label' in df.columns:
    #filter labels that are not 0/1. rows filtered away are considered corrupted
    df = df.loc[df['Label'].isin([0, 1])]
    print(df['Label'].unique())
    #df.to_csv(output_file, index=False)
else:
    print("No rows to remove.")

#Step 3: Drop rows where 'Subject' is empty
if 'Subject' in df.columns:
    df_cleaned = df[df['Subject'].notna() & df['Subject'].str.strip().astype(bool)]
    #df_cleaned.to_csv(output_file, index=False)
    print("Rows with empty 'Subject' removed.")
else:
    print("'Subject' column not found.")

# # Step 4: Remove duplicate 'Body' rows
if 'Body' in df.columns:
    df_cleaned = df.drop_duplicates(subset=['Body'], keep='first')
    #df_cleaned.to_csv(output_file, index=False)
    print("Duplicate 'Body' rows removed.")
else:
    print("'Body' column not found.")

# Step 5: Remove duplicates based on 'From', 'To', and 'Subject' and 'Body'
if all(col in df_cleaned.columns for col in ['From', 'To', 'Subject', 'Body']):
    df_cleaned = df_cleaned.drop_duplicates(subset=['From', 'To', 'Subject', 'Body'], keep='first')
    #df_cleaned.to_csv(output_file, index=False)
    print("Duplicates based on 'From', 'To', 'Subject', 'Body' removed.")
else:
    print("One or more columns for duplication check not found.")

# Step 6: Remove duplicates based on 'Message-ID' and 'Mail-ID'
if all(col in df_cleaned.columns for col in ['Message-ID', 'Mail-ID']):
    df_cleaned = df_cleaned.drop_duplicates(subset=['Message-ID', 'Mail-ID'], keep='first')
    #df_cleaned.to_csv(output_file, index=False)
    print("Duplicates based on 'Message-ID' and 'Mail-ID' removed.")
else:
    print("One or both columns for duplication check not found.")

# Step 7: Filter by Message-ID pattern
if 'Message-ID' in df_cleaned.columns:
    pattern = r'<.*JavaMail.*>'
    df_filtered = df_cleaned[df_cleaned['Message-ID'].str.contains(pattern, regex=True, na=False)]
    #df_filtered.to_csv(output_file, index=False)
    print("Rows filtered by 'Message-ID' pattern.")
else:
    print("'Message-ID' column not found.")

# Step 8: Keep only valid email addresses in 'From' and 'To'
if all(col in df_filtered.columns for col in ['From', 'To']):
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    df_filtered = df_filtered[
        df_filtered['From'].str.contains(email_pattern, regex=True, na=False) &
        df_filtered['To'].str.contains(email_pattern, regex=True, na=False)
    ]
    df_filtered.to_csv(output_file, index=False)
    print("Filtered rows with valid email addresses in 'From' and 'To'.")
else:
    print("One or both columns for email validation not found.")

print(f"Final filtered data saved to '{output_file}'")
