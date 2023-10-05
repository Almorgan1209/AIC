# Import necessary libraries
import pandas as pd
from collections import Counter


# Define file paths for data
#raw_data_path =
#profile_data_path = 
#class_membership_path = 

# Load dataset sheets into dataframes
data_1 = pd.read_excel(raw_data_path, sheet_name='QP - Conjoint Raw Data- 1')
data_2 = pd.read_excel(raw_data_path, sheet_name='QP - Conjoint Raw Data- 2')

# Merge data into one dataframe
data = pd.concat([data_1, data_2], ignore_index=True)

# Sort the data by Response ID and Task ID
data.sort_values(by=['Response ID', 'Task ID'], inplace=True)

# Group the data by Response ID
grouped_data = data.groupby('Response ID')

# Load profiles data
profiles = pd.read_excel(profile_data_path)

# Mapping dictionaries for attribute columns
author_mapping = {
    'Billie Thomas Ph.D - Professor': 1,
    'Taylor Marsh - Salesperson': 2
}

medium_mapping = {
    'Traditional news media': 1,
    'Academic source': 2,
    'Direct verbal or written communication': 3,
    'Online platform not associated with traditional news media sources': 4,
    'Social media platform not associated with traditional news media': 5
}

structure_mapping = {
    'Argument about The Pastry War with supporting facts and numbers': 1,
    'Personal story about their experience with The Pastry War': 2
}

# Apply mappings to convert columns to numeric values
profiles['Author - Occupation'] = profiles['Author - Occupation'].replace(author_mapping)
profiles['Medium'] = profiles['Medium'].replace(medium_mapping)
profiles['Structure'] = profiles['Structure'].replace(structure_mapping)

# Create an empty list to store results
results = []

# Iterate over each group to analyze response patterns
for group_name, group_df in grouped_data:
    # Access the Concept IDs or Selected column to examine the choices made
    concept_ids = group_df['Concept ID'].tolist()
    selected_choices = group_df[group_df['Selected'] == 1][['Author - Occupation', 'Medium', 'Structure']].values.tolist()

    # Check if the participant consistently chose the same profile throughout
    if len(set(concept_ids)) == 1:
        print(f"Participant {group_name} consistently chose Concept ID {concept_ids[0]} for all tasks.")

    # Calculate the frequency of each selected profile
    profile_frequencies = pd.DataFrame(selected_choices, columns=['Author - Occupation', 'Medium', 'Structure']).apply(tuple, axis=1).value_counts()
    profile_frequencies = profile_frequencies.reset_index()
    profile_frequencies.columns = ['Profile', 'Frequency']

    # Compare profile frequencies with profiles DataFrame
    profile_frequencies['Profile'] = profile_frequencies['Profile'].apply(lambda x: tuple(x))

    # Create a dictionary to map profile values to Sr No.
    profile_mapping = profiles.set_index(['Author - Occupation', 'Medium', 'Structure'])['Sr No.'].to_dict()

    # Add 'Sr No.' column to profile_frequencies DataFrame (Sr No. = profile number given by Questionpro)
    profile_frequencies['Sr No.'] = profile_frequencies['Profile'].map(profile_mapping)
    print(profile_frequencies)

    # Sort the profile frequencies by frequency in descending order
    profile_frequencies.sort_values(by='Frequency', ascending=False, inplace=True)

    # Print the frequency, Sr No., and group_name for profiles with frequency > 3
    for index, row in profile_frequencies.iterrows():
        profile = row['Profile']
        frequency = row['Frequency']
        sr_no = row['Sr No.']

        if frequency > 3:
            print(f"Frequency: {frequency}, Sr No.: {sr_no}, Group Name: {group_name}")
            results.append((group_name, sr_no, frequency))

# Count the frequencies
frequency_counts = Counter([result[2] for result in results])

# Sort the results by frequencies in descending order
results.sort(key=lambda x: x[2], reverse=True)

# Print the results
for result in results:
    group_name, sr_no, frequency = result
    print(f"Group Name: {group_name}, Sr No.: {sr_no}, Frequency: {frequency}")

# Print the frequency counts
print("\nFrequency Counts:")
for frequency, count in frequency_counts.items():
    print(f"Frequency: {frequency}, Count: {count}")
