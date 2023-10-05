# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:13:41 2023

@author: almor
"""

import pandas as pd
import argparse


def load_data(file_name, sheet_names):
    """
   Load data from an excel file and concatenate sheets.

   Parameters
   ----------
   file_name : str
       The name of the excel file containing discrete choice data.
   sheet_names : list of str
       A list of sheet names to load from the file.

   Returns
   -------
   pd.DataFrame
       Dataframe that concatenates data from specified sheets.

   """
    dfs = []
    for sheet_name in sheet_names:
        df = pd.read_excel(file_name,sheet_name=sheet_name, header=0)
        dfs.append(df)
    return pd.concat(dfs)

def remove_unwanted_columns(df):
    """
    Remove unwanted columns from the dataframes.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to remove unwanted columns from.

    Returns
    -------
    pd.Dataframe
        Dataframe with unwanted columns removed.

    """
    return df.drop(columns=['Parts Worth', 'Standard Deviation',
                            'Confidence Interval Range 1', 'Confidence Interval Range 2'])

def rename_columns(df):
    """
    Renames dataframe columns.

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe to rename columns in..

    Returns
    -------
    df : pd.Dataframe
       Dataframe with columns renamed.
    """
    df.columns = ['ID', 'Task', 'Concept', 'Author', 'Medium', 'Structure', 'Selected']
    return df

def load_and_filter_data(file_name, exclude_ids):
    """
    Load the Dataframes from excel files and filter out the ID's previously
    determined to be removed.

    Parameters
    ----------
    file_name : str
        Name of the excel file to be loaded and filtered.
    exclude_ids : str
        Name of the excel file containing ID numbers of participants who need
        to be removed(e.g., duplicates, significant missing data, etc.).

    Returns
    -------
    filtered_df : pd.DataFrame
        Cleaned dataframe with unwanted columns removed, columns renamed,
        and excluded IDs removed.

    """
    df = load_data(file_name, sheet_names=[1,2])
    df = remove_unwanted_columns(df)
    df = rename_columns(df)
    filtered_df = df[~df['ID'].isin(exclude_ids)]
    return filtered_df

def count_unique_people(df):
    return len(df['ID'].unique())

def process_file(file_paths, exclude_ids_path):
    """
    Process multiple files, filter data, and 
    check for excluded IDs.
    
    Function iterates through list of file paths, loads
    and filters data from each file, and checks if any of
    the excluded IDs are still present in the filtered data.

    Parameters
    ----------
    file_paths : list of str
        List of file paths to excel files containing data to be
        processed.
    exclude_ids_path : str
        File path to CSV containing IDs to be excluded from data.
    """
    exclude_ids = pd.read_csv(exclude_ids_path)
    
    for file_path in file_paths:
        # Clean the file path to create a valid variable name
    
        filtered_df = load_and_filter_data(file_path, exclude_ids['Response_ID'].tolist())
    
        num_unique_people = count_unique_people(filtered_df)
        print(f'Number of unique people for {file_path}: {num_unique_people}')
        
        check_filter = filtered_df['ID'].isin(exclude_ids['Response_ID'])
        if check_filter.any():
            print('Warning: Some IDs from remove list are still present')
        else:
            print('All IDs removed')
            
        # Save filtered_df to csv file
        output_csv_file = file_path.replace('.xls', '_filtered.csv')
        filtered_df.to_csv(output_csv_file, index=False)
       

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process Excel files and filter data.")
    
    # Add arguments for file paths
    parser.add_argument("file_paths", nargs="+", help="List of excel file paths to process.")
    parser.add_argument("--exclude_ids_path", required=True, help="Path to CSV containing IDs to exclude from data.")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call the process_files function with the specified arguments
    process_file(args.file_paths, args.exclude_ids_path)

    
if __name__ == '__main__':
    main()
