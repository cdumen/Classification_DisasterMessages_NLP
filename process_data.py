# Import required libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Data Loading Function
    INPUT:
        'message_filepath' : a string/path to a csv file
        'categories_filepath' : a string/path to a csv file
    OUTPUT:
        'df' : a loaded/transformed pandas dataframe
    '''
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    # Merge datasets
    df = messages.merge(categories, on = 'id')
        
    return df


def clean_data(df):
    '''
    Preprocess for Cleaning and Cleaning Fucntion
    INPUT:
        df: Merged dataset from messages and categories
    OUTPUT:
        df: Cleaned dataset
    '''
    # Create a df of the 36 individual category columns
    categories = df['categories'].str.split(pat = ';', expand = True)
    # Select the first row of the categories df
    row = categories.loc[0]
    # Use this row to extract a list of new column names for categories
    columns = []
    for col in row:
        columns.append(col[:-2])
        category_columns = columns
    
    #rename the columns of categories
    categories.columns = category_columns
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        #set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Replace the categories column in df with a new category columns
    # Drop the original categories column from df
    df.drop('categories', axis = 1, inplace = True)
    # Concatenate the original df with the new categories df
    df = pd.concat([df, categories], axis = 1)
    # Remove the duplicates
    df.drop_duplicates(inplace = True)
    
    return df


def save_data(df, database_filename):
    '''
    Function to save df to SQLite db
    INPUT:
        df: cleaned dataset
        database_filename: database name, e.g. DisasterMessages.db
    OUTPUT: 
        A SQLite database
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster', engine, index=False)
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()