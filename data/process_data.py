# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads our messages and categories data through these steps:
    1. Read the messages and categories data from csv file into two separate pandas DataFrames based on file paths
    2. Merge them into a new DataFrame 'df'
    3. Return 'df'
    
    Args:
    messages_filepath (str)
    categories_filepath (str)
    
    Returns:
    df - DataFrame
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on = 'id')
    
    return df

    
def clean_data(df):
    '''
    This function cleans the DataFrame, which includes splitting a category column into individual category columns. 
    
    Args:
    df - DataFrame
    
    Returns:
    df_clean - DataFrame
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat = ';', expand=True)
    
    # selecting the first row of the categories dataframe
    row = categories.loc[0,:]
    
    # extracting a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:].astype(str)
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)    
    
    # concatenate the original dataframe with the new `categories` dataframe
    frames = [df, categories]

    df_clean = pd.concat(frames, axis=1)
    
    # drop duplicates
    df_clean = df_clean.drop_duplicates()
    
    return df_clean
    
        
def save_data(df, database_filename):
    '''
    This function cleans the DataFrame, which includes splitting a category column into individual category columns. 
    
    Args:
    df - DataFrame
    
    Returns:
    N/A
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('df', engine, index=False)


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