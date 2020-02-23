import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load data of messages and catagories. Two dataset will be merge together base on ID

    Args:
        messages_filepath: string, file path of csv messages dataset
        categroies_filepath: string, file path of csv categries dataset
    
    Returns:
        df: DataFrame, a merged data frame of messages and categries. 
    '''
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, how='left', left_on='id', right_on='id').drop(labels=["id"], axis=1)
    return df


def clean_data(df):
    '''
    Args:
        df: DataFrame, a data frame contains all messages and categories
    
    Returns:
        df: DataFrame, cleanup data
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    # select the first row of the categories dataframe
    title_row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = []
    for i in title_row:
        category_colnames.append(i.split("-")[0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories[1:]:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int).apply(lambda x: 1 if x == 1 else 0)

    # drop the original categories column from `df`
    df = df.drop(labels=["categories"], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    use cleaned data frame as input save it into the database file

    Args:
        df: cleaned dataframe 
        database_filename: target database file name
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages_categories', engine, index=False, if_exists = 'replace')


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