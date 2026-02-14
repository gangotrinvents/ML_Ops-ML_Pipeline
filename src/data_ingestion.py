# To fetch data from source

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

# Ensuring the "logs" directory exists: So that we can save all the logs
log_dir = 'logs'
os.makedirs(log_dir, exist_ok= True)

# Logging configuration
logger = logging.getLogger('data_ingestion') # we are creating logger object and name it as 'data ingestion'(editable)
logger.setLevel('DEBUG') # Assigned Debug level to our logger

# Console Handler configuration
console_handler = logging.StreamHandler() # we are having console handler
console_handler.setLevel('DEBUG') # Assign Debug level to our console handler

# File Handler configuration
log_file_path = os.path.join(log_dir, 'data_ingestion.log') # path to file logger creation
file_handler = logging.FileHandler(log_file_path) # creating file logger
file_handler.setLevel('DEBUG') # Assign Debug to our file handler

# Format configuration
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # defining format for logging
console_handler.setFormatter(formatter) # Assigning format to console handler
file_handler.setFormatter(formatter) # Assigning format to file handler

# Giving both handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url: str) -> pd.DataFrame:
    """ Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url) # At debug level logger will print the message mentioned 
        return df
    
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise

    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def pre_processing(df: pd.DataFrame) -> None:
    try:
        # Removing unnecessary columns
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace= True)
        # Columns name change
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace= True)
        logger.debug('Data pre-processing completed')
        return df
    
    except KeyError as e:
        logger.error('Missing column in the dataFrame: %s', e)
        raise
    
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """ Save the train and test datasets"""

    try:
        raw_data_path = os.path.join(data_path, 'raw') # Path where we have to save out data
        os.makedirs(raw_data_path, exist_ok= True) # creates the directory if doesnt exist
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index = False) # saving the file at that path
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index = False) # saving the file at that path
        logger.debug('Train and test data saved to %s', raw_data_path)
    
    except Exception as e:
        logger.error('Unexpected error occured while saving the data: %s', e)
        raise

def main():
    try:
        test_size = 0.2
        data_path = 'https://raw.githubusercontent.com/gangotrinvents/ML_Ops-ML_Pipeline/refs/heads/main/Experiment/spam.csv'
        df = load_data(data_url= data_path) # it will load the data from given path
        final_df = pre_processing(df) # runs the pre processing module and getting output
        train_data, test_data = train_test_split(final_df, test_size= test_size, random_state=2)
        
        # here './data' it will go to the very root (./) of the directory where we are working and create a folder 'data'
        save_data(train_data, test_data, data_path= './data')
    
    except Exception as e:
        logger.error('Failes to complete the data Ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()