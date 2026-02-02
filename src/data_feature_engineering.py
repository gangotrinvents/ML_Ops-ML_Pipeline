import os 
import logging
# Necessary Libraries
import numpy as np              # For numeric operations
import pandas as pd             # For data manipulation and analysis
import matplotlib.pyplot as plt # For data visualization
# %matplotlib inline              

from sklearn.preprocessing import LabelEncoder  # Label Encoder
from nltk.stem.porter import PorterStemmer      # Importing the Porter stemmer for text Stemming
from wordcloud import WordCloud                 # To visualize text
from sklearn.feature_extraction.text import TfidfVectorizer

# For Natural Language Processing
import nltk
from nltk.corpus import stopwords

# Importing the string module for handling special characters
import string

log_dir = 'logs'
os.makedirs(log_dir, exist_ok= True)

logger = logging.getLogger('data_feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler_path = os.path.join(log_dir,'data_feature_engineering_log')
file_handler = logging.FileHandler(file_handler_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path) -> pd.DataFrame:
    """
    Loading data
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace= True)
        logger.debug("Data load completed...")
        return df
    
    except pd.errors.ParserError as e:
        logger.debug("Failed to parse the CSV %s", e)
        raise

    except Exception as e:
        logger.debug("Unexpected error occured while loading the data: %s", e)
        raise 

def apply_tfid(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Applying TF-IDF on data for vectorization
    """
    try: 

        # Vector initialization
        vectorizer = TfidfVectorizer(max_features=max_features)

        # Extract data
        x_train = train_data['text'].values
        y_train = train_data['target'].values
        x_test = test_data['text'].values
        y_test = test_data['target'].values

        # Conversion to vectorizer
        x_train_vect = vectorizer.fit_transform(x_train)
        x_test_vect = vectorizer.transform(x_test)
        
        # Convert to array and assign label
        train_df = pd.DataFrame(x_train_vect.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_vect.toarray())
        test_df['label'] = y_test

        logger.debug('Bag of words applied and data transformed')
        return train_df, test_df
    
    except Exception as e:
        logger.error("Error during Vectorization: %s", e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """ Save the dataframe to a CSV file."""
    try: 
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        df.to_csv(file_path, index= False)
        logger.debug("Data saved to %s", file_path)
    
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise

def main():
    try:   

        # No of features/columns we need after the vectorization 
        max_features = 100

        train_data = load_data('./data/intermediate/train_processed.csv')
        test_data = load_data('./data/intermediate/test_processed.csv')

        train_df, test_df = apply_tfid(train_data= train_data, test_data= test_data, max_features= max_features)

        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
    
    except Exception as e:
        logger.error("Failed to complete the feature Engineering Process: %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()