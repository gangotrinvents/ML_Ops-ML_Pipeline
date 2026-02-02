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

# For Natural Language Processing
import nltk
from nltk.corpus import stopwords

# Importing the string module for handling special characters
import string

# Downloading NLTK data
nltk.download('stopwords') # Downloading stopwords data
nltk.download('punkt')     # Downloading tokenizer data

# Creating an instance of the Porter Stemmer
ps = PorterStemmer()

#---------Logging---------------
# Ensure log directory exist
log_dir = 'logs'
os.makedirs(log_dir, exist_ok= True) # will create directory in same folder

# Logging configuration

# It create logger object named "data_preprocessing" and store that object in logger
# if we keep getlogger() empty then we will never know which file logged the message and deficult to debug
logger = logging.getLogger('data_preprocessing') 
logger.setLevel('DEBUG') # logger level

# Console Handler configuration
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File Handler configuration
    # create the file where to save logs
log_file_path = os.path.join(log_dir,'data_preprocessing_log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Format Configuration
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # defining the format 
console_handler.setFormatter(formatter) # Assign the formatter to console
file_handler.setFormatter(formatter) # Assign the formatter to console

# Giving both handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation and stemming
    """
    ps = PorterStemmer()

    # Convert to lowercase
    text = text.lower()

    # Tokenization using NLTK
    text = nltk.word_tokenize(text)
    
    # Removing special characters
    text = [word for word in text if word.isalnum()]
    
    # Loop through the tokens and remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Stemming using Porter Stemmer
    text = [ps.stem(word) for word in text]
    
    # Join the processed tokens back into a single string
    return " ".join(text)

 
def preprocess(df):
    """
    Preprocess the dataframe by encoding the target column, removing duplicates and transforming the text column.
    """
    try:
        logger.debug("Starting Pre-Processing for DataFrame")

        # Encoding text to numerical
        encoder = LabelEncoder()
        df['target'] = encoder.fit_transform(df['target'])
        logger.debug("Taget column encoded")
        # Drop duplicate
        df = df.drop_duplicates(keep='first')
        logger.debug("Duplicate rows removed")
        #Apply text_transformation to specified text column
        df['transformed_text'] = df['text'].apply(transform_text)
        logger.debug("Text column Transformed")
        return df
    
    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error during text normalization: %s", e)
        raise

def main():
    """
    Main function to load raw data, preprocess it, and text transformation and save the data
    """
    try:
        # Fetch the data from data/raw
        
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Data Fetched")

        # Transform the data
        train_transform_data = preprocess(train_data)
        text_transform_data = preprocess(test_data)
        print(train_transform_data.head())
        # Store the data inside data/processd
        data_path = os.path.join("./data", "intermediate")
        os.makedirs(data_path, exist_ok = True)

        train_transform_data.to_csv(os.path.join(data_path, "train_processed.csv"), index = False)
        text_transform_data.to_csv(os.path.join(data_path, "test_processed.csv"), index = False)

        logger.debug("Processed data saved to %s", data_path)
    
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise

    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s", e)
        raise
    except Exception as e:
        logger.error("Failed to complete the data tranformation process: %s", e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()