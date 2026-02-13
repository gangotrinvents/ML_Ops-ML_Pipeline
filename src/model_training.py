import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

# Ensure Log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok= True)

# Logging configuration
logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File Handler
file_log_path = os.path.join(log_dir,'model_training.log')
file_handler = logging.FileHandler(file_log_path)
file_handler.setLevel('DEBUG')

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str)->pd.DataFrame:
    """
    Load data from data_feature_engineering
    Load CSV file
    input: path to csv file
    output: dataframe
    """
    try:
        ld_data = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s: ', file_path, ld_data.shape)
        return ld_data
    
    except pd.errors.ParserError as e:
        logger.error("Faile to parse the CSV file: %s", e)
        raise

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise 
    except Exception as e:
        logger.error("Unexpected error occured while loading the data: %s", e)
        raise

def train_model(x_train: np.ndarray, y_train: np.ndarray, params: dict)->RandomForestClassifier:
    """
    Train the Random Forest Classifier Model
    :x_train: Training features
    :y_train: Target feature
    :params: Dictionary of hyperparameter
    return: RandomForestClassifier Model
    """
    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("The no of samples in both x_train and y_train should be the same")
        logger.debug("Intializing RandomForecast model with parameters: %s", params)
        model = RandomForestClassifier(n_estimators= params['n_estimators'], random_state= params['random_state'])
        
        logger.debug("Model training started with %d samples", x_train.shape[0])
        model.fit(x_train, y_train)

        logger.debug("Model training completed")
        return model
    
    except ValueError as e:
        logger.error("ValueError during the model training: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error while model training: %s", e)
        raise

def save_model(model, file_path: str)->None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok= True)

        # Open the file with write access
        with open(file_path, 'wb') as file:
            # Dump the model in file where it has write option as well
            print(os.path.dirname(file_path))
            pickle.dump(model, file)
        logger.debug("Model saved to %s", file_path)
    
    except FileNotFoundError as e:
        logger.error("File path not found %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while saving the model %s", e)
        raise

def main():
    try:
        params = {'n_estimators': 25, 'random_state': 2}
        train_data = load_data('./data/processed/train_tfidf.csv')
        x_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1:].values

        model = train_model(x_train= x_train, y_train= y_train, params= params)
        
        # in this way it will look in the directory we are working in and create file accordingly
        save_model_path = 'models/model.pkl'

        # below code will make the file directly in drive only not in directory we are working in
        # only difference in "/" in front model
        # save_model_path = '/models/model.pkl'
        save_model(model= model, file_path= save_model_path)

        logger.debug("Model is saved at %s", save_model_path)
    
    except Exception as e:
        logger.error("Unexpected error occured and failed to completed model building %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()