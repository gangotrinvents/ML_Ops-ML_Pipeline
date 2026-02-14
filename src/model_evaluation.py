import logging
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import os

# Ensure log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok= True)

# Calling Logging
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File Handler
file_log_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(file_log_path)
file_handler.setLevel('DEBUG')

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Assign Formatter to handler
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Assigne handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    """
    Load the model
    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)

        logger.debug('Model is loaded from: %s', file_path)
        return model
    
    except FileNotFoundError as e:
        logger.error('File not found on the given path: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected Error occured while loading the model')
        raise

def load_data(file_path: str)-> pd.DataFrame:
    """
    Load the Test data from
    return test data
    """

    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)

        return df
    
    except pd.errors.ParserError as e:
        logger.error("Not able to parse the CSV file: %s", e)
        raise
    
    except Exception as e:
        logger.error("Unexpected Error occured while loading the model")
        raise

def model_evaluation(x_test: np.ndarray, y_test: np.ndarray, model):
    """
    Input dataset and model
    Evaluating the metrics
    Output metrics
    """
    try:
        y_predict = model.predict(x_test)

        # It will gives us the probability of each class: if its binary then 0 and 1 or 'No' and 'Yes' or False and True
        # First column of below out will be 0 or 'No' or False
        # Second column of below out will be 1 or 'Yes' or True
        # For ROC and AUC curve we are going to consider second column which is positive class and tell us positive rate 
        y_pred_proba = model.predict_proba(x_test)[:,1]
        print(y_pred_proba)
        print(y_predict)

        # Caculating the metrices
        accuracy = accuracy_score(y_test, y_predict)
        precision = precision_score(y_test, y_predict)
        recall = recall_score(y_test, y_predict)
        print("sdfdsfsd")
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        metrics = {
            'accuracy_score': accuracy,
            'precision_score': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }

        logger.debug('Evaluation metrics has been  calculated')        
        return metrics
    
    except Exception as e:
        logger.error('Unexpected Error occured while evaluating the metrics: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str):
    """
    Saving the Evaluation metrics
    Input: Metrics and file location to save
    """ 
    try:
        # Ensure directory is there to save evaluation metrics
        os.makedirs(os.path.dirname(file_path), exist_ok= True)

        with open(file_path,'w') as file:
            json.dump(metrics, file, indent= 4)
        logger.debug("Metrics saved on the location: %s", file_path)

    except Exception as e:
        logger.error('Unexpected error occured while saving the metrics: %s', e)
        raise

def main():
    try:
        model_path = 'models/model.pkl'
        test_data_path = 'data/processed/test_tfidf.csv'
        metric_path = 'metrics/metrics.json'

        model = load_model(model_path)
        print("model collected: ",model)

        data = load_data(test_data_path)
        print("load data")

        x_test = data.iloc[:,:-1].values
        y_test = data.iloc[:,-1:].values
        
        metrics = model_evaluation(x_test, y_test, model)
        print("metrics calcualted",metrics)

        save_metrics(metrics, metric_path)
        print("Metrics are saved now")

    
    except FileNotFoundError as e:
        logger.error('File not found on location: %s', e)
    
    except Exception as e:
        logger.error("Unexpected Error occured while model evaluation process: %s", e)

if __name__ == '__main__':
    main()