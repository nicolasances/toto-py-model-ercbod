import os
import uuid
import numpy as np
import pandas as pd
import json
import pickle
import joblib
import re

from pandas import json_normalize
from datetime import datetime as dt, timedelta

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

from toto_logger.logger import TotoLogger

from util.history import HistoryDownloader
from util.feature import FeatureEngineering

from totoml.model import TrainedModel

logger = TotoLogger()

class Trainer: 

    def __init__(self): 
        pass

    def train(self, model_info, context):

        # Create the folder where to store all the data
        folder = "{tmp}/{model_name}/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], model_name=model_info['name'], fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)
        
        # 1. Download the data
        history_filename = HistoryDownloader().download(folder, context)

        # 2. Engineer features
        (features_filename, word_vectorizer, user_encoder) = FeatureEngineering().do(folder, history_filename, context)

        # 3. Train
        logger.compute(context.correlation_id, '[ {context} ] - [ TRAINING ] - Starting model training'.format(context=context.process), 'info')
        
        # Read the features
        features_df = pd.read_csv(features_filename, index_col=0)

        # Extraction of X and y
        X = features_df.drop(columns=['category'])
        y = pd.get_dummies(features_df['category'])

        # Traing the NN 
        model = MLPClassifier(max_iter=1000, hidden_layer_sizes=(500, 500), alpha=1e-06)
        
        model.fit(X, y)

        logger.compute(context.correlation_id, '[ {context} ] - [ TRAINING ] - Model training completed'.format(context=context.process), 'info')
        
        # 4. Score
        logger.compute(context.correlation_id, '[ {context} ] - [ SCORE ] - Scoring trained model'.format(context=context.process), 'info')

        # Encode the y labels, so that we can use them to compute metrics
        category_encoder = LabelEncoder()
        category_encoder.fit(y.columns)
        encoded_categories = category_encoder.transform(y.columns)

        # Score
        pred = model.predict(X)

        score = [
            {"name": "precision", "value" : precision_score(y, pd.DataFrame(pred, columns=encoded_categories), labels=encoded_categories, average='weighted')},
            {"name": "recall", "value": recall_score(y, pd.DataFrame(pred, columns=encoded_categories), labels=encoded_categories, average='weighted')},
            {"name": "f1", "value": f1_score(y, pd.DataFrame(pred, columns=encoded_categories), labels=encoded_categories, average='weighted')}
        ]

        logger.compute(context.correlation_id, '[ {context} ] - [ SCORE ] - Done. Training complete.'.format(context=context.process), 'info')

        # 5. Save all the objects
        model_filepath = "{folder}/model".format(folder=folder)
        word_vectorizer_filepath = "{folder}/word-vectorizer".format(folder=folder)
        user_encoder_filepath = "{folder}/user_encoder".format(folder=folder)
        categories_df_filepath = "{folder}/categories".format(folder=folder)

        joblib.dump(model, model_filepath)
        joblib.dump(word_vectorizer, word_vectorizer_filepath)
        joblib.dump(user_encoder, user_encoder_filepath)
        pd.DataFrame(y.columns, columns=['category']).to_csv(categories_df_filepath)

        # 6. Return the trained model objects
        return TrainedModel({"model": model_filepath, "description-vectorizer": word_vectorizer_filepath, "user-encoder": user_encoder_filepath, "categories": categories_df_filepath}, [history_filename, features_filename], score)