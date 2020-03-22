import os
import uuid
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from toto_logger.logger import TotoLogger

from util.history import HistoryDownloader
from util.feature import FeatureEngineering

from totoml.model import ModelScore

logger = TotoLogger()

class Scorer: 
    """
    This class does the batch scoring of the model
    """

    def __init__(self): 
        pass

    def score(self, model, context): 
        """
        Score the provided model 
        """
        logger.compute(context.correlation_id, '[ {context} ] - Scoring model {m}.v{v}'.format(context=context.process, m=model.info['name'], v=model.info['version']), 'info')

        # Create the folder where to store all the data
        folder = "{tmp}/{model_name}/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], model_name=model.info['name'], fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)
        
        # 1. Download the data
        history_filename = HistoryDownloader().download(folder, context)

        # 2. Engineer features
        features_filename = FeatureEngineering().do_for_score(folder, history_filename, joblib.load(model.files['description-vectorizer']), joblib.load(model.files['user-encoder']), context)

        # 3. Score
        # Read the features
        features_df = pd.read_csv(features_filename, index_col=0)

        # Extraction of X and y
        X = features_df.drop(columns=['category'])
        y = pd.get_dummies(features_df['category'])

        trained_model = joblib.load(model.files['model'])

        y_pred = trained_model.predict(X)

        # Encode the y labels, so that we can use them to compute metrics
        category_encoder = LabelEncoder()
        category_encoder.fit(y.columns)
        encoded_categories = category_encoder.transform(y.columns)

        score = [
            {"name": "precision", "value": precision_score(y, pd.DataFrame(y_pred, columns=encoded_categories), labels=encoded_categories, average='weighted')},
            {"name": "recall", "value": recall_score(y, pd.DataFrame(y_pred, columns=encoded_categories), labels=encoded_categories, average='weighted')},
            {"name": "f1", "value": f1_score(y, pd.DataFrame(y_pred, columns=encoded_categories), labels=encoded_categories, average='weighted')}
        ]

        return ModelScore(score, [history_filename, features_filename])
