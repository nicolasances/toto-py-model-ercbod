import joblib
import pandas as pd

from util.feature import FeatureEngineering
from toto_logger.logger import TotoLogger

logger = TotoLogger()

class Predictor: 

    def __init__(self): 
        pass

    def predict (self, model, context, data):
        """
        Predicts the category of an expense given its basic info
        Requires the following data to be passed in data: "description", "user"
        """
        # 1. Feature engineering
        features_df = FeatureEngineering().do_for_predict(data, joblib.load(model.files['description-vectorizer']), joblib.load(model.files['user-encoder']), context)

        # 2. Load model & other required files
        trained_model = joblib.load(model.files['model'])
        categories = pd.read_csv(model.files['categories'], index_col=0)

        # 3. Predict
        pred = trained_model.predict(features_df)

        # 4. Return the prediction
        predicted_category = pd.DataFrame(pred, columns=categories['category']).idxmax(axis=1)[0]

        logger.compute(context.correlation_id, '[ {ctx} ] - [ PREDICTION ] - Model {model}.v{version} - Predicted category for description [{desc}]: {c}'.format(ctx=context.process, model=model.info['name'], version=model.info['version'], desc=data['description'], c=predicted_category), 'info')
        
        return {"category": predicted_category}