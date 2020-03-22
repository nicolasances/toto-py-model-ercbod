import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

from toto_logger.logger import TotoLogger

logger = TotoLogger()

class FeatureEngineering: 

    def __init__(self): 
        pass

    def __basic_ops(self, raw_data_df): 
        """
        This private method does the basic operations that are common and always used in feature engineering 
        on the raw data. 
        Those are lower casing of the description and removal of all special characters. 

        TODO: lemmization of the words

        Parameters
        ----------
        raw_data_df (pandas.DataFrame)
            the data on which the engineering must be performed

        Returns
        -------
        transformed_data (pandas.DataFrame)
            the transformed data

        """
        # Transformations on the DESCRIPTION COLUMN
        # Lower case 
        raw_data_df.loc[:,'description'] = raw_data_df['description'].str.lower()

        # Replace special characters
        raw_data_df.loc[:,'description'] = raw_data_df['description'].apply(lambda str : re.sub(r"[\/\-\&\'\(\)\\\.\*\,]*", "", str))
        raw_data_df.loc[:,'description'] = raw_data_df['description'].apply(lambda str : re.sub(r"[0-9]*", "", str))
        raw_data_df.loc[:,'description'] = raw_data_df['description'].apply(lambda str : re.sub(r"[éèêëęėē]", "e", str))
        raw_data_df.loc[:,'description'] = raw_data_df['description'].apply(lambda str : re.sub(r"[øòóºôöõœoō]", "o", str))
        raw_data_df.loc[:,'description'] = raw_data_df['description'].apply(lambda str : re.sub(r"[àáªâäæãåā]", "a", str))
        raw_data_df.loc[:,'description'] = raw_data_df['description'].apply(lambda str : re.sub(r"[ùúûüū]", "u", str))
        raw_data_df.loc[:,'description'] = raw_data_df['description'].apply(lambda str : re.sub(r"[ìíîïįī]", "i", str))
        raw_data_df.loc[:,'description'] = raw_data_df['description'].apply(lambda str : re.sub(r"[çćč]", "c", str))

        return raw_data_df


    def do(self, folder, raw_data_filename, context): 
        """
        Engineers the features
        """
        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Starting feature engineering'.format(context=context.process), 'info')

        # Load the data
        raw_data_df = pd.read_csv(raw_data_filename)

        # Standard transforms
        raw_data_df = self.__basic_ops(raw_data_df)

        # Create the bag of words
        cv_desc = CountVectorizer()

        cv_matrix = cv_desc.fit_transform(raw_data_df['description']).toarray()

        descriptions_df = pd.DataFrame(cv_matrix, columns=cv_desc.get_feature_names())

        # Change the user into a number
        ohe_user = OneHotEncoder()
        
        ohe_user_matrix = ohe_user.fit_transform(raw_data_df['user'].to_numpy().reshape(-1, 1)).toarray()

        users_df = pd.DataFrame(ohe_user_matrix, columns=ohe_user.get_feature_names())

        # Create the final features df
        features_df = pd.concat([raw_data_df['category'], users_df, descriptions_df], axis=1)

        # Save the features
        features_filename = '{folder}/features.csv'.format(folder=folder);

        features_df.to_csv(features_filename)

        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Feature engineering completed. Features Shape: {r}'.format(context=context.process, r=features_df.shape), 'info')

        # Return the file and the vectorizer
        return (features_filename, cv_desc, ohe_user)

    def do_for_predict(self, data, description_vectorizer, user_encoder, context): 
        """
        Engineers the features for a single expense

        Parameters
        ----------
        data (dict)
            A dictionnary with the following fields: "description", "user"

        Returns
        -------
        features_df (DataFrame)
            A Pandas DataFrame with the engineered features to be used in the prediction
        """
        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Starting feature engineering'.format(context=context.process), 'info')

        # Load the data
        raw_data_df = pd.DataFrame([[data['user'], data['description']]], columns=['user', 'description'])

        # Standard transforms
        raw_data_df = self.__basic_ops(raw_data_df)

        # Create the bag of words
        cv_matrix = description_vectorizer.transform(raw_data_df['description']).toarray()
        descriptions_df = pd.DataFrame(cv_matrix, columns=description_vectorizer.get_feature_names())

        # Change the user into a number
        user_matrix = user_encoder.transform(raw_data_df['user'].to_numpy().reshape(-1, 1)).toarray()
        users_df = pd.DataFrame(user_matrix, columns=user_encoder.get_feature_names())

        # Create the final features df
        features_df = pd.concat([users_df, descriptions_df], axis=1)

        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Feature engineering completed. Features shape: {s}'.format(context=context.process, s=features_df.shape), 'info')

        # Return the features
        return features_df

    def do_for_score(self, folder, raw_data_filename, description_vectorizer, user_encoder, context): 
        """
        Engineers the features for the scoring process. 
        IMPORTANT: why is it a separate method? 
            Well, because if new expenses have come in, you cannot add features for eventual new description words, 
            so you need to reuse the description_vectorizer and user_encoder!
        """
        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Starting feature engineering'.format(context=context.process), 'info')

        # Load the data
        raw_data_df = pd.read_csv(raw_data_filename)

        # Standard transforms
        raw_data_df = self.__basic_ops(raw_data_df)

        # Create the bag of words for the description
        cv_matrix = description_vectorizer.transform(raw_data_df['description']).toarray()
        descriptions_df = pd.DataFrame(cv_matrix, columns=description_vectorizer.get_feature_names())

        # Change the user into a number
        user_matrix = user_encoder.transform(raw_data_df['user'].to_numpy().reshape(-1, 1)).toarray()
        users_df = pd.DataFrame(user_matrix, columns=user_encoder.get_feature_names())

        # Create the final features df
        features_df = pd.concat([raw_data_df['category'], users_df, descriptions_df], axis=1)

        # Save the features
        features_filename = '{folder}/features.csv'.format(folder=folder);

        features_df.to_csv(features_filename)

        logger.compute(context.correlation_id, '[ {context} ] - [ FEATURE ENGINEERING ] - Feature engineering completed. Features shape: {s}'.format(context=context.process, s=features_df.shape), 'info')

        # Return the features
        return features_filename
