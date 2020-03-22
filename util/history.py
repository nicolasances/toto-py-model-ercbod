from pandas import json_normalize

from toto_logger.logger import TotoLogger

from remote.expenses import ExpensesAPI

logger = TotoLogger()

class HistoryDownloader: 

    def __init__(self): 
        pass

    def download(self, folder, context):
        """
        Downloads the history of payments to the specified folder
        """
        # Date from which the payments should be downloaded
        dateGte = '20100101'

        logger.compute(context.correlation_id, '[ {context} ] - [ HISTORICAL ] - Starting historical data download from date {date}'.format(context=context.process, date=dateGte), 'info')

        history_filename = '{folder}/history.csv'.format(folder=folder);

        # Download
        json_response = ExpensesAPI().get_expenses(dateGte, 'all', context.correlation_id)

        # Extract the expenses array
        try: 
            expenses = json_response['expenses']
        except: 
            logger.compute(context.correlation_id, '[ {context} ] - [ HISTORICAL ] - Error reading the following microservice response: {r}'.format(context=context.process, r=json_response), 'error')
            logger.compute(context.correlation_id, '[ {context} ] - [ HISTORICAL ] - No historical data'.format(context=context.process), 'warn')
            return None

        # Create the data frame
        raw_data_df = json_normalize(expenses)

        # Drop useless data
        raw_data_df.drop(labels=['id', 'date', 'amount', 'monthly', 'currency', 'additionalData', 'amountInEuro', 'yearMonth', 'additionalData.monthId', 'additionalData.source', 'additionalData.supermarketListId', 'cardId', 'cardMonth', 'cardYear', 'consolidated', 'weekendId'], axis=1, inplace=True)
        
        # Save 
        raw_data_df.to_csv(history_filename) 

        logger.compute(context.correlation_id, '[ {context} ] - [ HISTORICAL ] - Historical data downloaded: {r} rows'.format(context=context.process, r=len(raw_data_df)), 'info')

        return history_filename