import os
import pandas as pd
import requests

from toto_logger.logger import TotoLogger

logger = TotoLogger()

toto_auth = os.environ['TOTO_API_AUTH']
toto_host = os.environ['TOTO_HOST']

class ExpensesAPI: 

    def __init__(self):
        pass

    def get_expenses(self, dateGte, user, correlation_id): 
        """
        Retrieves the expenses from Toto Expenses API

        Parameters
        ----------
        dateGte (str)
            The date from which the expenses should be downloaded.
            The date is a string in a YYYYMMDD format

        user (str)
            The user for which the expenses should be downloaded
        """
        response = requests.get(
            'https://{host}/apis/expenses/expenses?user={user}&dateGte={dateGte}'.format(user=user, dateGte=dateGte, host=toto_host),
            headers={
                'Accept': 'application/json',
                'Authorization': toto_auth,
                'x-correlation-id': correlation_id
            }
        )

        # Convert to JSON
        return response.json()
