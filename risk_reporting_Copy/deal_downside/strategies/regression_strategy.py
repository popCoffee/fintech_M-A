import json
import pickle
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from django.db.models import Model

from bbgclient import bbgclient
from risk.models import MaDownsidePeerSource, MaDealsActionIdDetails
from risk_reporting.deal_downside.strategies.downside_strategy import DownsideStrategy
from risk_reporting.models import LinearRegressionDownside


def has_more_than_n_continuous_empty_days(df: pd.DataFrame, column_name: str, n: int) -> bool:
    """
    Check if the dataframe has more than `n` continuous days of empty data.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    column_name (str): The column to check for empty data.
    n (int): The threshold for continuous empty days.

    Returns:
    bool: True if there are more than `n` continuous days of empty data, False otherwise.
    """
    # Validate inputs
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the dataframe")
    if n < 1:
        raise ValueError("The threshold 'n' must be a positive integer")
    c_empty: int = max(df[column_name].isnull().astype(int)
                       .groupby(df[column_name].notnull().astype(int).cumsum()).sum())
    return c_empty > n


def trim_data(data: pd.DataFrame, year_duration: int) -> pd.DataFrame:
    """
    Trim the input dataframe to the last n years.

    Parameters:
    - data (pd.DataFrame): The input dataframe containing stock prices with a 'Date' column.
    - year_duration (int): The number of years to keep in the dataframe.

    Returns:
    - trimmed_data (pd.DataFrame): The trimmed dataframe.
    """

    assert year_duration > 0, "Year duration must be a positive integer"
    assert 'Date' in data.columns, "The 'Date' column is missing in the dataframe"

    # Ensure the 'Date' column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    # Set 'Date' as the index
    data.set_index('Date', inplace=True)

    # Get the last n years of data
    end_date = data.index[-1]
    start_date = end_date - pd.DateOffset(years=year_duration)
    trimmed_data = data[(data.index >= start_date) & (data.index <= end_date)]

    # Check each column of the dataframe for more than 5 continuous empty days, if so, remove the column
    for column in trimmed_data.columns:
        if has_more_than_n_continuous_empty_days(trimmed_data, column, n=5):
            trimmed_data = trimmed_data.drop(columns=[column])

    # back-fill and then forward fill empty values
    trimmed_data = trimmed_data.fillna(method='bfill')
    trimmed_data = trimmed_data.fillna(method='ffill')

    return trimmed_data


class LinearRegressionStrategy(DownsideStrategy):
    """ process for generating regression model for downside calculation"""

    def __init__(self, model_type=""):
        self.model = None
        self.pickled_model = None
        self.model_params = {}
        self.model_type = model_type
        self.peers = []
        self.target_ticker = ''
        self.time_series_data = pd.DataFrame()
        self.year_duration = None
        self.peer_source: Optional[MaDownsidePeerSource] = None

    def prepare_data(self, data: MaDownsidePeerSource) -> None:
        """ function for getting time series data used for generating the regression model
            Pulls historical price data for the target ticker and peers of the last 5 years
            :param data: MaDownsidePeerSource peer source object with ticker list
        """
        self.peer_source: MaDownsidePeerSource = data
        self.target_ticker: str = data.deal_id.target_ticker
        self.peers: list = [ticker.strip() for ticker in self.peer_source.proxy_list.split(',')]

        # Get the unaffected date from the action id details
        action_id_detail = MaDealsActionIdDetails.objects.filter(pk=data.deal_id.action_id).first()

        # Get the start and end date for the historical price data
        unaffected_date = action_id_detail.unaffected_date
        if not unaffected_date:
            return None
        end_date = unaffected_date
        start_date = end_date - timedelta(days=365 * 5)
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        ticker_list = self.peers + [self.target_ticker]
        ticker_list = list(set(ticker_list))  # remove duplicates

        # fetch tickers historical price data 5 years back to unaffected date
        prices_dict = bbgclient.get_timeseries_dict(ticker_list, ['PX_LAST'], start_date_str, end_date_str)
        prices_df = pd.DataFrame()

        for each_ticker in ticker_list:
            px_df = pd.DataFrame.from_dict(prices_dict[each_ticker]['PX_LAST']).reset_index().rename(
                columns={'index': 'Date', 0: each_ticker})[['Date', each_ticker]].copy().dropna()
            if prices_df.empty:
                prices_df = px_df
            else:
                prices_df = pd.merge(prices_df, px_df, on='Date', how='outer')
        prices_df = prices_df.sort_values(by='Date')
        self.time_series_data = prices_df

    def create_ols_model(self, data: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper or None:
        """
        Create an OLS model based on input dataframe.

        Parameters:
        - data (pd.DataFrame): The input dataframe containing stock prices with a 'Date' column.
        - target_ticker (str): The column name of the target ticker for prediction.

        Returns:
        - model (sm.regression.linear_model.RegressionResultsWrapper): The fitted OLS model.
        """

        # Ensure target ticker is in the dataframe and contains at least one other non-empty column, if not, return None
        if self.target_ticker not in data.columns:
            return None
        if len(data.columns) < 2:
            return None

        # Split the data into X (independent variables) and y (dependent variable)
        x = data.drop(columns=[self.target_ticker])
        y = data[self.target_ticker]

        # Add a constant to the independent variables matrix
        x = sm.add_constant(x)

        # Fit the OLS model
        model = sm.OLS(y, x).fit()

        return model

    def generate_model(self, params=None) -> bool:
        """
        Clears the existing model and generates a new model based on the time series data and optionally input params.
        """
        if params is None:
            params = {}
        if self.time_series_data.empty:
            # print("No time series data available for generating the model")
            return False
        if not self.target_ticker or self.target_ticker not in self.time_series_data.columns:
            # print("Target ticker not available in the time series data")
            return False

        self.year_duration = params.get('year_duration', 5)
        self.model = None
        self.model_type = params.get('model_type', self.model_type)
        self.model_params = dict()

        # print(f"Generating model for year duration: {self.year_duration}")
        data = trim_data(self.time_series_data.copy(), self.year_duration)

        # @TODO currently running only for daily price, standardized price or other values could be used
        model = self.create_ols_model(data)

        if model is None:
            return False

        self.model = model
        # save model params for faster access
        self.model_params['params'] = model.params
        self.model_params['r_squared'] = model.rsquared
        self.model_params['r_squared_adj'] = model.rsquared_adj
        self.model_params['summary'] = model.summary().as_html()
        return True

    def get_new_data(self, params=None) -> pd.DataFrame:
        # pull the latest data based on existing model
        # we only need the peers for predicting the target ticker downside
        latest_data = pd.DataFrame()
        tickers = self.peers + [self.target_ticker]
        if self.model_type == LinearRegressionDownside.DAILY_PRICES:
            latest_data = pd.DataFrame.from_dict(bbgclient.get_secid2field(tickers, 'tickers', ['PX_LAST'],
                                                                           overrides_dict={
                                                                               'BEST_FPERIOD_OVERRIDE': '1BF'},
                                                                           req_type='refdata'))

            # clean up data, convert data to numeric but keep none for empty values
            latest_data = latest_data.applymap(lambda x: pd.to_numeric(x[0], errors='coerce') if x != [None] else None)
        return latest_data

    def calculate_downside(self, latest_data: pd.DataFrame) -> float:
        assert self.model_params is not None, "No available model params, model was not generated"
        needed_tickers = list(self.model_params['params'].index)
        needed_tickers.remove('const')

        latest_x = latest_data[needed_tickers]
        latest_x = sm.add_constant(latest_x, has_constant='add')

        latest_x = latest_x.fillna(value=np.nan)

        # show which column is missing
        # missing_columns = latest_x.columns[latest_x.isnull().any()].tolist()
        # if missing_columns:
        #     print(f"Missing columns: {missing_columns}")

        # Calculate the predicted value using the dot product
        downside = (latest_x @ self.model_params['params']).iloc[0]
        return max(downside, 0)  # downside should be positive

    def serialize_model_params(self) -> str:
        """ helper function to serialize model params for saving in the database """
        peers = self.peers
        target_ticker = self.target_ticker
        params = self.model_params['params'].to_json()
        r_squared = self.model_params['r_squared']
        r_squared_adj = self.model_params['r_squared_adj']
        summary = self.model_params['summary']
        model_params = {'peers': peers,
                        'target_ticker': target_ticker,
                        'params': params,
                        'r_squared': r_squared,
                        'r_squared_adj': r_squared_adj,
                        'summary': summary}

        return json.dumps(model_params)

    def deserialize_model_params(self, json_model_params: str) -> None:
        """ helper function to deserialize model params from the database """
        model_params = json.loads(json_model_params)
        self.model_params = {}
        self.peers = model_params['peers']
        self.target_ticker = model_params['target_ticker']
        self.model_params['params'] = pd.read_json(model_params['params'], typ='series')
        self.model_params['r_squared'] = model_params['r_squared']
        self.model_params['r_squared_adj'] = model_params['r_squared_adj']
        self.model_params['summary'] = model_params['summary']

    def save_model(self) -> Model:
        saved_model = None
        try:
            pickled_model = pickle.dumps(self.model)
            self.pickled_model = pickled_model
            serialized_model_params = self.serialize_model_params()

            saved_model = LinearRegressionDownside(peer_source=self.peer_source,
                                                   model_type=self.model_type,
                                                   model_params=serialized_model_params,
                                                   pickled_model=pickled_model,
                                                   year_multiple=self.year_duration)
            saved_model.save()
        except:
            import traceback
            traceback.print_exc()

        return saved_model

    def load_model(self, db_model: LinearRegressionDownside) -> None:
        assert isinstance(db_model, LinearRegressionDownside), "Wrong model type passed"
        self.peer_source = db_model.peer_source
        self.model_type = db_model.model_type
        self.deserialize_model_params(db_model.model_params)
        self.pickled_model = db_model.pickled_model  # lazy load model, only parse the model when needed
        self.year_duration = db_model.year_multiple
