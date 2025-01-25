from abc import ABC, abstractmethod

import pandas as pd
from django.db.models import Model


class DownsideStrategy(ABC):
    """
    Abstract class for downside strategies
    on creation for a deal, the downside strategy is selected and the downside is calculated
    It should follow the following steps:
    1. prepare_data
    2. generate_model
    3. calculate_downside

    When saving it to the database, the following steps should be followed:
    1. save_model

    When loading it from the database for updating the downside, the following steps should be followed:
    1. load_model
    2. calculate_downside

    """
    @abstractmethod
    def calculate_downside(self, latest_data: pd.DataFrame) -> float:
        """ with the existing model, calculate the downside for the latest data"""
        pass

    @abstractmethod
    def save_model(self) -> Model:
        """ function to save exisiting parameters needed to calculate downside to database"""
        pass


    @abstractmethod
    def load_model(self, db_model_data: dict):
        """ function to load exisiting parameters needed to calculate downside from database"""
        pass

    @abstractmethod
    def generate_model(self, params=None) -> bool:
        """ function to generate model, extra parameters can be passed to the function for additional configurations
        returns True if the model was generated successfully, False otherwise"""
        pass

    @abstractmethod
    def prepare_data(self, data) -> None:
        """ function to generate data needed prior to generating the model"""
        pass

    @abstractmethod
    def get_new_data(self, params=None) -> pd.DataFrame:
        """ function for fetching new data for updating the model"""
        pass




