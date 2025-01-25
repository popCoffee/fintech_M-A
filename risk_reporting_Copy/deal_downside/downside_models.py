import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum
from risk_reporting.deal_downside.strategies.regression_strategy import LinearRegressionStrategy

class DownsideType(Enum):
    EQUAL_WEIGHTED = 'equal_weighted'
    REGRESSION = 'regression'
