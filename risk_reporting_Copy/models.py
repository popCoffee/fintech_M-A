import json
import os

import pandas as pd
from django.contrib.contenttypes.fields import GenericForeignKey, GenericRelation
from django.contrib.contenttypes.models import ContentType
from django.db import models
from risk.models import MA_Deals, MaDownsidePeerSource


# Create your models here.

class RiskAttributes(models.Model):
    strategy = models.CharField(max_length=50)  # Denotes the Tradegroup
    underlying_ticker = models.CharField(max_length=30)  # Underlying deal ticker
    target_acquirer = models.CharField(max_length=8)  # denotes whether current is target/acquirer if position is hedged
    risk_limit = models.CharField(max_length=10, null=True)  # Risk limit for the Deal
    currency = models.CharField(max_length=6)  # Curreny notation (eg USD)
    underlying_current_price = models.FloatField(default=0)  # Live price of the underlying ticker
    downside_base_case = models.FloatField()
    downside_outlier = models.FloatField()
    analyst = models.CharField(max_length=10)  # Allocated analyst
    last_update_date = models.DateTimeField(default='1900-01-01',null=True)  # Datetime stamp of the last downside update
    notes = models.TextField(null=True)  # Specific notes for each Tradegroup by analyst


class ArbNAVImpacts(models.Model):
    ''' This model is to be updated via a scheduled Celery Job every 15 mins from Market open to close..'''
    TradeDate = models.DateField()
    FundCode = models.CharField(max_length=10)
    TradeGroup = models.CharField(max_length=100)
    Sleeve = models.CharField(max_length=100)
    Bucket = models.CharField(max_length=100)
    Underlying = models.CharField(max_length=100)
    Ticker = models.CharField(max_length=100)
    BloombergGlobalId = models.CharField(max_length=100, null=True)
    SecType = models.CharField(max_length=10, null=True)
    MarketCapCategory = models.CharField(max_length=100, null=True)
    DealTermsCash = models.FloatField(null=True)
    DealTermsStock = models.FloatField(null=True)
    DealValue = models.FloatField(null=True)
    DealClosingDate = models.DateField(null=True)
    AlphaHedge = models.CharField(max_length=50)
    NetMktVal = models.FloatField()
    FxFactor = models.FloatField(null=True)
    Capital = models.FloatField(null=True)
    BaseCaseNavImpact = models.FloatField(null=True)
    OutlierNavImpact = models.FloatField(null=True)
    QTY = models.FloatField()
    LongShort = models.CharField(max_length=20)
    CatalystRating = models.FloatField(null=True)
    NAV = models.FloatField(null=True)
    Analyst = models.CharField(max_length=100, null=True)
    PM_BASE_CASE = models.FloatField(null=True)
    Outlier = models.FloatField(null=True)
    StrikePrice = models.FloatField(null=True)
    PutCall = models.CharField(max_length=20, null=True)
    LastPrice = models.FloatField(null=True)
    CurrMktVal = models.FloatField(null=True)
    RiskLimit = models.FloatField()
    PL_BASE_CASE = models.FloatField(null=True)
    BASE_CASE_NAV_IMPACT = models.FloatField(null=True)
    OUTLIER_PL = models.FloatField(null=True)
    OUTLIER_NAV_IMPACT = models.FloatField(null=True)


class DailyNAVImpacts(models.Model):
    ''' Model to Store latest NAV Impacts for each TradeGroup '''
    TradeGroup = models.CharField(max_length=100)
    RiskLimit = models.FloatField()
    LastUpdate = models.DateField(null=True)  # Reflects the date on which the underlying formula was updated
    BASE_CASE_NAV_IMPACT_AED = models.CharField(max_length=100)
    BASE_CASE_NAV_IMPACT_ARB = models.CharField(max_length=100)
    BASE_CASE_NAV_IMPACT_CAM = models.CharField(max_length=100)
    BASE_CASE_NAV_IMPACT_LEV = models.CharField(max_length=100)
    BASE_CASE_NAV_IMPACT_LG = models.CharField(max_length=100)
    BASE_CASE_NAV_IMPACT_MACO = models.CharField(max_length=100)
    BASE_CASE_NAV_IMPACT_TAQ = models.CharField(max_length=100)
    BASE_CASE_NAV_IMPACT_WED = models.CharField(max_length=100, null=True)
    BASE_CASE_NAV_IMPACT_WIC = models.CharField(max_length=100, null=True)
    BASE_CASE_NAV_IMPACT_MALT = models.CharField(max_length=100, null=True)
    BASE_CASE_NAV_IMPACT_PRELUDE = models.CharField(max_length=100, null=True)
    BASE_CASE_NAV_IMPACT_EVNT = models.CharField(max_length=100, null=True)
    OUTLIER_NAV_IMPACT_AED = models.CharField(max_length=100)
    OUTLIER_NAV_IMPACT_ARB = models.CharField(max_length=100)
    OUTLIER_NAV_IMPACT_CAM = models.CharField(max_length=100)
    OUTLIER_NAV_IMPACT_LEV = models.CharField(max_length=100)
    OUTLIER_NAV_IMPACT_LG = models.CharField(max_length=100)
    OUTLIER_NAV_IMPACT_MACO = models.CharField(max_length=100)
    OUTLIER_NAV_IMPACT_TAQ = models.CharField(max_length=100)
    OUTLIER_NAV_IMPACT_WED = models.CharField(max_length=100, null=True)
    OUTLIER_NAV_IMPACT_WIC = models.CharField(max_length=100, null=True)
    OUTLIER_NAV_IMPACT_MALT = models.CharField(max_length=100, null=True)
    OUTLIER_NAV_IMPACT_PRELUDE = models.CharField(max_length=100, null=True)
    OUTLIER_NAV_IMPACT_EVNT = models.CharField(max_length=100, null=True)


class FormulaeBasedDownsides(models.Model):
    id = models.IntegerField(primary_key=True, unique=True)
    TradeGroup = models.CharField(max_length=100, null=True)
    Underlying = models.CharField(max_length=100, null=True)
    DealValue = models.FloatField(null=True)
    TargetAcquirer = models.CharField(max_length=14, null=True)
    Analyst = models.CharField(max_length=20, null=True)
    OriginationDate = models.DateField(null=True)
    LastUpdate = models.DateField(null=True)
    LastPrice = models.FloatField(null=True)
    IsExcluded = models.CharField(max_length=22, default='No')  # Denote by Yes/No
    RiskLimit = models.FloatField(null=True)
    BaseCaseDownsideType = models.CharField(max_length=50, null=True)  # Store the downside type
    BaseCaseReferenceDataPoint = models.CharField(max_length=50, null=True)  # Based on Downside Type
    BaseCaseReferencePrice = models.CharField(max_length=50, null=True)  # Based on Downside Type
    BaseCaseOperation = models.CharField(max_length=5, null=True)  # +,-,*,/
    BaseCaseCustomInput = models.CharField(max_length=50, null=True)
    base_case = models.CharField(max_length=50, null=True)  # Based on Downside Type
    base_case_notes = models.TextField(null=True)
    cix_ticker = models.CharField(max_length=50, null=True)
    OutlierDownsideType = models.CharField(max_length=50, null=True)  # Store the downside type
    OutlierReferenceDataPoint = models.CharField(max_length=50, null=True)  # Based on Downside Type
    OutlierReferencePrice = models.CharField(max_length=50, null=True)  # Based on Downside Type
    OutlierOperation = models.CharField(max_length=5, null=True)  # +,-,*,/
    OutlierCustomInput = models.CharField(max_length=50, null=True)
    outlier = models.CharField(max_length=50, null=True)  # Based on Downside Type
    outlier_notes = models.TextField(null=True)
    deal_currency = models.CharField(max_length=10, null=True, blank=True, default='USD')
    is_cross_border_deal = models.BooleanField(null=False, blank=False, default=False)
    day_one_downside = models.TextField(max_length=50, null=True)
    unaffected_downsides = models.CharField(max_length=50, null=True, blank=True)
    backup_bid = models.FloatField(null=True)
    PM = models.CharField(max_length=40, null=True)
    Projected = models.CharField(max_length=20, null=True)


class HistoricalFormulaeBasedDownsides(models.Model):
    id = models.AutoField(primary_key=True)
    TradeGroup = models.CharField(max_length=100, null=True)
    Underlying = models.CharField(max_length=100, null=True)
    DealValue = models.FloatField(null=True)
    TargetAcquirer = models.CharField(max_length=14, null=True)
    Analyst = models.CharField(max_length=20, null=True)
    OriginationDate = models.DateField(null=True)
    LastUpdate = models.DateField(null=True)
    LastPrice = models.FloatField(null=True)
    IsExcluded = models.CharField(max_length=22, default='No')  # Denote by Yes/No
    RiskLimit = models.FloatField(null=True)
    BaseCaseDownsideType = models.CharField(max_length=50, null=True)  # Store the downside type
    BaseCaseReferenceDataPoint = models.CharField(max_length=50, null=True)  # Based on Downside Type
    BaseCaseReferencePrice = models.CharField(max_length=50, null=True)  # Based on Downside Type
    BaseCaseOperation = models.CharField(max_length=5, null=True)  # +,-,*,/
    BaseCaseCustomInput = models.CharField(max_length=50, null=True)
    base_case = models.CharField(max_length=50, null=True)  # Based on Downside Type
    base_case_notes = models.TextField(null=True)
    cix_ticker = models.CharField(max_length=50, null=True)
    OutlierDownsideType = models.CharField(max_length=50, null=True)  # Store the downside type
    OutlierReferenceDataPoint = models.CharField(max_length=50, null=True)  # Based on Downside Type
    OutlierReferencePrice = models.CharField(max_length=50, null=True)  # Based on Downside Type
    OutlierOperation = models.CharField(max_length=5, null=True)  # +,-,*,/
    OutlierCustomInput = models.CharField(max_length=50, null=True)
    outlier = models.CharField(max_length=50, null=True)  # Based on Downside Type
    outlier_notes = models.TextField(null=True)
    deal_currency = models.CharField(max_length=10, null=True, blank=True, default='USD')
    is_cross_border_deal = models.BooleanField(null=False, blank=False, default=False)
    day_one_downside = models.TextField(max_length=50, null=True)
    Datestamp = models.DateField(null=True)
    unaffected_downsides = models.CharField(max_length=50, null=True, blank=True)
    backup_bid = models.FloatField(null=True)
    PM = models.CharField(max_length=40, null=True)
    Projected = models.CharField(max_length=20, null=True)


class CreditDealsUpsideDownside(models.Model):
    id = models.IntegerField(primary_key=True, unique=True)
    tradegroup = models.CharField(max_length=100, null=True)
    ticker = models.CharField(max_length=100, null=True)
    analyst = models.CharField(max_length=20, null=True)
    origination_date = models.DateField(null=True)
    last_updated = models.DateTimeField(null=True)
    spread_index = models.CharField(max_length=50, null=True)
    deal_value = models.FloatField(null=True, blank=True)
    last_price = models.FloatField(null=True, blank=True)
    is_excluded = models.CharField(max_length=22, default='No')
    risk_limit = models.FloatField(null=True, blank=True)
    downside_type = models.CharField(max_length=50, null=True)
    downside = models.CharField(max_length=50, null=True, blank=True)
    downside_notes = models.TextField(null=True)
    upside_type = models.CharField(max_length=50, null=True)
    upside = models.CharField(max_length=50, null=True, blank=True)
    upside_notes = models.TextField(null=True)
    bloomberg_id = models.CharField(max_length=50, null=True, blank=True)
    last_refreshed = models.DateTimeField(null=True)


class PositionLevelNAVImpacts(models.Model):
    TradeGroup = models.CharField(max_length=300)
    Ticker = models.CharField(max_length=200)
    PM_BASE_CASE = models.FloatField(null=True)
    Outlier = models.FloatField(null=True)
    LastPrice = models.FloatField(null=True)
    BASE_CASE_NAV_IMPACT_AED = models.FloatField(null=True)
    BASE_CASE_NAV_IMPACT_ARB = models.FloatField(null=True)
    BASE_CASE_NAV_IMPACT_CAM = models.FloatField(null=True)
    BASE_CASE_NAV_IMPACT_LEV = models.FloatField(null=True)
    BASE_CASE_NAV_IMPACT_LG = models.FloatField(null=True)
    BASE_CASE_NAV_IMPACT_MACO = models.FloatField(null=True)
    BASE_CASE_NAV_IMPACT_MALT = models.FloatField(null=True)
    BASE_CASE_NAV_IMPACT_TAQ = models.FloatField(null=True)
    BASE_CASE_NAV_IMPACT_PRELUDE = models.FloatField(null=True)
    BASE_CASE_NAV_IMPACT_EVNT = models.FloatField(null=True)
    OUTLIER_NAV_IMPACT_AED = models.FloatField(null=True)
    OUTLIER_NAV_IMPACT_ARB = models.FloatField(null=True)
    OUTLIER_NAV_IMPACT_CAM = models.FloatField(null=True)
    OUTLIER_NAV_IMPACT_LEV = models.FloatField(null=True)
    OUTLIER_NAV_IMPACT_LG = models.FloatField(null=True)
    OUTLIER_NAV_IMPACT_MACO = models.FloatField(null=True)
    OUTLIER_NAV_IMPACT_MALT = models.FloatField(null=True)
    OUTLIER_NAV_IMPACT_TAQ = models.FloatField(null=True)
    OUTLIER_NAV_IMPACT_PRELUDE = models.FloatField(null=True)
    OUTLIER_NAV_IMPACT_EVNT = models.FloatField(null=True)
    CALCULATED_ON = models.DateTimeField(null=True)


class CixTickerPxLastHistory(models.Model):
    underlying = models.CharField(max_length=100)
    cix_ticker = models.CharField(max_length=50)
    last_threshold_price = models.FloatField(null=True, blank=True)
    curr_price = models.FloatField(null=True, blank=True)
    curr_diff = models.FloatField(null=True, blank=True)
    alert_sent = models.BooleanField(default=False)


class DealDownside(models.Model):
    """ Generic downside type for all deals"""
    deal = models.ForeignKey(MA_Deals, on_delete=models.CASCADE)
    underlying = models.CharField(max_length=100)
    downside = models.DecimalField(max_digits=10, decimal_places=3)
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey('content_type', 'object_id')

    def __str__(self):
        return f'{self.deal} |{self.underlying}|{self.downside_type}| Downside: <{self.downside}>'


class HistoricalDealDownside(models.Model):
    deal = models.ForeignKey(MA_Deals, on_delete=models.CASCADE)
    underlying = models.CharField(max_length=100)
    date = models.DateField(db_index=True)
    downside = models.DecimalField(max_digits=10, decimal_places=3, null=True)
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey('content_type', 'object_id')


class EqualWeightedDownside(models.Model):
    peer_source = models.ForeignKey(MaDownsidePeerSource, on_delete=models.CASCADE)
    number_of_peers = models.IntegerField()
    downsides = GenericRelation('risk_reporting.DealDownside', related_query_name='equal_weighted_downside')

class LinearRegressionDownside(models.Model):
    DAILY_PRICES = 'DP'
    DAILY_RETURN = 'DR'
    STANDARDIZED_PRICES = 'SP'
    PE_RATIO = 'PR'

    MODEL_TYPES = [
        (DAILY_PRICES, 'Daily Prices'),
        (DAILY_RETURN, 'Daily Return'),
        (STANDARDIZED_PRICES, 'Standardized Prices'),
        (PE_RATIO, 'PE Ratio')
    ]
    peer_source = models.ForeignKey(MaDownsidePeerSource, on_delete=models.CASCADE)
    model_type = models.CharField(max_length=10, choices=MODEL_TYPES)
    model_params = models.TextField()
    pickled_model = models.BinaryField()
    create_date = models.DateField(auto_now_add=True)
    year_multiple = models.IntegerField()
    is_selected = models.BooleanField(default=False)
    downsides = GenericRelation('risk_reporting.DealDownside', related_query_name='regression_downside')

    def get_params(self):
        model_params = json.loads(self.model_params)
        model_params['params'] = pd.read_json(model_params['params'], typ='series')
        return model_params

    def get_tickers(self):
        model_params = json.loads(self.model_params)
        return model_params['peers'] + [model_params['target_ticker']]

class DownsideNote(models.Model):
    deal_downside = models.ForeignKey(DealDownside, on_delete=models.CASCADE, related_name='notes')
    note = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class AnalystDownside(models.Model):
    """ Downside specific to an analyst, used for tracking downside updates by analysts
        Value is pulled from external source thus no actual calculation is done here
    """
    downsides = GenericRelation('risk_reporting.DealDownside', related_query_name='analyst_downside')
    analyst = models.CharField(max_length=100)
