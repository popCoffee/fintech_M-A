from django.db import models


class EtfRecRecords(models.Model):
    """
    This model stores the daily computed weight tracking for index, basket and eze weights.
    Updated daily through Celery tasks
    """
    date = models.DateField()
    sedol = models.CharField(max_length=20)
    sectype = models.CharField(max_length=20, null=True)
    eze_ticker = models.CharField(max_length=25)
    deal = models.CharField(max_length=25)
    index = models.FloatField(null=True)
    eze = models.FloatField(null=True)
    basket = models.FloatField(null=True)
    index_eze = models.FloatField(null=True)
    basket_eze = models.FloatField(null=True)
    index_basket = models.FloatField(null=True)
    weight_tracked = models.FloatField(null=True)
    pct_tracked = models.FloatField(null=True)
    additional_etf_exposure = models.FloatField(null=True, default=0)
    notes = models.TextField(null=True)


class EtfRecSummary(models.Model):
    """
    This model stores the long/short summary for the Index Weight, Weight Tracked and % Tracked
    """
    date = models.DateField()
    index_weights_long = models.FloatField(null=True)
    index_weights_short = models.FloatField(null=True)
    index_weights_forwards = models.FloatField(null=True)
    index_weights_gross = models.FloatField(null=True)
    weight_tracked_long = models.FloatField(null=True)
    weight_tracked_short = models.FloatField(null=True)
    weight_tracked_forwards = models.FloatField(null=True)
    weight_tracked_gross = models.FloatField(null=True)
    pct_tracked_long = models.FloatField(null=True)
    pct_tracked_short = models.FloatField(null=True)
    pct_tracked_forwards = models.FloatField(null=True)
    pct_tracked_gross = models.FloatField(null=True)
    additional_etf_exposure_long = models.FloatField(null=True)
    additional_etf_exposure_short = models.FloatField(null=True)
    additional_etf_exposure_forwards = models.FloatField(null=True)
    additional_etf_exposure_gross = models.FloatField(null=True)


class CustomUserInputs(models.Model):
    """ Model to store custom user inputs for serving the P&L and BSKT tabs
    """
    date_updated = models.DateField()
    collateral_buffer = models.FloatField(default=0)
    net_td_creations = models.IntegerField(default=0)
    net_td_redemptions = models.IntegerField(default=0)
    tax = models.FloatField(default=0)
    fees = models.FloatField(default=0)


# Tables to Store the Monitors


class ETFMonitors(models.Model):
    """ Store the JSON equivalents for all the ETF monitors.
        Table will be updated every 15 mins with ability to run live as well...
    """
    updated_on = models.DateTimeField()
    spread_monitor = models.TextField(null=True)
    bid_ask_monitor = models.TextField(null=True)
    bid_ask_spread_monitor = models.TextField(null=True)
    nav_monitor = models.TextField(null=True)
    basket_valuation_monitor = models.TextField(null=True)
    unit_activity_monitor = models.TextField(null=True)
    spread_estimate_monitor = models.TextField(null=True)


class ETFLivePnL(models.Model):
    """ Model to store the individual stock returns for the ETf. Updates every 15 mins"""
    updated_on = models.DateTimeField()
    ticker = models.CharField(max_length=25, null=True)
    cur = models.CharField(max_length=5, null=True)
    sedol = models.CharField(max_length=12, null=True)
    basket_unit_size = models.FloatField(null=True)
    base_mv = models.FloatField(null=True)
    local_price = models.FloatField(null=True)
    base_price = models.FloatField(null=True)
    weight = models.FloatField(null=True)
    cil = models.CharField(max_length=2, null=True)
    tradegroup = models.CharField(max_length=50, null=True)
    eze_ticker = models.CharField(max_length=25, null=True)
    eze_shares = models.FloatField(null=True)
    pct_nav_cu = models.FloatField(null=True)
    live_price = models.FloatField(null=True)
    px_change = models.FloatField(null=True)
    stock_return = models.FloatField(null=True)
    deal_return = models.FloatField(null=True)
    one_day_return = models.FloatField(null=True)
    five_day_return = models.FloatField(null=True)
    ytd_return = models.FloatField(null=True)
    live_ytd_return = models.FloatField(null=True)
    bid_market_value = models.FloatField(null=True)
    ask_market_value = models.FloatField(null=True)
    usd_live_price = models.FloatField(null=True)
    cil_mv = models.FloatField(null=True)
    security_bid = models.FloatField(null=True)
    security_ask = models.FloatField(null=True)
    live_fx_price = models.FloatField(null=True)


class DailyIndexPnL(models.Model):
    """ Model to store the daily index PNL for the ETF. Updates every day. used to analyze tracking errors"""
    date = models.DateTimeField()
    updated_on = models.DateTimeField()
    ticker = models.CharField(max_length=25, null=True)
    cur = models.CharField(max_length=5, null=True)
    sedol = models.CharField(max_length=12, null=True)
    tradegroup = models.CharField(max_length=12, null=True)
    pnl = models.FloatField(null=True)
    dividend_pnl = models.FloatField(null=True)
    # tradar_pnl = models.FloatField(null=True)
    closing_price = models.FloatField(null=True)
    opening_price = models.FloatField(null=True)
    closing_fx = models.FloatField(null=True)
    percentage_weighting = models.FloatField(null=True)
    fraction_of_shares= models.FloatField(null=True)
    closing_hedged_weight = models.FloatField(null=True)
    sectype = models.CharField(max_length=25, null=True)


class MarketOnClose(models.Model):
    """ Model to store the Market on Close list of trades """
    security = models.CharField(max_length=25, null=True)
    side = models.CharField(max_length=20, null=True)
    prt = models.CharField(max_length=25, null=True)
    amount = models.FloatField(null=True)
    trader = models.CharField(max_length=25, null=True)
    manager = models.CharField(max_length=25, null=True)
    broker = models.CharField(max_length=25, null=True)
    strategy1 = models.CharField(max_length=50, null=True)
    tradedate = models.DateTimeField()

