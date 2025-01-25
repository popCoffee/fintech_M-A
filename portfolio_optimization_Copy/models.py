import datetime

from django.db import models


class EssDealTypeParameters(models.Model):
    deal_type = models.CharField(max_length=100, unique=True)
    long_probability = models.FloatField(null=True)
    long_irr = models.FloatField(null=True)
    long_max_risk = models.FloatField(null=True)
    long_max_size = models.FloatField(null=True)
    short_probability = models.FloatField(null=True)
    short_irr = models.FloatField(null=True)
    short_max_risk = models.FloatField(null=True)
    short_max_size = models.FloatField(null=True)


class NormalizedSizingByRiskAdjProb(models.Model):
    arb_max_risk = models.FloatField()
    win_probability = models.FloatField()
    lose_probability = models.FloatField()
    risk_adj_loss = models.FloatField()


class SoftCatalystNormalizedRiskSizing(models.Model):
    tier = models.CharField(max_length=10)
    win_probability = models.FloatField()
    lose_probability = models.FloatField()
    max_risk = models.FloatField()
    avg_position = models.FloatField()


class EssPotentialLongShorts(models.Model):
    ess_idea_id = models.IntegerField(null=False)
    Date = models.DateField(null=True)
    alpha_ticker = models.CharField(max_length=100)
    price = models.FloatField(null=True)
    pt_up = models.FloatField(null=True)
    pt_wic = models.FloatField(null=True)
    pt_down = models.FloatField(null=True)
    unaffected_date = models.DateField(null=True)
    expected_close = models.DateField(null=True)
    price_target_date = models.DateField(null=True)
    cix_index = models.CharField(max_length=50, null=True)
    category = models.CharField(max_length=100, null=True)
    catalyst = models.CharField(max_length=50, null=True)
    deal_type = models.CharField(max_length=50, null=True)
    catalyst_tier = models.CharField(max_length=10, null=True)
    gics_sector = models.CharField(max_length=100, null=True)
    hedges = models.CharField(max_length=10, null=True)
    lead_analyst = models.CharField(max_length=10, null=True)
    model_up = models.FloatField(null=True)
    model_wic = models.FloatField(null=True)
    model_down = models.FloatField(null=True)
    implied_probability = models.FloatField(null=True)
    return_risk = models.FloatField(null=True)
    gross_irr = models.FloatField(null=True)
    days_to_close = models.FloatField(null=True)
    ann_irr = models.FloatField(null=True)
    adj_ann_irr = models.FloatField(null=True)
    long_prob = models.FloatField(null=True)
    long_irr = models.FloatField(null=True)
    short_prob = models.FloatField(null=True)
    short_irr = models.FloatField(null=True)
    potential_long = models.CharField(max_length=10, null=True)
    potential_short = models.CharField(max_length=10, null=True)


class EssUniverseImpliedProbability(models.Model):
    Date = models.DateField()
    deal_type = models.CharField(max_length=100, null=True)
    implied_probability = models.FloatField(null=True)
    count = models.IntegerField(null=True)  # To count how many names in the avg. implied probability


# --------------------------- Merger Arbitrage Optimization Models -------------------------------------------

class ArbOptimizationUniverse(models.Model):
    date_updated = models.DateField()
    tradegroup = models.CharField(max_length=100)
    sleeve = models.CharField(max_length=50)
    bucket = models.CharField(max_length=50)
    catalyst = models.CharField(max_length=50, null=True)
    catalyst_rating = models.CharField(max_length=5, null=True)
    closing_date = models.DateField(null=True)
    target_ticker = models.CharField(max_length=100, null=True)
    long_short = models.CharField(max_length=50, null=True)
    target_last_price = models.FloatField(null=True)
    deal_value = models.FloatField(null=True)
    pnl_impact = models.FloatField(null=True)
    all_in_spread = models.FloatField(null=True)
    deal_downside = models.FloatField(null=True)
    days_to_close = models.IntegerField(null=True)
    pct_of_sleeve_current = models.FloatField(null=True)
    gross_ror = models.FloatField(null=True)
    ann_ror = models.FloatField(null=True)
    base_case_nav_impact = models.FloatField(null=True)
    risk_pct = models.FloatField(null=True)
    risk_pct_unhedged = models.FloatField(null=True)
    expected_vol = models.FloatField(null=True)
    deal_status = models.CharField(max_length=100, null=True)
    hedge_ror = models.FloatField(null=True, blank=True, default=0)


class HardFloatOptimization(models.Model):
    date_updated = models.DateField()
    tradegroup = models.CharField(max_length=100)
    sleeve = models.CharField(max_length=50)
    catalyst = models.CharField(max_length=50, null=True)
    catalyst_rating = models.CharField(max_length=5, null=True)
    closing_date = models.DateField(null=True)
    target_ticker = models.CharField(max_length=100, null=True)
    target_last_price = models.FloatField(null=True)
    deal_value = models.FloatField(null=True)
    days_to_close = models.IntegerField(null=True)
    gross_ror = models.FloatField(null=True)
    ann_ror = models.FloatField(null=True)
    risk_pct = models.FloatField(null=True)
    risk_pct_unhedged = models.FloatField(null=True)
    expected_vol = models.FloatField(null=True)
    total_qty = models.FloatField(null=True)
    total_qty_1x = models.FloatField(null=True)
    total_qty_2x = models.FloatField(null=True)
    eqy_float = models.FloatField(null=True)
    current_pct_of_float = models.FloatField(null=True)
    firm_pct_float_mstrat_1x = models.FloatField(null=True)
    firm_pct_float_mstrat_2x = models.FloatField(null=True)
    aed_aum_mult = models.FloatField(null=True)
    taq_aum_mult = models.FloatField(null=True)
    notes = models.TextField(null=True)
    all_in_spread = models.FloatField(null=True)
    aed_risk_mult = models.FloatField(null=True)
    taq_risk_mult = models.FloatField(null=True)
    arb_outlier_risk = models.FloatField(null=True)
    aed_outlier_risk = models.FloatField(null=True)
    taq_outlier_risk = models.FloatField(null=True)
    arb_pct_of_aum = models.FloatField(null=True)
    aed_pct_of_aum = models.FloatField(null=True)
    taq_pct_of_aum = models.FloatField(null=True)
    rebal_multiples = models.FloatField(null=True)
    rebal_target = models.FloatField(null=True)
    rebal_multiples_taq = models.FloatField(null=True)
    rebal_target_taq = models.FloatField(null=True)
    is_excluded = models.BooleanField(default=False)
    weighted_gross_nav_potential = models.FloatField(null=True)
    weighted_nav_cumsum = models.FloatField(null=True)
    non_excluded_pct_aum_cumsum = models.FloatField(null=True)
    curr_rtn_wt_duration = models.FloatField(null=True)
    curr_rwd_ror = models.FloatField(null=True)
    weighted_gross_nav_potential_taq = models.FloatField(null=True)
    weighted_nav_cumsum_taq = models.FloatField(null=True)
    non_excluded_pct_aum_cumsum_taq = models.FloatField(null=True)
    curr_rtn_wt_duration_taq = models.FloatField(null=True)
    curr_rwd_ror_taq = models.FloatField(null=True)
    deal_status = models.CharField(max_length=100, null=True)
    curr_cwd_ror = models.FloatField(null=True)
    curr_cwd_ror_taq = models.FloatField(null=True)
    aed_weighted_ror = models.FloatField(null=True)
    taq_weighted_ror = models.FloatField(null=True)


class HardOptimizationSummary(models.Model):
    date_updated = models.DateField()
    average_optimized_rors = models.FloatField(null=True)
    weighted_arb_rors = models.FloatField(null=True)
    weighted_aed_ror = models.FloatField(null=True)
    weighted_taq_ror = models.FloatField(null=True)
    arb_number_of_deals = models.IntegerField(null=True)
    arb_pct_invested = models.FloatField(null=True)
    aed_number_of_deals = models.IntegerField(null=True)
    aed_hard_pct_invested = models.FloatField(null=True)
    aed_fund_pct_invested = models.FloatField(null=True)
    aed_currently_invested = models.FloatField(null=True)
    taq_number_of_deals = models.IntegerField(null=True)
    taq_hard_pct_invested = models.FloatField(null=True)
    taq_fund_pct_invested = models.FloatField(null=True)
    taq_currently_invested = models.FloatField(null=True)


class CreditHardFloatOptimization(models.Model):
    date_updated = models.DateField()
    tradegroup = models.CharField(max_length=100)
    sleeve = models.CharField(max_length=50)
    catalyst = models.CharField(max_length=50, null=True)
    catalyst_rating = models.CharField(max_length=5, null=True)
    target_last_price = models.FloatField(null=True, blank=True, default=0)
    px_ask_price = models.FloatField(null=True, blank=True, default=0)
    deal_upside = models.FloatField(null=True, blank=True, default=0)
    coupon = models.FloatField(null=True, blank=True, default=0)
    closing_date = models.DateField(null=True)
    days_to_close = models.IntegerField(null=True)
    gross_ror = models.FloatField(null=True, blank=True, default=0)
    ann_ror = models.FloatField(null=True, blank=True, default=0)
    risk_pct_unhedged = models.FloatField(null=True, blank=True, default=0)
    nav_impact = models.FloatField(null=True, blank=True, default=0)
    pct_of_sleeve_current = models.FloatField(null=True, blank=True, default=0)
    m_strat_pct_aum = models.FloatField(null=True, blank=True, default=0)
    rebal_multiples = models.FloatField(null=True, blank=True)
    rebal_target = models.FloatField(null=True, blank=True)
    weighted_gross_nav_potential = models.FloatField(null=True, blank=True, default=0)
    non_excluded_pct_aum = models.FloatField(null=True, blank=True, default=0)
    curr_rtn_wt_duration = models.FloatField(null=True, blank=True, default=0)
    curr_rwd_ror = models.FloatField(null=True, blank=True, default=0)
    curr_cwd_ror = models.FloatField(null=True, blank=True, default=0)
    mstrat_weighted_ror = models.FloatField(null=True, blank=True, default=0)
    target_ticker = models.CharField(max_length=100, null=True, blank=True)
    expected_vol = models.FloatField(null=True)
    notes = models.TextField(null=True, blank=True, default='')
    is_excluded = models.BooleanField(default=False)
    deal_status = models.CharField(max_length=100, null=True, blank=True, default='ACTIVE')
    hedge_ror = models.FloatField(null=True, blank=True, default=0)
    taco_pct_risk = models.FloatField(null=True, blank=True, default=0)
    arb_pct_risk = models.FloatField(null=True, blank=True, default=0)
    arb_pct_aum = models.FloatField(null=True, blank=True, default=0)


class PnlPotentialDate(models.Model):
    start_date = models.DateField(null=True)
    end_date = models.DateField(null=True)
    date_name = models.CharField(max_length=50, null=True, blank=True)
    sleeve = models.CharField(null=True, blank=True, max_length=100)


class PnlPotentialExclusions(models.Model):
    deal_name = models.CharField(max_length=100, null=True, blank=True)
    sleeve = models.CharField(null=True, blank=True, max_length=100)


class PnlPotentialScenarios(models.Model):
    scenario_name = models.CharField(max_length=50, null=True, blank=True)
    date_deal_name = models.CharField(max_length=50, null=True, blank=True)
    scenario_value = models.FloatField(null=True, blank=True, default=0)
    sleeve = models.CharField(null=True, blank=True, max_length=100)


class PnlPotentialIncremental(models.Model):
    incremental_name = models.CharField(max_length=50, null=True, blank=True)
    incremental_value = models.FloatField(null=True, blank=True, default=0)
    sleeve = models.CharField(null=True, blank=True, max_length=100)


class PnlPotentialDateHistory(models.Model):
    date_updated = models.DateField(default=datetime.date.today)
    start_date = models.DateField(null=True)
    end_date = models.DateField(null=True)
    date_name = models.CharField(max_length=50, null=True, blank=True)
    sleeve = models.CharField(null=True, blank=True, max_length=100)


class PnlPotentialExclusionsHistory(models.Model):
    date_updated = models.DateField(default=datetime.date.today)
    deal_name = models.CharField(max_length=100, null=True, blank=True)
    sleeve = models.CharField(null=True, blank=True, max_length=100)


class PnlPotentialScenariosHistory(models.Model):
    date_updated = models.DateField(default=datetime.date.today)
    scenario_name = models.CharField(max_length=50, null=True, blank=True)
    date_deal_name = models.CharField(max_length=50, null=True, blank=True)
    scenario_value = models.FloatField(null=True, blank=True, default=0)
    sleeve = models.CharField(null=True, blank=True, max_length=100)


class PnlPotentialIncrementalHistory(models.Model):
    date_updated = models.DateField(default=datetime.date.today)
    incremental_name = models.CharField(max_length=50, null=True, blank=True)
    incremental_value = models.FloatField(null=True, blank=True, default=0)
    sleeve = models.CharField(null=True, blank=True, max_length=100)


class PnlPotentialDailySummary(models.Model):
    date_updated = models.DateField(default=datetime.date.today)
    sleeve = models.CharField(max_length=50, null=True)
    scenario_name = models.CharField(max_length=50, null=True, blank=True)
    cut_name = models.CharField(max_length=50, null=True, blank=True)
    value = models.FloatField(null=True)


class PnlPotentialESSConstituents(models.Model):
    aed_nav = models.FloatField(null=True)
    tradegroup = models.CharField(null=True, max_length=100)
    current_mkt_val_pct = models.FloatField(null=True)
    customized_mkt_val_pct = models.FloatField(null=True, blank=True, default=0)
    is_customized = models.BooleanField(default=False)
    alpha_ticker = models.CharField(max_length=100, null=True)
    pt_up = models.FloatField(null=True)
    pt_wic = models.FloatField(null=True)
    pt_down = models.FloatField(null=True)
    model_up = models.FloatField(null=True)
    model_wic = models.FloatField(null=True)
    model_down = models.FloatField(null=True)
    up_probability = models.FloatField(default=100)
    down_probability = models.FloatField(default=100)
    upside_field = models.CharField(default='model_up', max_length=50)
    downside_field = models.CharField(default='model_down', max_length=50)
    upside_value = models.FloatField(default=0)
    downside_value = models.FloatField(default=0)
    scenario_name = models.CharField(max_length=50, null=True, blank=True)
    scenario_type = models.CharField(max_length=50, null=True, blank=True)
    px_last = models.FloatField(null=True)
    pnl_potential_100 = models.FloatField(null=True)
    pnl_potential_50 = models.FloatField(null=True)
    pnl_potential_0 = models.FloatField(null=True)


class PnlPotentialESSConstituentsHistory(models.Model):
    date_updated = models.DateField(default=datetime.date.today)
    aed_nav = models.FloatField(null=True)
    tradegroup = models.CharField(null=True, max_length=100)
    current_mkt_val_pct = models.FloatField(null=True)
    customized_mkt_val_pct = models.FloatField(null=True, blank=True, default=0)
    is_customized = models.BooleanField(default=False)
    alpha_ticker = models.CharField(max_length=100, null=True)
    pt_up = models.FloatField(null=True)
    pt_wic = models.FloatField(null=True)
    pt_down = models.FloatField(null=True)
    model_up = models.FloatField(null=True)
    model_wic = models.FloatField(null=True)
    model_down = models.FloatField(null=True)
    up_probability = models.FloatField(default=100)
    down_probability = models.FloatField(default=100)
    upside_field = models.CharField(default='model_up', max_length=50)
    downside_field = models.CharField(default='model_down', max_length=50)
    upside_value = models.FloatField(default=0)
    downside_value = models.FloatField(default=0)
    scenario_name = models.CharField(max_length=50, null=True, blank=True)
    scenario_type = models.CharField(max_length=50, null=True, blank=True)
    px_last = models.FloatField(null=True)
    pnl_potential_100 = models.FloatField(null=True)
    pnl_potential_50 = models.FloatField(null=True)
    pnl_potential_0 = models.FloatField(null=True)


class ArbCreditPnLPotentialDrilldown(models.Model):
    aed_nav = models.FloatField() # updates daily
    tradegroup = models.CharField(max_length=100, null=True)
    sleeve = models.CharField(max_length=100, null=True)
    current_mkt_val_pct = models.FloatField(null=True)
    customized_mkt_val_pct = models.FloatField(null=True)
    bucket = models.CharField(max_length=100, null=True)
    catalyst = models.CharField(max_length=20, null=True)
    catalyst_rating = models.CharField(max_length=20, null=True)
    closing_date = models.DateField(null=True)
    long_short = models.CharField(max_length=20, null=True)
    target_last_price = models.FloatField(null=True)
    deal_value = models.FloatField(null=True)
    all_in_spread = models.FloatField(null=True)
    deal_downside = models.FloatField(null=True)
    days_to_close = models.IntegerField(null=True)
    pct_of_sleeve_current = models.FloatField(null=True)
    gross_ror = models.FloatField(null=True)
    pnl_potential = models.FloatField(null=True)
    is_customized = models.BooleanField(default=False)
    implied_probability = models.FloatField(default=0, null=True, blank=True)


class ArbCreditPnLPotentialDrilldownHistory(models.Model):
    date_updated = models.DateField(default=datetime.date.today)
    aed_nav = models.FloatField()
    tradegroup = models.CharField(max_length=100, null=True)
    sleeve = models.CharField(max_length=100, null=True)
    current_mkt_val_pct = models.FloatField(null=True)
    customized_mkt_val_pct = models.FloatField(null=True)
    bucket = models.CharField(max_length=100, null=True)
    catalyst = models.CharField(max_length=20, null=True)
    catalyst_rating = models.CharField(max_length=20, null=True)
    closing_date = models.DateField(null=True)
    long_short = models.CharField(max_length=20, null=True)
    target_last_price = models.FloatField(null=True)
    deal_value = models.FloatField(null=True)
    all_in_spread = models.FloatField(null=True)
    deal_downside = models.FloatField(null=True)
    days_to_close = models.IntegerField(null=True)
    pct_of_sleeve_current = models.FloatField(null=True)
    gross_ror = models.FloatField(null=True)
    pnl_potential = models.FloatField(null=True)
    is_customized = models.BooleanField(default=False)
    implied_probability = models.FloatField(default=0, null=True, blank=True)


class PnlPotentialOtherValues(models.Model):
    date_updated = models.DateTimeField()
    field_name = models.CharField(max_length=50, default='N/A', null=True, blank=True)
    field_value = models.FloatField(null=True, blank=True)
