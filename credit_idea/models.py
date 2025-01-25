from django.db import models


DEAL_BUCKET = (
    ('Catalyst-Driven Credit', 'Catalyst-Driven Credit'),
    ('Merger Related Credit', 'Merger Related Credit'),
    ('Relative Value Credit', 'Relative Value Credit'),
    ('Distressed', 'Distressed'),
    ('Yield to Call', 'Yield to Call'),
)

DEAL_STRATEGY_TYPE = (
    ('Refinancing', 'Refinancing'),
    ('Speculated M&A', 'Speculated M&A'),
    ('Merger Arbitrage', 'Merger Arbitrage'),
    ('Relative Value', 'Relative Value'),
    ('Definitive M&A', 'Definitive M&A'),
    ('Special Situation', 'Special Situation'),
    ('Levering', 'Levering'),
    ('Deep Value', 'Deep Value'),
    ('De-Levering', 'De-Levering'),
    ('Spin-off', 'Spin-off'),
)

DEAL_CATEGORY = (
    ('Actionable', 'Actionable'),
    ('Not Actionable', 'Not Actionable'),
    ('Bank Debt', 'Bank Debt'),
    ('Convertible Only', 'Convertible Only'),
    ('No Credit', 'No Credit'),
    ('Archive', 'Archive'),
)

CATALYST = (
    ('Hard', 'Hard'),
    ('Soft', 'Soft'),
)

CATALYST_TIER = (
    ('1', '1'),
    ('2', '2'),
    ('3', '3'),
)


class CreditIdea(models.Model):
    """
    Models for Credit Idea Database
    """
    id = models.AutoField(primary_key=True)
    arb_tradegroup = models.CharField(max_length=100, null=True)
    other_tradegroup = models.CharField(max_length=100, null=True, blank=True, default="")
    analyst = models.CharField(max_length=100, null=True, blank=True)
    deal_bucket = models.CharField(null=True, blank=True, max_length=100, choices=DEAL_BUCKET)
    deal_strategy_type = models.CharField(null=True, blank=True, max_length=100, choices=DEAL_STRATEGY_TYPE)
    catalyst = models.CharField(null=True, blank=True, max_length=100, choices=CATALYST)
    catalyst_tier = models.CharField(null=True, blank=True, max_length=10, choices=CATALYST_TIER)
    target_sec_cusip = models.CharField(null=True, blank=True, max_length=50)
    coupon = models.CharField(null=True, blank=True, max_length=100)
    hedge_sec_cusip = models.CharField(null=True, blank=True, max_length=50)
    estimated_closing_date = models.DateField(null=True, blank=True)
    upside_price = models.FloatField(null=True, blank=True)
    downside_price = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    comments = models.TextField(null=True, blank=True)
    deal_category = models.CharField(null=True, blank=True, max_length=100, choices=DEAL_CATEGORY)


class CreditIdeaDetails(models.Model):
    credit_idea = models.OneToOneField(CreditIdea, on_delete=models.CASCADE, primary_key=True)
    nav_pct_impact = models.FloatField(null=True, blank=True)
    topping_big_upside = models.FloatField(null=True, blank=True)
    base_case_downside = models.FloatField(null=True, blank=True)
    base_case_downside_type = models.CharField(null=True, blank=True, max_length=50)
    outlier_downside = models.FloatField(null=True, blank=True)
    outlier_downside_type = models.CharField(null=True, blank=True, max_length=50)
    target_ticker = models.CharField(null=True, blank=True, max_length=20)
    acq_ticker = models.CharField(null=True, blank=True, max_length=20)
    cash_consideration = models.FloatField(null=True, blank=True)
    share_consideration = models.FloatField(null=True, blank=True)
    deal_value = models.FloatField(null=True, blank=True)
    target_dividend = models.FloatField(null=True, blank=True)
    acq_dividend = models.FloatField(null=True, blank=True)
    fund_assets = models.FloatField(null=True, blank=True)
    float_so = models.FloatField(null=True, blank=True)
    acq_pb_rate = models.FloatField(null=True, blank=True)
    target_pb_rate = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class CreditIdeaComments(models.Model):
    credit_idea = models.OneToOneField(CreditIdea, on_delete=models.CASCADE, primary_key=True)
    summary_comments = models.TextField(null=True, blank=True, default='')
    press_release_comments = models.TextField(null=True, blank=True, default='')
    proxy_statement_comments = models.TextField(null=True, blank=True, default='')
    change_of_control_comments = models.TextField(null=True, blank=True, default='')
    restricted_payments_comments = models.TextField(null=True, blank=True, default='')
    liens_indebtedness_comments = models.TextField(null=True, blank=True, default='')
    other_comments = models.TextField(null=True, blank=True, default='')


class CreditIdeaCreditDetails(models.Model):
    credit_idea = models.OneToOneField(CreditIdea, on_delete=models.CASCADE, primary_key=True)
    bond_ticker = models.CharField(null=True, blank=True, max_length=50)
    face_value_of_bonds = models.FloatField(null=True, blank=True)
    bbg_security_name = models.CharField(null=True, blank=True, max_length=50)
    bbg_interest_rate = models.FloatField(null=True, blank=True)
    bbg_issue_size = models.FloatField(null=True, blank=True)
    bond_est_purchase_price = models.FloatField(null=True, blank=True)
    bbg_bid_price = models.FloatField(null=True, blank=True)
    bbg_ask_price = models.FloatField(null=True, blank=True)
    bbg_last_price = models.FloatField(null=True, blank=True)
    base_break_price = models.FloatField(null=True, blank=True)
    conservative_break_price = models.FloatField(null=True, blank=True)
    call_price = models.FloatField(null=True, blank=True)
    make_whole_price = models.FloatField(null=True, blank=True)
    equity_claw_percent = models.FloatField(null=True, blank=True)
    equity_claw_value = models.FloatField(null=True, blank=True)
    blend = models.FloatField(null=True, blank=True)
    change_of_control = models.FloatField(null=True, blank=True)
    acq_credit = models.FloatField(null=True, blank=True)
    other_acq_credit = models.FloatField(null=True, blank=True)
    proposed_ratio = models.FloatField(null=True, blank=True)
    break_spread = models.FloatField(null=True, blank=True)
    bbg_est_daily_vol = models.FloatField(null=True, blank=True)
    bbg_actual_thirty_day = models.FloatField(null=True, blank=True)
    credit_team_view = models.IntegerField(null=True, blank=True, default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class CreditIdeaScenario(models.Model):
    credit_idea = models.ForeignKey(CreditIdea, on_delete=models.CASCADE)
    scenario = models.CharField(null=True, blank=True, max_length=50)
    last_price = models.FloatField(null=True, blank=True, default=0)
    dividends = models.FloatField(null=True, blank=True, default=0)
    rebate = models.FloatField(null=True, blank=True, default=0)
    hedge = models.FloatField(null=True, blank=True, default=0)
    deal_value = models.FloatField(null=True, blank=True, default=0)
    spread = models.FloatField(null=True, blank=True, default=0)
    gross_pct = models.FloatField(null=True, blank=True, default=0)
    annual_pct = models.FloatField(null=True, blank=True, default=0)
    estimated_closing_date = models.DateField(null=True, blank=True)
    days_to_close = models.IntegerField(null=True, blank=True, default=0)
    dollars_to_make = models.FloatField(null=True, blank=True, default=0)
    dollars_to_lose = models.FloatField(null=True, blank=True, default=0)
    implied_prob = models.FloatField(null=True, blank=True, default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class CreditIdeaCreditScenario(models.Model):
    credit_idea = models.ForeignKey(CreditIdea, on_delete=models.CASCADE)
    scenario = models.CharField(null=True, blank=True, max_length=50)
    is_hedge = models.BooleanField(null=False, blank=False)
    bond_last_price = models.FloatField(null=True, blank=True, default=0)
    bond_redemption_type = models.TextField(null=True, blank=True, default='Base Break Price')
    bond_redemption = models.FloatField(null=True, blank=True, default=0)
    bond_carry_earned = models.FloatField(null=True, blank=True, default=0)
    bond_rebate = models.FloatField(null=True, blank=True, default=0)
    bond_hedge = models.FloatField(null=True, blank=True, default=0)
    bond_deal_value = models.FloatField(null=True, blank=True, default=0)
    bond_spread = models.FloatField(null=True, blank=True, default=0)
    returns_gross_pct = models.FloatField(null=True, blank=True, default=0)
    returns_annual_pct = models.FloatField(null=True, blank=True, default=0)
    returns_estimated_closing_date = models.DateField(null=True, blank=True)
    returns_days_to_close = models.IntegerField(null=True, blank=True, default=0)
    profits_principal = models.FloatField(null=True, blank=True, default=0)
    profits_carry = models.FloatField(null=True, blank=True, default=0)
    profits_rebate = models.FloatField(null=True, blank=True, default=0)
    profits_hedge = models.FloatField(null=True, blank=True, default=0)
    profits_total = models.FloatField(null=True, blank=True, default=0)
    profits_day_of_break = models.FloatField(null=True, blank=True, default=0)
    is_deal_closed = models.TextField(null=True, blank=True, default='No')
    is_upside = models.BooleanField(null=False, blank=False, default=False)
    is_downside = models.BooleanField(null=False, blank=False, default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class CreditIdeaCreditScenarioComments(models.Model):
    credit_idea = models.ForeignKey(CreditIdea, on_delete=models.CASCADE)
    scenario = models.CharField(null=True, blank=True, max_length=50)
    comments = models.CharField(null=True, blank=True, max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class CreditStaticScreen(models.Model):
    id = models.AutoField(primary_key=True, null=False)

    yas_yld_spread = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)
    yas_oas_sprd = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)
    yas_ispread_to_govt = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)
    yas_asw_spread = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)
    yas_zspread = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)

    maturity = models.DateField(max_length=50, null=True, blank=True)
    time_to_maturity = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    make_whole_call  = models.BooleanField(default=False)
    make_whole_call_spread = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    dtm = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    eff_duration = models.DecimalField(decimal_places=6, max_digits=13, null=True, blank=True)
    dte = models.DecimalField(decimal_places=6, max_digits=13, null=True, blank=True)
    present_value = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)

    return_5d = models.DecimalField(decimal_places=6, max_digits=13, null=True, blank=True)
    return_mtd = models.DecimalField(decimal_places=6, max_digits=13, null=True, blank=True)
    return_ytd = models.DecimalField(decimal_places=6, max_digits=13, null=True, blank=True)
    return_1_yr = models.DecimalField(decimal_places=6, max_digits=13, null=True, blank=True)
    return_3_yr = models.DecimalField(decimal_places=6, max_digits=13, null=True, blank=True)
    return_5_yr = models.DecimalField(decimal_places=6, max_digits=13, null=True, blank=True)
    return_5d_index = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    return_mtd_index = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    return_ytd_index = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    return_1_yr_index = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    return_3_yr_index = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    return_5_yr_index = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    ytm_weighted_avrg = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    ytw_weighted_avrg = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    ytc_weighted_avrg = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    yte = models.DecimalField(decimal_places=6, max_digits=13, null=True, blank=True)
    ytm = models.DecimalField(decimal_places=6, max_digits=13, null=True, blank=True)
    ytw = models.DecimalField(decimal_places=6, max_digits=13, null=True, blank=True)
    ytc = models.DecimalField(decimal_places=6, max_digits=15, null=True, blank=True)

    date_updated = models.DateField(max_length=50, null=True, blank=True)
    manual_edit = models.BooleanField(default=False)
    issue_dt = models.DateField(max_length=50, null=True, blank=True)
    amt_issued = models.DecimalField(decimal_places=6, max_digits=20, null=True, blank=True)
    rtg_moody = models.CharField(max_length=10, null=True, blank=True)
    rtg_sp = models.CharField(max_length=10, null=True, blank=True)
    bb_compste_rating_ig_hy_indctr = models.CharField(max_length=30, null=True, blank=True)
    payment_rank = models.CharField(max_length=40, null=True, blank=True)
    country = models.CharField(max_length=40, null=True, blank=True)
    market_issue = models.CharField(max_length=40, null=True, blank=True)
    collat_typ = models.CharField(max_length=40, null=True, blank=True)
    last_close_trr_5d = models.DecimalField(decimal_places=6, max_digits=15, null=True, blank=True)
    last_close_trr_mtd = models.DecimalField(decimal_places=6, max_digits=15, null=True, blank=True)
    last_close_trr_ytd = models.DecimalField(decimal_places=6, max_digits=15, null=True, blank=True)
    last_close_trr_1yr = models.DecimalField(decimal_places=6, max_digits=15, null=True, blank=True)
    last_close_trr_3yr = models.DecimalField(decimal_places=6, max_digits=15, null=True, blank=True)
    last_close_trr_5yr = models.DecimalField(decimal_places=6, max_digits=15, null=True, blank=True)
    debt_to_equity_fundamentals_tkr = models.CharField(max_length=40, null=True, blank=True)
    market_cap_name = models.CharField(max_length=40, null=True, blank=True)
    industry = models.CharField(max_length=40, null=True, blank=True)
    deal_upside = models.DecimalField(decimal_places=6, max_digits=15, null=True, blank=True)
    deal_downside = models.DecimalField(decimal_places=6, max_digits=15, null=True, blank=True)
    call_schedule_yeild1 = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    call_schedule_price1 = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)
    call_schedule_date1 = models.DateField(max_length=50, null=True, blank=True)
    call_schedule_yeild2 = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    call_schedule_price2 = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)
    call_schedule_date2 = models.DateField(max_length=50, null=True, blank=True)
    call_schedule_yeild3 = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    call_schedule_price3 = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)
    call_schedule_date3 = models.DateField(max_length=50, null=True, blank=True)
    call_schedule_yeild4 = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    call_schedule_price4 = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)
    call_schedule_date4 = models.DateField(max_length=50, null=True, blank=True)
    #
    catalyst_type_wic = models.CharField(max_length=40, null=True, blank=True)
    bucket = models.CharField(max_length=40, null=True, blank=True)
    tradegroup = models.CharField(max_length=80, null=True, blank=True)
    target_ticker = models.CharField(max_length=80, null=True, blank=True)
    isin = models.CharField(max_length=20, null=True, blank=True)
    price = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    closing_date = models.DateField(max_length=50, null=True, blank=True)
    pct_aum = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    yld_ytm_mid = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    yld_cnv_mid = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    ytc1 = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    ytc2 = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    dur_adj_mid = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    dur_adj_mty_mid = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    nxt_call_dt = models.DateField(max_length=50, null=True, blank=True)
    nxt_call_px = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    second_call_dt = models.DateField(max_length=50, null=True, blank=True)
    second_call_px = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    workout_date_mid_to_worst = models.DateField(max_length=50, null=True, blank=True)
    yas_workout_px = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    cpn = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    cpn_freq = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    cpn_typ = models.CharField(max_length=20, null=True, blank=True)
    par = models.DecimalField(decimal_places=6, max_digits=10, null=True, blank=True)
    callable = models.BooleanField(default=False)
    called = models.BooleanField(default=False)
    dv01 = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)
    cr01 = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)
    fund = models.CharField(max_length=16, null=True, blank=True)
    time_to_event = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)
    bullet = models.CharField(max_length=20, null=True, blank=True)
    makewhole_end = models.DateField(max_length=50, null=True, blank=True)
    mw_spread = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)
    clawback = models.CharField(max_length=20, null=True, blank=True)
    claw_end = models.DateField(max_length=50, null=True, blank=True)
    claw_price = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)
    equity_claw_pct = models.DecimalField(decimal_places=6, max_digits=12, null=True, blank=True)

