import decimal
import math

import bbgclient
import pandas as pd
from celery import shared_task
from datetime import date, datetime, timedelta
from django.conf import settings
from django.db import transaction, connection
from scipy.optimize import fsolve
from dateutil.relativedelta import relativedelta

import dbutils
import holiday_utils
from credit_idea.models import (CreditIdea, CreditIdeaComments, CreditIdeaCreditDetails, CreditIdeaCreditScenario,
                                CreditIdeaCreditScenarioComments, CreditIdeaDetails, CreditIdeaScenario,
                                CreditStaticScreen)
from credit_idea.utils import (append_equity_to_ticker, calculate_number_of_days, convert_to_float_else_zero,
    convert_to_str_decimal)
from credit_idea.views import get_dict_from_flat_file
from django_slack import slack_message
from slack_utils import get_channel_name
import QuantLib as ql



BBG_FED_FUND_INDEX = 'FEDL01 INDEX'


@shared_task
def refresh_credit_idea_deals():
    """ Refreshes the credit idea deals with data from the flat file every 24 hours """

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    credit_idea_list = CreditIdea.objects.all()
    credit_idea_details_list = CreditIdeaDetails.objects.all()
    target_ticker_list = set()
    acq_ticker_list = set()
    target_ticker_dict = {}
    acq_ticker_dict = {}
    cash_consideration_dict = {}
    share_consideration_dict = {}
    target_dividend_dict = {}
    acq_dividend_dict = {}
    acq_pb_rate_dict = {}
    closing_date_dict = {}
    fx_local_to_base_dict = {}
    nav_pct_impact_dict = {}
    fund_assets_dict = {}
    base_case_downside_dict = {}
    target_pb_rate_dict = {}
    with transaction.atomic():
        for credit_idea_detail in credit_idea_details_list:
            credit_idea = credit_idea_list.get(id=credit_idea_detail.credit_idea_id)
            credit_idea_id = credit_idea.id
            arb_tradegroup = credit_idea.arb_tradegroup
            arb_target_ticker = credit_idea_detail.target_ticker
            if arb_tradegroup.lower() != 'other':
                flat_file_dict = get_dict_from_flat_file(arb_tradegroup, arb_target_ticker)
                fx_local_to_base = flat_file_dict.get('fx_local_to_base') or 1.00
                fx_local_to_base_dict[credit_idea_id] = fx_local_to_base
                fund_assets = flat_file_dict.get('fund_assets')
                fund_assets_dict[credit_idea_id] = fund_assets
                base_case_downside = convert_to_float_else_zero(flat_file_dict.get('base_case_downside'))
                credit_idea_detail.base_case_downside = base_case_downside
                base_case_downside_dict[credit_idea_id] = base_case_downside
                deal_value = convert_to_float_else_zero(flat_file_dict.get('deal_value'))
                credit_idea_detail.deal_value = deal_value
                credit_idea_detail.topping_big_upside = deal_value
                credit_idea_detail.outlier_downside = convert_to_float_else_zero(flat_file_dict.get('outlier_downside'))
                nav_pct_impact_dict[credit_idea_id] = credit_idea_detail.nav_pct_impact
                target_ticker = append_equity_to_ticker(flat_file_dict.get('target_ticker'))
                if not target_ticker:
                    target_ticker = append_equity_to_ticker(credit_idea_detail.target_ticker)
                credit_idea_detail.target_ticker = target_ticker
                if target_ticker:
                    target_ticker_list.add(target_ticker)
                target_ticker_dict[credit_idea_id] = target_ticker
                acq_ticker = append_equity_to_ticker(flat_file_dict.get('acq_ticker'))
                if not acq_ticker:
                    acq_ticker = append_equity_to_ticker(credit_idea_detail.acq_ticker)
                credit_idea_detail.acq_ticker = acq_ticker
                if acq_ticker:
                    acq_ticker_list.add(acq_ticker)
                acq_ticker_dict[credit_idea_id] = acq_ticker
                cash_consideration = convert_to_float_else_zero(flat_file_dict.get('cash_consideration'))
                credit_idea_detail.cash_consideration = cash_consideration
                cash_consideration_dict[credit_idea_id] = cash_consideration
                share_consideration = convert_to_float_else_zero(flat_file_dict.get('share_consideration'))
                credit_idea_detail.share_consideration = share_consideration
                share_consideration_dict[credit_idea_id] = share_consideration
                target_dividend = convert_to_float_else_zero(flat_file_dict.get('target_dividend'))
                credit_idea_detail.target_dividend = target_dividend
                target_dividend_dict[credit_idea_id] = target_dividend
                acq_dividend = convert_to_float_else_zero(flat_file_dict.get('acq_dividend'))
                credit_idea_detail.acq_dividend = acq_dividend
                acq_dividend_dict[credit_idea_id] = acq_dividend
                credit_idea_detail.fund_assets = flat_file_dict.get('fund_assets')
                acq_pb_rate_dict[credit_idea_id] = credit_idea_detail.acq_pb_rate
                target_pb_rate_dict[credit_idea_id] = credit_idea_detail.target_pb_rate
                credit_idea_detail.save()

                closing_date = flat_file_dict.get('closing_date')
                closing_date_dict[credit_idea_id] = closing_date
                credit_idea.estimated_closing_date = closing_date
                credit_idea.save()
            else:
                nav_pct_impact_dict[credit_idea_id] = credit_idea_detail.nav_pct_impact

    credit_idea_credit_details = CreditIdeaCreditDetails.objects.all()
    credit_idea_credit_details_df = pd.DataFrame.from_records(list(credit_idea_credit_details.values()))
    bond_ticker_list = set(credit_idea_credit_details_df.bond_ticker.dropna().tolist())
    bond_ticker_list = list(bond_ticker_list)
    target_ticker_list = list(target_ticker_list)
    acq_ticker_list = list(acq_ticker_list)
    all_tickers = [BBG_FED_FUND_INDEX] + target_ticker_list + bond_ticker_list + acq_ticker_list
    bbg_fields = ['PX_LAST', 'DVD_SH_LAST', 'SECURITY_NAME', 'COUPON', 'AMT_OUTSTANDING', 'PX_BID', 'PX_ASK']
    api_host = bbgclient.bbgclient.get_next_available_host()
    live_price = bbgclient.bbgclient.get_secid2field(all_tickers, 'tickers', bbg_fields, req_type='refdata',
                                                     api_host=api_host)
    fed_fund_price = live_price.get(BBG_FED_FUND_INDEX)
    if fed_fund_price:
        px_last_value = fed_fund_price.get('PX_LAST')
        fed_funds_last_price = px_last_value[0] if len(px_last_value) > 0 else 0.00
        fed_funds_last_price = convert_to_float_else_zero(fed_funds_last_price)
    else:
        fed_funds_last_price = 0.00

    est_purchase_price_dict = {}
    potential_outcomes_dict = {}
    interest_rate_dict = {}
    face_value_bonds_dict = {}
    proposed_ratio_dict = {}
    base_spread_dict = {}
    base_rebate_dict = {}
    alternate_break_spread_dict = {}
    with transaction.atomic():
        for credit_idea_credit_detail in credit_idea_credit_details:
            credit_idea = credit_idea_list.get(id=credit_idea_credit_detail.credit_idea_id)
            credit_idea_id = credit_idea.id
            temp_outcomes_dict = {}
            arb_tradegroup = credit_idea.arb_tradegroup
            if arb_tradegroup.lower() != 'other':
                bbg_security_name = credit_idea_credit_detail.bbg_security_name
                bond_ticker = credit_idea_credit_detail.bond_ticker
                bond_ticker_price = live_price.get(bond_ticker)
                if bond_ticker_price:
                    if not bbg_security_name:
                        bbg_security_name = bond_ticker_price.get('SECURITY_NAME')
                        bbg_security_name = bbg_security_name[0] if len(bbg_security_name) > 0 else ''
                        bbg_security_name = bbg_security_name if bbg_security_name else ''
                        credit_idea_credit_detail.bbg_security_name = bbg_security_name
                    bbg_interest_rate = bond_ticker_price.get('COUPON')
                    bbg_interest_rate = convert_to_float_else_zero(bbg_interest_rate[0]) if len(bbg_interest_rate) > 0 else 0.00
                    interest_rate_dict[credit_idea_id] = bbg_interest_rate
                    bbg_issue_size = bond_ticker_price.get('AMT_OUTSTANDING')
                    bbg_issue_size = convert_to_float_else_zero(bbg_issue_size[0]) / 1000000 if len(bbg_issue_size) > 0 else 0.00
                    bbg_bid_price = bond_ticker_price.get('PX_BID')
                    bbg_bid_price = convert_to_float_else_zero(bbg_bid_price[0]) if len(bbg_bid_price) > 0 else 0.00
                    bbg_ask_price = bond_ticker_price.get('PX_ASK')
                    bbg_ask_price = convert_to_float_else_zero(bbg_ask_price[0]) if len(bbg_ask_price) > 0 else 0.00
                    bbg_last_price = bond_ticker_price.get('PX_LAST')
                    bbg_last_price = convert_to_float_else_zero(bbg_last_price[0]) if len(bbg_last_price) > 0 else 0.00
                    credit_idea_credit_detail.bbg_interest_rate = bbg_interest_rate
                    credit_idea_credit_detail.bbg_issue_size = bbg_issue_size
                    credit_idea_credit_detail.bbg_bid_price = bbg_bid_price
                    credit_idea_credit_detail.bbg_ask_price = bbg_ask_price
                    credit_idea_credit_detail.bbg_last_price = bbg_last_price
                    credit_idea_credit_detail.bond_est_purchase_price = bbg_last_price
                    est_purchase_price_dict[credit_idea_id] = bbg_last_price
                    credit_idea_credit_detail.save()
                else:
                    interest_rate_dict[credit_idea_id] = 0.00
                    est_purchase_price_dict[credit_idea_id] = 0.00
                temp_outcomes_dict['base_break_price'] = convert_to_float_else_zero(credit_idea_credit_detail.base_break_price)
                temp_outcomes_dict['conservative_break_price'] = convert_to_float_else_zero(credit_idea_credit_detail.conservative_break_price)
                temp_outcomes_dict['call_price'] = convert_to_float_else_zero(credit_idea_credit_detail.call_price)
                temp_outcomes_dict['change_of_control'] = convert_to_float_else_zero(credit_idea_credit_detail.change_of_control)
                temp_outcomes_dict['make_whole_price'] = convert_to_float_else_zero(credit_idea_credit_detail.make_whole_price)
                temp_outcomes_dict['blend'] = convert_to_float_else_zero(credit_idea_credit_detail.blend)
                temp_outcomes_dict['other_1'] = convert_to_float_else_zero(credit_idea_credit_detail.acq_credit)
                temp_outcomes_dict['other_2'] = convert_to_float_else_zero(credit_idea_credit_detail.other_acq_credit)
                potential_outcomes_dict[credit_idea_id] = temp_outcomes_dict
                face_value_bonds_dict[credit_idea_id] = convert_to_float_else_zero(credit_idea_credit_detail.face_value_of_bonds)
                proposed_ratio_dict[credit_idea_id] = convert_to_float_else_zero(credit_idea_credit_detail.proposed_ratio)
                alternate_break_spread_dict[credit_idea_id] = credit_idea_credit_detail.break_spread

    credit_idea_scenarios = CreditIdeaScenario.objects.all()
    with transaction.atomic():
        for credit_idea in credit_idea_list:
            credit_idea_id = credit_idea.id
            arb_tradegroup = credit_idea.arb_tradegroup
            if arb_tradegroup.lower() != 'other':
                credit_idea_scenarios_filter = credit_idea_scenarios.filter(credit_idea_id=credit_idea.id)
                bbg_target_ticker = append_equity_to_ticker(target_ticker_dict.get(credit_idea_id))
                target_ticker_price = live_price.get(bbg_target_ticker)
                target_live_price = 0.00
                if target_ticker_price:
                    px_last_value = target_ticker_price.get('PX_LAST')
                    target_live_price = px_last_value[0] if len(px_last_value) > 0 else 0.00
                    target_live_price = convert_to_float_else_zero(target_live_price)
                    if 'ln' in bbg_target_ticker.lower().split(' '):
                        target_live_price = target_live_price * 0.01
                target_live_price = round(target_live_price, 2)

                bbg_acq_ticker = append_equity_to_ticker(acq_ticker_dict.get(credit_idea_id))
                acq_ticker_price = live_price.get(bbg_acq_ticker)
                acq_last_price = 0.00
                if acq_ticker_price:
                    px_last_value = acq_ticker_price.get('PX_LAST')
                    acq_last_price = px_last_value[0] if len(px_last_value) > 0 else 0.00
                    acq_last_price = convert_to_float_else_zero(acq_last_price)
                    if 'ln' in bbg_acq_ticker.lower().split(' '):
                        acq_last_price = acq_last_price * 0.01
                acq_last_price = round(acq_last_price, 2)

                for credit_idea_scenario in credit_idea_scenarios_filter:
                    rebate_pct = convert_to_float_else_zero(fed_funds_last_price) - convert_to_float_else_zero(acq_pb_rate_dict.get(credit_idea_id))
                    rebate_pct = round(rebate_pct, 2)
                    estimated_closing_date = credit_idea_scenario.estimated_closing_date
                    scenario = credit_idea_scenario.scenario
                    if scenario and scenario.lower() == 'base date':
                        estimated_closing_date = closing_date_dict.get(credit_idea_id)
                    elif scenario and scenario.lower() == 'earlier date':
                        estimated_closing_date = closing_date_dict.get(credit_idea_id) - timedelta(days=31)
                    elif scenario and scenario.lower() == 'worst date':
                        estimated_closing_date = closing_date_dict.get(credit_idea_id) + timedelta(days=31)
                    target_dividend = convert_to_float_else_zero(target_dividend_dict.get(credit_idea_id))
                    acq_dividend = convert_to_float_else_zero(acq_dividend_dict.get(credit_idea_id))
                    share_consideration = convert_to_float_else_zero(share_consideration_dict.get(credit_idea_id))
                    dividends = target_dividend - (acq_dividend * share_consideration)
                    hedge = convert_to_float_else_zero(credit_idea_scenario.hedge)
                    days_to_close = calculate_number_of_days(estimated_closing_date)
                    cash_consideration = round(convert_to_float_else_zero(cash_consideration_dict.get(credit_idea_id)), 2)
                    rebate = share_consideration * convert_to_float_else_zero(acq_last_price) * rebate_pct * 0.01 * \
                             days_to_close / 365
                    rebate = round(rebate, 2)
                    current_deal_value = round(cash_consideration + share_consideration * acq_last_price, 2)
                    deal_value = round(current_deal_value + dividends + rebate - hedge, 2)
                    spread = round(deal_value - target_live_price, 2)
                    if scenario and scenario.lower() == 'base date':
                        base_spread_dict[credit_idea_id] = spread
                        base_rebate_dict[credit_idea_id] = rebate
                    gross_pct = 0
                    if convert_to_float_else_zero(target_live_price) != 0:
                        gross_pct = (spread / target_live_price) * 100
                    gross_pct = round(gross_pct, 2)
                    annual_pct = 0
                    if convert_to_float_else_zero(days_to_close) != 0:
                        annual_pct = (365 / days_to_close) * gross_pct
                    annual_pct = round(annual_pct, 2)
                    fx_local_to_base = convert_to_float_else_zero(fx_local_to_base_dict.get(credit_idea_id)) or 1
                    acq_thirty_premium = (1 + 0.3) * acq_last_price
                    base_case_downside = convert_to_float_else_zero(base_case_downside_dict.get(credit_idea_id))
                    topping_spread = ((acq_thirty_premium * share_consideration) + cash_consideration) - base_case_downside
                    gross_spread = deal_value - target_live_price
                    size_shares = 0
                    if (topping_spread - gross_spread) != 0:
                        size_shares = (convert_to_float_else_zero(nav_pct_impact_dict.get(credit_idea_id)) * 0.01 * \
                                      convert_to_float_else_zero(fund_assets_dict.get(credit_idea_id))) / \
                                      (topping_spread - gross_spread)
                        size_shares = round(abs(size_shares / fx_local_to_base))
                    dollars_to_make = round(spread * size_shares * fx_local_to_base)
                    dollars_to_lose = round((gross_spread - topping_spread) * size_shares * fx_local_to_base)
                    implied_prob = 0
                    if topping_spread != 0:
                        implied_prob = (1 - (gross_spread / topping_spread)) * 100
                    implied_prob = round(implied_prob, 2)

                    credit_idea_scenario.last_price = target_live_price
                    credit_idea_scenario.rebate = rebate
                    credit_idea_scenario.deal_value = deal_value
                    credit_idea_scenario.spread = spread
                    credit_idea_scenario.gross_pct = gross_pct
                    credit_idea_scenario.annual_pct = annual_pct
                    credit_idea_scenario.estimated_closing_date = estimated_closing_date
                    credit_idea_scenario.days_to_close = days_to_close
                    credit_idea_scenario.dollars_to_make = dollars_to_make
                    credit_idea_scenario.dollars_to_lose = dollars_to_lose
                    credit_idea_scenario.implied_prob = implied_prob
                    credit_idea_scenario.save()

                scenario_hedges = CreditIdeaCreditScenario.objects.filter(credit_idea_id=credit_idea_id)
                for scenario_hedge in scenario_hedges:
                    scenario = scenario_hedge.scenario
                    is_hedge = scenario_hedge.is_hedge
                    est_purchase_price = convert_to_float_else_zero(est_purchase_price_dict.get(credit_idea_id))
                    face_value_bond = convert_to_float_else_zero(face_value_bonds_dict.get(credit_idea_id))
                    proposed_ratio = convert_to_float_else_zero(proposed_ratio_dict.get(credit_idea_id))
                    target_pb_rate = convert_to_float_else_zero(target_pb_rate_dict.get(credit_idea_id))
                    fx_local_to_base = convert_to_float_else_zero(fx_local_to_base_dict.get(credit_idea_id))
                    bond_redemption_type = scenario_hedge.bond_redemption_type
                    bond_redemption = scenario_hedge.bond_redemption
                    outcomes_dict = potential_outcomes_dict.get(credit_idea_id)
                    interest_rate = interest_rate_dict.get(credit_idea_id)
                    est_closing_date = scenario_hedge.returns_estimated_closing_date
                    base_spread = base_spread_dict.get(credit_idea_id)
                    base_rebate = base_rebate_dict.get(credit_idea_id)
                    alternate_break_spread = convert_to_float_else_zero(alternate_break_spread_dict.get(credit_idea_id))
                    if outcomes_dict:
                        if bond_redemption_type.lower() == 'base break price':
                            bond_redemption = outcomes_dict.get('base_break_price')
                        elif bond_redemption_type.lower() == 'call price':
                            bond_redemption = outcomes_dict.get('call_price')
                        elif bond_redemption_type.lower() == 'conservative break price':
                            bond_redemption = outcomes_dict.get('conservative_break_price')
                        elif bond_redemption_type.lower() == 'change of control':
                            bond_redemption = outcomes_dict.get('change_of_control')
                        elif bond_redemption_type.lower() == 'blend':
                            bond_redemption = outcomes_dict.get('blend')
                        elif bond_redemption_type.lower() == 'make-whole price':
                            bond_redemption = outcomes_dict.get('make_whole_price')
                        elif bond_redemption_type.lower() == 'other (1)':
                            bond_redemption = outcomes_dict.get('other_1')
                        elif bond_redemption_type.lower() == 'other (2)':
                            bond_redemption = outcomes_dict.get('other_2')   
                    bond_redemption = convert_to_float_else_zero(bond_redemption)
                    if scenario and scenario.lower() in ['bonds called (redemption)', 'change of control (coc)', 'no deal (base case)', 'no deal (conservative case)']:
                        est_closing_date = closing_date_dict.get(credit_idea_id)
                    days_to_close = calculate_number_of_days(est_closing_date)
                    carry_earned = round(100 * interest_rate * days_to_close / 365 * 0.01, 3)
                    passive_arb_spend = round(face_value_bond * est_purchase_price * 0.01)
                    hedge_dollar = round(passive_arb_spend * proposed_ratio * 0.01)
                    target_short = 0
                    if target_live_price != 0:
                        target_short = round(hedge_dollar / round(target_live_price * fx_local_to_base, 2))
                    bond_rebate = convert_to_float_else_zero(scenario_hedge.bond_rebate)
                    bond_hedge = convert_to_float_else_zero(scenario_hedge.bond_hedge)
                    is_deal_closed = scenario_hedge.is_deal_closed
                    if is_hedge:
                        less_rebate = -1 * base_rebate * fx_local_to_base
                        less_short_rebate = round(target_live_price * fx_local_to_base, 2) * -1 * (fed_funds_last_price - target_pb_rate) * 0.01 * days_to_close / 365
                        less_short_rebate = round(less_short_rebate, 2)
                        if face_value_bond != 0:
                            bond_rebate = target_short * less_short_rebate / face_value_bond * 100 * -1
                        normal_usd_spread = (target_live_price - base_case_downside) * fx_local_to_base
                        if is_deal_closed and is_deal_closed.lower() == 'no':
                            if face_value_bond != 0:
                                bond_hedge = normal_usd_spread * target_short / face_value_bond * 100
                        elif is_deal_closed and is_deal_closed.lower() == 'yes':
                            if face_value_bond != 0:
                                arb_spread = round((base_spread + less_rebate) * fx_local_to_base, 2)
                                bond_hedge = -1 * target_short * arb_spread / face_value_bond * 100
                        elif is_deal_closed and is_deal_closed.lower() == 'other':
                            if face_value_bond != 0:
                                bond_hedge = (alternate_break_spread - base_spread) * target_short / face_value_bond * 100

                    bond_rebate = round(bond_rebate, 3)
                    bond_hedge = round(bond_hedge, 3)
                    bond_deal_value = round(bond_redemption + carry_earned + bond_rebate + bond_hedge, 3)
                    bond_spread = round(bond_deal_value - est_purchase_price, 3)
                    gross_pct = convert_to_float_else_zero(scenario_hedge.returns_gross_pct)
                    if est_purchase_price != 0:
                        gross_pct = bond_spread / est_purchase_price * 100
                    gross_pct = round(gross_pct, 2)
                    annual_pct = convert_to_float_else_zero(scenario_hedge.returns_annual_pct)
                    if days_to_close != 0:
                        annual_pct = (365 / days_to_close) * gross_pct
                    annual_pct = round(annual_pct, 2)
                    profits_principal = round((bond_redemption - est_purchase_price) * face_value_bond * 0.01)
                    profits_carry = round(carry_earned * face_value_bond * 0.01)
                    profits_rebate = round(bond_rebate * face_value_bond * 0.01)
                    profits_hedge = round(bond_hedge * face_value_bond * 0.01)
                    profits_total = round(profits_principal + profits_carry + profits_rebate + profits_hedge)

                    if is_deal_closed and is_deal_closed.lower() == 'yes':
                        profits_day_of_break = 0
                    else:
                        profits_day_of_break = profits_hedge + profits_principal
                    profits_day_of_break = round(profits_day_of_break)

                    scenario_hedge.bond_last_price = est_purchase_price
                    scenario_hedge.bond_redemption = bond_redemption
                    scenario_hedge.bond_carry_earned = carry_earned
                    scenario_hedge.returns_estimated_closing_date = est_closing_date
                    scenario_hedge.returns_days_to_close = days_to_close
                    scenario_hedge.bond_rebate = bond_rebate
                    scenario_hedge.bond_hedge = bond_hedge
                    scenario_hedge.bond_deal_value = bond_deal_value
                    scenario_hedge.bond_spread = bond_spread
                    scenario_hedge.returns_gross_pct = gross_pct
                    scenario_hedge.returns_annual_pct = annual_pct
                    scenario_hedge.profits_principal = profits_principal
                    scenario_hedge.profits_carry = profits_carry
                    scenario_hedge.profits_rebate = profits_rebate
                    scenario_hedge.profits_hedge = profits_hedge
                    scenario_hedge.profits_total = profits_total
                    scenario_hedge.profits_day_of_break = profits_day_of_break
                    scenario_hedge.save()
    slack_message('generic.slack',
                  {'message': 'Merger Arb Credit updated successfully from Flat File on : ' +
                   datetime.now().strftime('%B %d, %Y %H:%M')},
                  channel=get_channel_name('portal-task-reports'),
                  token=settings.SLACK_TOKEN,
                  name='ESS_IDEA_DB_ERROR_INSPECTOR')
    dbutils.add_task_record()


def yield_formula(settlement_date,maturity_date,coupon_rate,face_value,price, cpn_freq):
    '''new yield calc. Face value = redemption. constraints: freq must be int, maturity > settlement.'''
    # c6 = upside, c4 = coupon rate, c14 = n or days diff /180
    # print('settlement_date,maturity_date,coupon_rate,face_value,price, cpn_freq',settlement_date,maturity_date,coupon_rate,face_value,price, cpn_freq)
    if cpn_freq !=2 :
        print('potential error! with cpn_freq')
    if not settlement_date or not maturity_date:
        return None
    coupon_rate = coupon_rate/100
    C14 = (abs(settlement_date-maturity_date).days-0) / 180
    C11 =coupon_rate*face_value
    Yield = ( (face_value-price)/C14 + C11/2 ) / ( (face_value+price)/2 ) * 2
    return Yield



def formula_yield(r, dt1, dt2 , face, price, Yield, fractional, fractional_tail):
    ''' overall equation for credit. up to 6 periods. time t should be greater than 1 period'''
    # dt1,dt2,face,price,Yield = datetime(2026, 6, 1),datetime(2023, 10, 23),100, 99.5083, .06125
    t = (dt1-dt2).days / 180
    cf = face * Yield
    zeroes = [0]*6
    remain = [0]*6
    subract = [0]*6
    periods = [ x+1 for x in range(math.ceil(t)  )]
    for i in periods:
        zeroes[i-1] = 1
    remain[len(periods)-1] = t-int(t)
    subract[len(periods)-1] = 1
    tail_piece = (cf/2)*(fractional_tail)
    print(cf,t,remain,zeroes, subract,'cf,t,R,Z,S' , fractional)
    return ( zeroes[0]*(cf)/((1+r)**(fractional)) +zeroes[1]*(cf+face*subract[1])/((1+r)**(1-subract[1]+fractional)) +zeroes[2]*(cf+face*subract[2])/((1+r)**(2-subract[2]+fractional))+ \
           zeroes[3]*(cf+face*subract[3])/((1+r)**(3-subract[3]+fractional)) +zeroes[4]*(cf+face*subract[4])/((1+r)**(4-subract[4]+fractional)) +zeroes[5]*(cf+face*subract[5])/((1+r)**(5-subract[5]+fractional)) -price -tail_piece )
    # else:
    #     return ( ( (cf+face) / ((1 + r ) ** (t) ) ) -price )

def run_solver_credit_yield(dt1, dt2 , face, price, Yield, guess=.04):
    ''' solve yield equation for credit with datetimes, price, yield=r. Use float(Answer[0]) on returned answer. dt1 = maturity.  '''
    # face = 100
    # cf = face * .0975
    # pr = 99.04
    period,schedules = pay_periods_calculator(dt1, dt2, dt1)
    dsc ,_ = maturity_schedules(dt1,dt2,schedules)
    fractional = dsc/180
    fractional_tail = (180-dsc)/180
    final_yield = fsolve(formula_yield, guess ,args=(dt1, dt2 , face, price, Yield, fractional, fractional_tail) )
    return (final_yield)


def maturity_schedules(maturity,given_date,schedules,greaterThan=1):
    '''  find closest date to scheduled list of maturities thats semi-annual. return string and # of days, inputs datetime'''
    # given_date = datetime.strptime(given_date,'%Y-%m-%d')
    ## date_range doesnt create accurate dates separated periodically
    # schedules = pd.date_range(end=maturity, periods=count, freq='6MS').tolist()
    count = int(abs(given_date.year - maturity.year))*2 +1
    if greaterThan:
        closest_date = min(schedules, key=lambda x: (x < given_date, abs(given_date - x)))  #(x < given_date, abs(given_date - x)
        diff_days = (closest_date - given_date).days
    else:
        closest_date = min(schedules, key=lambda x: (x > given_date, abs(given_date - x)))
        diff_days = ( given_date-closest_date).days
    return [diff_days,closest_date.strftime("%Y-%m-%d")]


def pay_periods_calculator(maturity_date,settle_date,close_date):
    '''counts full pay periods between dates where maturity != bond closing date, input datetimes, return number and stirngs'''
    tup = []
    # settle_date = datetime.strptime(settle,'%Y-%m-%d')
    # close_date = datetime.strptime(close, '%Y-%m-%d')
    # maturity_date = datetime.strptime(maturity,'%Y-%m-%d')
    # day_int = maturity_date.day
    count = int(abs(settle_date.year - maturity_date.year))*2 +1
    six_months = maturity_date
    for i in range(count):
        tup.append(six_months)
        six_months -= relativedelta(months=+6)
    # pd.date_range doesnt produce exact 6month intervals!!!
    res = [x for x in tup if x < close_date and x > settle_date]
    return  len(res), res


def yield_bbg_pull(tgt_isin, nxt_call_date_str, dt_today_str, px_nxt, price):
    '''yield and spread analysis bbg  '''
    override_yld = {'YAS_WORKOUT_DT': nxt_call_date_str,
                    'SETTLE_DT': dt_today_str,
                    'YAS_WORKOUT_PX': str(px_nxt),
                    'YAS_BOND_PX': str(price)}
    ytc = bbgclient.bbgclient.get_secid2field([tgt_isin], "BBGID", overrides_dict=override_yld,
                                               fields=['YAS_BOND_YLD'], req_type="refdata")
    YTC1 = ytc.get(tgt_isin, []).get('YAS_BOND_YLD', [])
    YTC1 = float(YTC1[0]) if YTC1[0] else None
    return YTC1


def mod_duration(settlement_date,maturity_date,coupon_rate,face_value,yields):
    '''excel Modified duration equivalent funct.
        excel: MDURATION(settlement, maturity, coupon, yld, frequency).
        goal: summing products of time periods and present values of cash flows.
        maturity_date -> datetime, coupon_rate-> decimal'''
    #test
    # settlement_date = datetime(2023,10,6,0,0)
    # maturity_date = datetime(2025,9,30,0,0)
    # coupon_rate = .0559
    # yields = 0.0575
    # face_value = 100.75
    period = 0
    listper = []
    listyear = []
    cashflow,weight_cf,weight,presentvalue_weight,time_weight = [],[],[],[],[]
    tmp =0
    cf = 0
    modDuration = None
    years = (maturity_date - settlement_date ).days /360
    while years > 0:
        period +=1
        listper.append(period)
        if years >= 0.5:
            tmp += 0.5
            listyear.append(tmp)
        elif years > 0:
            listyear.append(tmp+years)
        else:
            break
        years -= 0.5
        period +=1

    for i in range(len(listper)):
        if i != len(listper)-1:
            cf = ((coupon_rate * face_value)/2)
            cashflow.append( cf )
        else:
            cf = (((coupon_rate * face_value) / 2) + face_value)
            cashflow.append( cf )
        presentvalue_weight.append( (cashflow[i] / ((1 + yields) ** listyear[i])) )
        time_weight.append( (presentvalue_weight[i] * listyear[i]) )
    # sum
    wsum = sum(time_weight)
    pvsum = sum(presentvalue_weight)
    denominator = (1+yields/2)
    modDuration = (wsum/pvsum)/denominator
    return modDuration


def process_credit_data(tgt_isin='',deal_name='',aum_pct='',dv='',cv='',fund='',aums=''):
    ''' daily process to calc credit data and then save at EOD.
    deal_name variable can be in form "XXX-XXX%" '''

    # if 'ZEV' not in deal_name:
    #     return
    global worst_call_date_str
    save = True
    test = False
    new_dict = {}
    threshold_yield = 40
    if len(tgt_isin) ==0 and len(deal_name)==0:
        return
    if test:
        tgt_isin = 'BBG00RPJR9J2' #"BBG00S2DZB77"
        # coupon_rate,face_value,price,cpn_freq = 4.25, 102.125, 98.58, 2
        yield_to_maturity = 1
        deal_name = 'VVV%'  # 'MCC-CDR%'

    field_search = ['DUR_ADJ_MID', 'LAST_CLOSE_TRR_5D', 'LAST_CLOSE_TRR_MTD', 'Maturity', 'Nxt_call_dt',
                    'Nxt_call_px', 'Second_call_dt', 'Second_call_px', 'Call_Schedule', 'DUR_ADJ_MTY_MID', 'RISK_MID',
                    'YLD_YTM_MID', 'YLD_YTC_MID', 'YLD_CNV_MID', 'WORKOUT_DATE_MID_TO_WORST', 'YAS_WORKOUT_PX',
                    'YAS_WORKOUT_DT',
                    'Cpn', 'Cpn_freq', 'Cpn_typ', 'Par', 'CALLABLE', 'CALLED',
                    'DEBT_TO_EQUITY_FUNDAMENTALS_TKR', 'Collat_Typ',
                    'Market_Issue', 'Country',
                    'Issue_Dt', 'Amt_Issued', 'Rtg_Moody', 'Rtg_SP', 'bb_compste_rating_ig_hy_indctr',
                    'Payment_Rank',
                    'YAS_YLD_Spread', 'YAS_OAS_SPRD', 'YAS_ISpread_To_Govt', 'YAS_ASW_Spread', 'YAS_ZSpread']
    field_claws = ['bullet', 'make whole termination date', 'make whole call spread', 'clawback indicator',
                   'clawback termination date', 'clawback price', 'clawback percentage']

    # flatfile
    if '%' in deal_name:
        flatfile_df = pd.read_sql_query('Select * from ' +
                                        'wic.daily_flat_file_db where tradegroup like "' + deal_name + '" ',
                                        con=connection)  # and fund="' +fund + '";
    else:
        flatfile_df = pd.read_sql_query('Select * from ' +
                                        'wic.daily_flat_file_db where tradegroup="' + deal_name + '" AND ' +
                                        'BloombergID="' + str(tgt_isin) + '"',
                                        con=connection)  # and fund="' +fund + '";

    flatfile_df = flatfile_df.sort_values(by=['Flat_file_as_of']).tail(1)

    if flatfile_df['Flat_file_as_of'].any():
        new_dict['closing_date'] = flatfile_df['ClosingDate'].values[0]
        new_dict['deal_upside'] = flatfile_df['DealUpside'].values[0]
        new_dict['deal_downside'] = flatfile_df['DealDownside'].values[0]
        new_dict['market_cap_name'] = flatfile_df['MarketCapName'].values[0]
        new_dict['industry'] = flatfile_df['Industry'].values[0]
        new_dict['price'] = flatfile_df['Price'].values[0]
        new_dict['bucket'] = flatfile_df['Bucket'].values[0]
        new_dict['tradegroup'] = flatfile_df['TradeGroup'].values[0]
        new_dict['target_ticker'] = flatfile_df['Ticker'].values[0]
        new_dict['isin'] = flatfile_df['ISIN'].values[0]
        new_dict['catalyst_type_wic'] = flatfile_df['CatalystTypeWIC'].values[0]
        new_dict['pct_aum'] = aum_pct
    else:
        new_dict['closing_date'], new_dict['deal_upside'], new_dict['deal_downside'], new_dict['market_cap_name'], \
        new_dict['industry'], \
        new_dict['price'], new_dict['bucket'], new_dict['tradegroup'], new_dict['target_ticker'], new_dict['isin'], \
        new_dict['pct_aum'] = [None] * 11

    override = {
        'PX_ASK': new_dict['price']}  # 'USER_LOCAL_TRADE_DATE': as_of_yyyymmdd, 'TRADE_DATE_CV_MODEL': as_of_yyyymmdd}
    bbg_output = bbgclient.bbgclient.get_secid2field([tgt_isin], "BBGID", overrides_dict=override, fields=field_search,
                                                     req_type="refdata")
    bbg_output = bbg_output[tgt_isin]
    bbg_claw_output = bbgclient.bbgclient.get_secid2field([tgt_isin + ' corp'], "BBGID", overrides_dict=override,
                                                          fields=field_claws, req_type="refdata")
    bbg_claw_output = bbg_claw_output[tgt_isin + ' corp']
    bbg_output = {**bbg_output, **bbg_claw_output}
    bbg_sectype = bbgclient.bbgclient.get_secid2field([tgt_isin], "BBGID", fields=['SECURITY_TYP'], req_type="refdata")
    bbg_sectype = bbg_sectype[tgt_isin]

    if len(bbg_output) == 0:
        return

    # yield calcs
    coupon_rate, price, cpn_freq = bbg_output.get('Cpn', [0])[0], new_dict['price'], bbg_output.get('Cpn_freq', [None])[
        0]
    face_value = new_dict['deal_upside']
    cpn_freq = int(cpn_freq) if cpn_freq else None
    coupon_rate = float(coupon_rate) if coupon_rate else None
    closing_date = new_dict['closing_date']
    close_date_copy = new_dict['closing_date']
    # maturity = datetime.strptime(bbg_output.get('Maturity', [None])[0], '%Y-%m-%d')
    # yrM, mthM, dyM = maturity.year, maturity.month, maturity.day
    if closing_date:
        yrC, mthC, dyC = closing_date.year, closing_date.month, closing_date.day
        closing_date = datetime.combine(closing_date, datetime.min.time())
        closing_date_str = closing_date.strftime('%Y%m%d')
    else:
        yrC, mthC, dyC = 0, 0, 0
    dt_today = date.today()
    yrN, mthN, dyN = dt_today.year, dt_today.month, dt_today.day
    dt_today = datetime.combine(dt_today, datetime.min.time())
    dt_today_str = dt_today.strftime('%Y%m%d')
    if bbg_output.get('Nxt_call_dt', [None])[0]:
        nxt_call_date = datetime.strptime(bbg_output['Nxt_call_dt'][0], '%Y-%m-%d')
        nxt_call_date_str = nxt_call_date.strftime('%Y%m%d')
        yrCS, mthCS, dyCS = nxt_call_date.year, nxt_call_date.month, nxt_call_date.day
    else:
        yrCS, mthCS, dyCS = 0, 0, 0
        nxt_call_date_str = None
    if bbg_output.get('Second_call_dt', [None])[0]:
        snd_call_date = datetime.strptime(bbg_output.get('Second_call_dt', [None])[0], '%Y-%m-%d')
        snd_call_date_str = snd_call_date.strftime('%Y%m%d')
        yrCS2, mthCS2, dyCS2 = snd_call_date.year, snd_call_date.month, snd_call_date.day
    else:
        yrCS2, mthCS2, dyCS2 = 0, 0, 0
        snd_call_date_str = None
    #
    if bbg_output.get('WORKOUT_DATE_MID_TO_WORST', [None])[0]:
        worst_call_date = datetime.strptime(bbg_output.get('WORKOUT_DATE_MID_TO_WORST', [None])[0], '%Y-%m-%d')
        worst_call_date_str = worst_call_date.strftime('%Y%m%d')
        yrCS3, mthCS3, dyCS3 = worst_call_date.year, worst_call_date.month, worst_call_date.day
    else:
        yrCS3, mthCS3, dyCS3 = 0, 0, 0
        worst_call_date_str = None
    # yield duration
    yld_duration = float(bbg_output['YLD_YTM_MID'][0]) if (bbg_output['YLD_YTM_MID'][0]) else None

    # YTC = yield_credit(ql.Date(1, 1, 2023), ql.Date(31, 12, 2023), coupon_rate, face_value, price, cpn_freq)  # 6.75, 100, 98.28)
    if yld_duration and face_value and coupon_rate:
        #  TL is the the only security type for which we need to change the duration field to RISK_MID
        # print('coupon_rate, face_value, yld_duration','-----inputsDte ',coupon_rate, face_value, yld_duration)
        # DTE = mduration(ql.Date(dyN, mthN, yrN), ql.Date(dyC, mthC, yrC), coupon_rate, face_value, yld_duration) if not ((dyC+mthC+yrC == 0) or (closing_date<=dt_today) )  else None
        DTE = mod_duration(dt_today, closing_date, coupon_rate / 100, face_value, yld_duration / 100) if not (
                    (dyC + mthC + yrC == 0) or (closing_date <= dt_today)) else None
        # mod_duration_at_maturity = mod_duration(dt_today, maturity, coupon_rate/100, face_value,yld_duration/100) if not ((dyM + mthM + yrM == 0) or (maturity <= dt_today)) else None
        DTM = new_dict['RISK_MID'] if bbg_sectype['SECURITY_TYP'] == 'TERM' else \
        bbg_output.get('DUR_ADJ_MTY_MID', [None])[0]
        new_dict['dte'] = DTE
        new_dict['dtm'] = DTM if not DTM else float(DTM)

    # restructure lowercase keys dict
    for k, v in bbg_output.items():
        if k in ["RISK_MID", "SECURITY_TYP"]:
            continue
        knew = k.lower()
        new_dict[knew] = v[0] if len(v) else None
        # exception
        if k == 'YLD_YTC_MID':
            new_dict['ytc'] = v[0] if len(v) else None
            # delete old key
            del new_dict['yld_ytc_mid']
        if knew in ['callable', 'called']:
            new_dict[knew] = True if v[0] == 'Y' else False
        if knew in ['issue_dt', 'workout_date_mid_to_worst', 'nxt_call_dt']:
            new_dict[knew] = datetime.strptime(v[0], '%Y-%m-%d').date() if v[0] else None
        if knew == 'dur_adj_mid':
            new_dict['eff_duration'] = new_dict.get(knew, None)
        if knew == 'call_schedule':
            if new_dict.get(knew, None):
                dt_str = new_dict[knew]['Call Date'] if new_dict[knew].get('Call Date', None) else None
                px_str = new_dict[knew]['Call Price'] if new_dict[knew] else None
                px_call_schedule = float(px_str)
            else:
                px_call_schedule = None
            #
            px_nxt = bbg_output['Nxt_call_px'][0] if bbg_output['Nxt_call_px'] else None
            px_nxt = float(px_nxt) if px_nxt else None
            #
            px_2nd = bbg_output['Second_call_px'][0] if bbg_output['Second_call_px'] else None
            px_2nd = float(px_2nd) if px_2nd else None
            #
            px_worst = bbg_output['YAS_WORKOUT_PX'][0] if bbg_output['YAS_WORKOUT_PX'] else None
            px_worst = float(px_worst) if px_worst else None
            dt_worst = datetime.strptime(bbg_output.get('YAS_WORKOUT_DT', [None])[0], '%Y-%m-%d') if \
            bbg_output.get('YAS_WORKOUT_DT', [None])[0] else None
            dt_worst = dt_worst.strftime('%Y%m%d') if dt_worst else None
            # dt_val = datetime.strptime(dt_str, '%Y-%m-%d').date()
            yr2, yr3, yr4 = yrCS2, yrCS3, 0
            YTC1, YTC2 , YTC3= None, None, None
            # ytc1,2 taken out due to unusual values
            # YTC1 = yield_bbg_pull(tgt_isin, nxt_call_date_str, dt_today_str, px_nxt, price) if nxt_call_date_str else None#if not ((dyCS+mthCS+yrCS == 0) or (nxt_call_date<=dt_today) or px_nxt==None or nxt_call_date_str or price==None)  else None
            # YTC2 = yield_bbg_pull(tgt_isin, snd_call_date_str, dt_today_str, px_2nd, price) if snd_call_date_str else None #if not (yrCS2+mthCS2+dyCS2 == 0 or snd_call_date<=dt_today or px_2nd==None or snd_call_date_str or price==None) else None
            # YTC3 is the new YTC
            if bbg_output.get('CALLABLE', ['N'])[0] =='Y':
                YTC3 = yield_bbg_pull(tgt_isin, dt_worst, dt_today_str, px_worst,
                                  price) if worst_call_date_str else None  # if not (dyCS3+mthCS3+yr3 == 0 or worst_call_date<=dt_today or px_worst==None or worst_call_date_str or price==None) else None
            # YTC1 = yield_formula(dt_today, nxt_call_date, coupon_rate, price, px_nxt, cpn_freq)  if not ((dyCS+mthCS+yrCS == 0) or (nxt_call_date<=dt_today) or cpn_freq==None or px_nxt==None or coupon_rate==None or price==None)  else None
            new_dict[knew + '_date1'] = datetime(yrCS, mthCS, dyCS) if dyCS + mthCS + yrCS != 0 else None
            new_dict[knew + '_date2'] = datetime(yr2, mthCS2, dyCS2) if dyCS2 + mthCS2 + yr2 != 0 else None
            new_dict[knew + '_date3'] = datetime(yr3, mthCS3, dyCS3) if dyCS3 + mthCS3 + yr3 != 0 else None
            # new_dict[knew+'_date4']= datetime(yr4,mthCS,dyCS)
            new_dict[knew + '_price1'] = px_call_schedule
            new_dict[knew + '_price2'] = float(px_2nd) if px_2nd else None
            new_dict[knew + '_price3'] = float(px_worst) if px_worst else None
            # print('YTC1 ' , YTC1 , dt_today, coupon_rate, price, px_nxt, cpn_freq)
            if YTC1:
                new_dict[knew + '_yeild1'] = YTC1 if not math.isinf(YTC1) else None
            else:
                new_dict[knew + '_yeild1'] = YTC1
            if YTC2:
                new_dict[knew + '_yeild2'] = YTC2 if not math.isinf(YTC2) else None
            else:
                new_dict[knew + '_yeild2'] = YTC2
            if YTC3:
                new_dict[knew + '_yeild3'] = YTC3 if not math.isinf(YTC3) else None
            else:
                new_dict[knew + '_yeild3'] = YTC3
            # delete call sched
            del new_dict['call_schedule']
        if knew == 'make whole termination date':
            new_dict['makewhole_end'] = new_dict['make whole termination date']
            del new_dict['make whole termination date']
        if knew == 'make whole call spread':
            new_dict['mw_spread'] = new_dict['make whole call spread']
            del new_dict['make whole call spread']
        if knew == 'clawback indicator':
            new_dict['clawback'] = new_dict['clawback indicator']
            del new_dict['clawback indicator']
        if knew == 'clawback termination date':
            new_dict['claw_end'] = new_dict['clawback termination date']
            del new_dict['clawback termination date']
        if knew == 'clawback percentage':
            new_dict['equity_claw_pct'] = new_dict['clawback percentage']
            del new_dict['clawback percentage']
        if knew == 'clawback price':
            new_dict['claw_price'] = new_dict['clawback price']
            del new_dict['clawback price']
    del new_dict['yas_workout_dt']
    new_dict['date_updated'] = datetime.today()
    new_dict['time_to_maturity'] = round(( datetime.strptime(new_dict['maturity'],'%Y-%m-%d')- datetime.today()).days / 365.25,6) if new_dict['maturity'] else None
    new_dict['time_to_event'] = (closing_date- datetime.combine(datetime.today(), datetime.min.time())  ).days / 365.25 if new_dict['closing_date'] else None
    #reassigning fields
    new_dict['ytm'],new_dict['ytw'] = new_dict['yld_ytm_mid'],new_dict['yld_cnv_mid']
    new_dict['par'] = round(float(new_dict['price'])) if new_dict['price'] else None
    # use bbg override for yte .
    if face_value and price and closing_date_str:
        # YAS_WORKOUT_PX = upside or redemption, Bond_Px = bond price
        override_yld={'YAS_WORKOUT_DT': closing_date_str,
         'SETTLE_DT': dt_today_str,
         'YAS_WORKOUT_PX': str(face_value),
         'YAS_BOND_PX': str(price)}
        credit_yield =bbgclient.bbgclient.get_secid2field([tgt_isin], "BBGID", overrides_dict=override_yld, fields=['YAS_BOND_YLD'],req_type="refdata")
        credit_yield = credit_yield.get(tgt_isin,[]).get('YAS_BOND_YLD',[])
        new_dict['yte'] = float(credit_yield[0]) if credit_yield[0] else None

    if new_dict.get('yte',None) != None:
        # edge case
        if new_dict.get('yte',0) > 1e9:
            new_dict['yte'] = None

    # override of 40% for YTE YTM YTW YTC, respectively
    for key_yield in ['yte','yld_ytm_mid','yld_cnv_mid','call_schedule_yeild3']:
        val = new_dict.get(key_yield,0)
        if val:
            if float(val) > threshold_yield:
                new_dict[key_yield] = coupon_rate

    # corrections for large values for storage
    if new_dict.get('yas_yld_spread',None) and float(new_dict.get('yas_yld_spread',0)) > 90000:
        new_dict['yas_yld_spread'] = None
    if new_dict.get('yas_ispread_to_govt',None) and float(new_dict.get('yas_ispread_to_govt',0)) > 90000:
        new_dict['yas_ispread_to_govt']= None

    new_dict['dv01'] = dv
    new_dict['cr01'] = cv
    if save:
        credit_fields = CreditStaticScreen._meta.get_fields()
        list_type_field = [[f.__class__.__name__, f.name] for f in credit_fields]
        # verify decimal/date fields dont have '' or 0, for objects.create
        for t in list_type_field:
            if t[0] == 'DecimalField':
                k = t[1]
                if k in new_dict.keys():
                    if new_dict[k]== '' or new_dict[k] == '0':
                        new_dict[k] = None
        ## go through various funds for the same bond
        for i in range(len(fund)):
            new_dict['fund'] = fund[i]
            new_dict['pct_aum'] = aums[i]
            # sometimes there are arbitrary and large values
            try:
                CreditStaticScreen.objects.create(**new_dict)
            except:
                # print(new_dict)
                pass



def pull_single_credit_tickers(ticker,index):
    ''' pull single fund and benchmark data for Returns '''

    fields = ['LAST_CLOSE_TRR_5D','LAST_CLOSE_TRR_MTD','LAST_CLOSE_TRR_YTD','LAST_CLOSE_TRR_1Yr',
    'LAST_CLOSE_TRR_3Yr','LAST_CLOSE_TRR_5Yr']
    bbg_output = bbgclient.bbgclient.get_secid2field([ticker,index], "tickers", fields, req_type="refdata")
    bbg_output_t = bbg_output[ticker]
    bbg_output_i = bbg_output[index]
    if len(bbg_output_t)==0 and len(bbg_output_i)==0:
        return
    final = []
    new_dict ={}
    new_dict['date_updated'] = date.today()
    new_dict['tradegroup'] = ticker
    for k,v in bbg_output_t.items():
        knew = k.lower()
        new_dict[knew] = v[0] if len(v) else None
    final.append(new_dict)
    new_dict = {}
    new_dict['date_updated'] = date.today()
    new_dict['tradegroup'] = index
    for k,v in bbg_output_i.items():
        knew = k.lower()
        new_dict[knew] = v[0] if len(v) else None
    final.append(new_dict)
    credit_fields = CreditStaticScreen._meta.get_fields()
    list_type_field = [[f.__class__.__name__, f.name] for f in credit_fields]
    # verify decimal/date fields dont have '' or 0, for objects.create
    for t in list_type_field:
        if t[0] == 'DecimalField':
            k = t[1]
            if k in new_dict.keys():
                if new_dict[k]== '' or new_dict[k] == '0':
                    new_dict[k] = None
    # print('------',final)
    # print('saving')
    for m in final:
        CreditStaticScreen.objects.create(**m)

def sum_weighted_stats(df,indicies):
    '''calc the weighted avrg stats for credit portfolio '''
    skip_list = ['AGGREGATE_STATS']+indicies
    dt = datetime.today().strftime('%Y-%m-%d')
    credit_data = list( CreditStaticScreen.objects.filter(date_updated=dt).values() )
    credit_data = [x for x in credit_data if x.get('tradegroup',None) not in skip_list]
    for fund in df['Fund'].unique():

        # aum_total has two values: for ytc it has additional callable logic that determines the average
        aum_total = 0
        aum_total_ytc = 0
        new_dict={}
        new_dict['dv01'] =0
        new_dict['cr01'] =0
        new_dict['ytm'] =0
        new_dict['ytc'] =0
        new_dict['ytw'] =0
        new_dict['yte'] =0
        new_dict['dtm'] =0
        new_dict['eff_duration'] =0
        new_dict['dte'] =0
        for dct in credit_data:
            if dct['fund'] != fund:
                continue
            aum_df = df[(df['TradeGroup'] == dct['tradegroup']) & (df['Fund'] == dct['fund'])][['CurrentMktVal_Pct','Flat_file_as_of']]
            aum_df = aum_df.sort_values(by=['Flat_file_as_of'])
            # aum = decimal.Decimal(aum.values[0])
            aum_df = aum_df.tail(1)
            aum_df = aum_df['CurrentMktVal_Pct']
            aum = decimal.Decimal(aum_df.values[0] )/100 if aum_df.any() else None
            if aum:
                aum_total += aum
                new_dict['ytm'] += round(aum*dct.get('ytm',0),6) if dct.get('ytm',0) else 0
                # for ytc, it has additional callable logic that determines the average. Zero means passing on aggregating count
                ytc_indiv = round(aum*dct.get('call_schedule_yeild3',0),6) if dct.get('call_schedule_yeild3',False) else 0
                new_dict['ytc'] += ytc_indiv
                if ytc_indiv:
                    aum_total_ytc += aum
                # print(' prod ytc ,parts aum ytc-- ',aum*dct.get('ytc',0) if dct.get('ytc',0) else 0 , aum,dct.get('ytc',0), dct['tradegroup'])
                # new_dict['ytc_all'].append(aum*dct.get('ytc',0) if dct.get('ytc',0) else 0)
                # new_dict['aum'].append(aum)
                # new_dict['ytc_individual']=dct.get('ytc',0)
                # new_dict['tg'] = dct['tradegroup']
                new_dict['ytw'] += round(aum*dct.get('ytw',0),6) if dct.get('ytw',0) else 0
                new_dict['yte'] += round(aum*dct.get('yte',0),6) if dct.get('yte',0) else 0
                new_dict['dtm'] += round(aum*dct.get('dtm',0),6) if dct.get('dtm',0) else 0
                new_dict['eff_duration'] += round(aum*dct.get('eff_duration',0),6) if dct.get('eff_duration',0) else 0
                new_dict['dte'] += round(aum*dct.get('dte',0),6) if dct.get('dte',0) else 0
                new_dict['dv01'] += round(aum*dct.get('dv01',0),6) if dct.get('dv01',0) else 0
                new_dict['cr01'] += round(aum*dct.get('cr01',0),6) if dct.get('cr01',0) else 0
        aum_total = round(aum_total,7)
        aum_total_ytc = round(aum_total_ytc, 7)
        # normalize
        new_dict['ytm'] = new_dict['ytm']/aum_total if aum_total else None
        new_dict['ytc'] = new_dict['ytc']/aum_total_ytc if aum_total_ytc else None
        new_dict['ytw'] = new_dict['ytw']/aum_total if aum_total else None
        new_dict['yte'] = new_dict['yte']/aum_total if aum_total else None
        new_dict['dtm'] = new_dict['dtm']/aum_total if aum_total else None
        new_dict['eff_duration'] = new_dict['eff_duration']/aum_total if aum_total else None
        new_dict['dte'] = new_dict['dte']/aum_total if aum_total else None
        new_dict['dv01'] = new_dict['dv01']/aum_total if aum_total else None
        new_dict['cr01'] = new_dict['cr01']/aum_total if aum_total else None
        new_dict['fund'] = fund
        new_dict['date_updated'] = date.today()
        new_dict['tradegroup'] = 'AGGREGATE_STATS'
        # print(new_dict['ytm'] , fund, '-----')
        CreditStaticScreen.objects.create(**new_dict)

def run_store_credit_data(all_credit,credit_deals_df):
    '''loop to run process_credit_data'''
    fail=[]
    for tg in all_credit:
        fundL = []
        aumL = []
        bbgid = credit_deals_df[ credit_deals_df['TradeGroup']==tg ]['BloombergID'].unique()
        bbgid = bbgid[0] if len(bbgid) else ''
        credit_one_sorted = credit_deals_df.sort_values(by=['Flat_file_as_of'])
        # check if not empty df
        if len(credit_one_sorted[credit_one_sorted['TradeGroup'] == tg]['CR_01']):
            cr = credit_one_sorted[credit_one_sorted['TradeGroup'] == tg]['CR_01'].values[0]
            dv = credit_one_sorted[credit_one_sorted['TradeGroup'] == tg]['DV_01'].values[0]
        funds = credit_one_sorted[credit_one_sorted['TradeGroup'] == tg]['Fund']
        mkt_val = credit_one_sorted[credit_one_sorted['TradeGroup'] == tg][['CurrentMktVal_Pct','Flat_file_as_of']]
        # mkt_val = mkt_val.tail(1)
        mkt_vals = mkt_val['CurrentMktVal_Pct']
        aums = mkt_val['CurrentMktVal_Pct']
        mkt_val = min(mkt_vals) if mkt_vals.any() else None
        # print(bbgid,tg)
        # if tg == 'PDCE-CVX MERGER BONDS':
        ## go through various funds for the same bond
        for i in range(len(funds)):
            paum = aums.values[i]
            fund = funds.values[i]
            fundL.append(fund)
            aumL.append(paum)
        # print(aumL,fundL)
        process_credit_data(bbgid,tg,mkt_val,dv,cr,fundL,aumL)

def check_credit_duplicates():
    '''check for same day credit data values and delete'''
    dt = datetime.today().strftime('%Y-%m-%d')
    credit_data_len = len(list(CreditStaticScreen.objects.filter(date_updated=dt).values()))
    if credit_data_len > 0:
        # delete
        CreditStaticScreen.objects.filter(date_updated=dt ).delete()


@shared_task
def run_all_credit_deals():
    '''process to run through all credit deals '''

    ticker_credit = 'ACFIX US EQUITY'
    ticker_index ='G0O1 Index'

    credit_deals_df = pd.read_sql_query('Select Fund, TradeGroup, Flat_file_as_of, Ticker, BloombergID, '+
                                                 'Bucket, StrategyTypeWic, DV_01, CR_01 ,CurrentMktVal_Pct, AUM, amount, SecType '+
                                                 ' FROM wic.daily_flat_file_db WHERE '+
                                                 'Flat_file_as_of = (SELECT MAX(Flat_file_as_of) FROM '+
                                                 'wic.daily_flat_file_db) AND Sleeve = "Credit Opportunities" '+
                                                 'AND AlphaHedge = "Alpha"', con=connection)
    # remove amount = 0 bonds
    credit_deals_df.drop(credit_deals_df[credit_deals_df.amount == 0].index, inplace=True)
    # check if there are same day values prior to running
    check_credit_duplicates()
    # save all credit data
    all_credit = credit_deals_df['TradeGroup'].unique()
    run_store_credit_data(all_credit,credit_deals_df)
    # portfolio calcs
    sum_weighted_stats(credit_deals_df, [ticker_credit,ticker_index])
    # # save benchmark & fund
    pull_single_credit_tickers(ticker_credit,ticker_index)
    dbutils.add_task_record()