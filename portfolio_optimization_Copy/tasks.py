import os
import datetime
import numpy as np
import pandas as pd
import re
from tabulate import tabulate
import time

import dbutils
import holiday_utils

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WicPortal_Django.settings")
import django
django.setup()
from celery_progress.backend import ProgressRecorder
from celery import shared_task
from django.db import connection
from sqlalchemy import create_engine
from django.conf import settings
from django_slack import slack_message

import bbgclient
from credit_idea.utils import convert_to_float_else_zero
from risk_reporting.models import CreditDealsUpsideDownside, FormulaeBasedDownsides
from .models import PnlPotentialDailySummary, PnlPotentialESSConstituents, PnlPotentialDate, PnlPotentialDateHistory, \
    PnlPotentialExclusions, PnlPotentialExclusionsHistory, PnlPotentialIncremental, \
    PnlPotentialIncrementalHistory, PnlPotentialScenarios, PnlPotentialScenariosHistory, \
    PnlPotentialESSConstituentsHistory, ArbCreditPnLPotentialDrilldown, ArbCreditPnLPotentialDrilldownHistory
from portfolio_optimization.models import (ArbOptimizationUniverse, CreditHardFloatOptimization, EssPotentialLongShorts,
    EssUniverseImpliedProbability, EssDealTypeParameters, HardFloatOptimization, HardOptimizationSummary)
from slack_utils import get_channel_name
from portfolio_optimization.portfolio_optimization_utils import (calculate_pl_sec_impact, calculate_cr_hedge_pl,
    calculate_cr_hedge_ror, calculate_gross_spread_ror, calculate_arb_hedged_pl, calculate_cr_break_pl,
    calculate_mstrat_weighted_ror)
from .utils import parse_fld, format_data


def clean_model_up(row):
    return row['pt_up'] if not row['model_up'] else row['model_up']


def clean_model_down(row):
    return row['pt_down'] if not row['model_down'] else row['model_down']


@shared_task
def refresh_ess_long_shorts_and_implied_probability(today_date=None):
    """ Shared Task executes at 8.15am and Post to Slack with an updated universe Table """

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    if not today_date:
        today = datetime.datetime.now().date()
    else:
        today = today_date
    api_host = bbgclient.bbgclient.get_next_available_host()
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)

    con = engine.connect()
    try:
        EssUniverseImpliedProbability.objects.filter(Date=today).delete()  # Delete todays records
        EssPotentialLongShorts.objects.filter(Date=today).delete()
        ess_ideas_df = pd.read_sql_query(
            "SELECT  A.id as ess_idea_id, A.alpha_ticker, A.price, A.pt_up, A.pt_wic, A.pt_down,"
            " A.unaffected_date, A.expected_close, A.gross_percentage, A.ann_percentage, "
            "A.hedged_volatility, A.implied_probability, A.category, A.catalyst,"
            " A.deal_type, A.catalyst_tier, A.gics_sector, A.hedges, A.lead_analyst, "
            "IF(model_up=0, A.pt_up, model_up) as model_up, "
            "IF(model_down=0, A.pt_down, model_down) as model_down, "
            "IF(model_wic=0, A.pt_wic, model_wic) as model_wic, A.is_archived FROM " +
            settings.CURRENT_DATABASE +
            ".risk_ess_idea AS A INNER JOIN "
            "(SELECT deal_key, MAX(version_number) AS max_version FROM  "
            + settings.CURRENT_DATABASE + ".risk_ess_idea GROUP BY deal_key) AS B "
                                          "ON A.deal_key = B.deal_key AND "
                                          "A.version_number = B.max_version AND "
                                          "A.is_archived=0 "
                                          "AND A.status in ('Reviewed', 'ReadyForReview')"
                                          "LEFT JOIN "
                                          "(SELECT DISTINCT X.deal_key,"
                                          "X.pt_up as model_up, "
                                          "X.pt_down AS model_down, X.pt_wic AS model_wic "
                                          "FROM "
            + settings.CURRENT_DATABASE + ".risk_ess_idea_upside_downside_change_records  "
                                          "AS X "
                                          "INNER JOIN "
                                          "(SELECT deal_key, MAX(date_updated) AS "
                                          "MaxDate FROM " + settings.CURRENT_DATABASE +
            ".risk_ess_idea_upside_downside_change_records GROUP BY deal_key) AS Y ON "
            "X.deal_key = Y.deal_key WHERE X.date_updated = Y.MaxDate) AS ADJ ON "
            "ADJ.deal_key = A.deal_key ", con=con)

        # Take only Relevant Columnns
        ess_ideas_df = ess_ideas_df[['ess_idea_id', 'alpha_ticker', 'price', 'pt_up', 'pt_wic', 'pt_down',
                                     'unaffected_date', 'expected_close', 'category', 'catalyst',
                                     'deal_type', 'catalyst_tier', 'gics_sector', 'hedges', 'lead_analyst', 'model_up',
                                     'model_down', 'model_wic']]

        ess_ideas_tickers = ess_ideas_df['alpha_ticker'].unique()

        ess_ideas_live_prices = pd.DataFrame.from_dict(bbgclient.bbgclient.get_secid2field(ess_ideas_tickers, 'tickers',
                                                                                           ['PX_LAST'],
                                                                                           req_type='refdata',
                                                                                           api_host=api_host),
                                                       orient='index').reset_index()
        ess_ideas_live_prices.columns = ['alpha_ticker', 'Price']
        ess_ideas_live_prices['Price'] = ess_ideas_live_prices['Price'].apply(lambda px: float(px[0]) if px[0] else 0)
        ess_ideas_df = pd.merge(ess_ideas_df, ess_ideas_live_prices, how='left', on='alpha_ticker')
        del ess_ideas_df['price']
        ess_ideas_df.rename(columns={'Price': 'price'}, inplace=True)

        def adjust_for_london_stocks(row):
            if 'LN EQUITY' in row['alpha_ticker'].upper():
                return row['price']/100

            return row['price']

        ess_ideas_df['price'] = ess_ideas_df.apply(adjust_for_london_stocks, axis=1)

        ess_ideas_df['model_up'] = ess_ideas_df.apply(clean_model_up, axis=1)
        ess_ideas_df['model_down'] = ess_ideas_df.apply(clean_model_down, axis=1)
        ess_ideas_df['Implied Probability'] = 1e2 * (ess_ideas_df['price'] - ess_ideas_df['model_down']) / (
                ess_ideas_df['model_up'] - ess_ideas_df['model_down'])

        implied_prob_attention_names = list(ess_ideas_df[ess_ideas_df['Implied Probability'] > 100]['alpha_ticker'].unique())
        implied_prob_attention_names += list(ess_ideas_df[ess_ideas_df['Implied Probability'] < 0]['alpha_ticker'].unique())

        def clean_implied_probability(row):
            if row['Implied Probability'] > 100 or row['Implied Probability'] < 0:
                return 1e2 * (row['price'] - row['pt_down']) / (
                row['pt_up'] - row['pt_down'])

            return row['Implied Probability']

        ess_ideas_df['Implied Probability'] = ess_ideas_df.apply(clean_implied_probability, axis=1)
        ess_ideas_df['Return/Risk'] = abs(
            (ess_ideas_df['model_up'] / ess_ideas_df['price'] - 1) / (ess_ideas_df['model_down'] / ess_ideas_df['price'] - 1))
        ess_ideas_df['Gross IRR'] = (ess_ideas_df['model_up'] / ess_ideas_df['price'] - 1) * 1e2

        ess_ideas_df['Days to Close'] = (ess_ideas_df['expected_close'] - today).dt.days
        ess_ideas_df['Ann IRR'] = (ess_ideas_df['Gross IRR'] / ess_ideas_df['Days to Close']) * 365

        def calculate_adj_ann_irr(row):
            if row['hedges'] == 'Yes':
                return row['Ann IRR'] - 15

            return row['Ann IRR']

        ess_ideas_df['Adj. Ann IRR'] = ess_ideas_df.apply(calculate_adj_ann_irr, axis=1)
        # Targets currently hard-coded (should be customizable)
        ls_targets_df = pd.DataFrame.from_records(EssDealTypeParameters.objects.all().values())
        ls_targets_df.rename(columns={'long_probability': 'long_prob', 'short_probability': 'short_prob'}, inplace=True)
        ls_targets_df.drop(columns=['id', 'long_max_risk', 'long_max_size', 'short_max_risk', 'short_max_size'],
                           inplace=True)
        ess_ideas_df = pd.merge(ess_ideas_df, ls_targets_df, how='left', on='deal_type')
        ess_ideas_df['Potential Long'] = ess_ideas_df.apply(
            lambda x: 'Y' if (
                    (x['Implied Probability'] < x['long_prob']) and (x['Adj. Ann IRR'] > x['long_irr'])) else '',
            axis=1)
        ess_ideas_df['Potential Short'] = ess_ideas_df.apply(lambda x: 'Y' if (
                (x['Implied Probability'] > x['short_prob']) and (x['Adj. Ann IRR'] < x['short_irr'])) else '', axis=1)

        ess_ideas_df['Date'] = today
        x = ess_ideas_df.rename(columns={'Implied Probability': 'implied_probability', 'Return/Risk': 'return_risk',
                                         'Gross IRR': 'gross_irr', 'Days to Close': 'days_to_close',
                                         'Ann IRR': 'ann_irr', 'Adj. Ann IRR': 'adj_ann_irr',
                                         'Potential Long': 'potential_long',
                                         'Potential Short': 'potential_short'})

        deals_with_exp_close_today = x[x['days_to_close'] <= 0].alpha_ticker.unique().tolist()
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        x.to_sql(con=con, if_exists='append', schema=settings.CURRENT_DATABASE,
                 name='portfolio_optimization_esspotentiallongshorts', index=False, chunksize=100)

        x['count'] = x['implied_probability'].apply(lambda y: 1 if not pd.isna(y) else np.nan)
        avg_imp_prob = x[['deal_type', 'count', 'implied_probability']].groupby('deal_type').agg(
            {'implied_probability': 'mean',
             'count': 'sum'}).reset_index()
        x.drop(columns=['count'], inplace=True)
        avg_imp_prob.loc[len(avg_imp_prob)] = ['Soft Universe Imp. Prob',
                                               x[x['catalyst'] == 'Soft']['implied_probability'].mean(),
                                               len(x[x['catalyst'] == 'Soft'])]

        avg_imp_prob['Date'] = today
        # --------------- SECTION FOR Tracking Univese, TAQ, AED Long/Short Implied Probabilities ---------------------
        query = "SELECT DISTINCT flat_file_as_of as `Date`, TradeGroup, Fund, Ticker, " \
                "LongShort, SecType, DealUpside, DealDownside " \
                "FROM wic.daily_flat_file_db  " \
                "WHERE Flat_file_as_of = (SELECT MAX(flat_file_as_of) from wic.daily_flat_file_db) AND Fund  " \
                "IN ('AED', 'TAQ') and AlphaHedge = 'Alpha' AND  " \
                "LongShort IN ('Long', 'Short') AND SecType = 'EQ' " \
                "AND Sleeve = 'Equity Special Situations' and amount<>0"

        imp_prob_tracker_df = pd.read_sql_query(query, con=con)
        imp_prob_tracker_df['Ticker'] = imp_prob_tracker_df['Ticker'] + ' EQUITY'
        ess_tickers = imp_prob_tracker_df['Ticker'].unique()

        live_price_df = pd.DataFrame.from_dict(bbgclient.bbgclient.get_secid2field(ess_tickers, 'tickers',
                                                                                   ['CRNCY_ADJ_PX_LAST'],
                                                                                   req_type='refdata',
                                                                                   api_host=api_host),
                                               orient='index').reset_index()
        live_price_df.columns = ['Ticker', 'Price']
        live_price_df['Price'] = live_price_df['Price'].apply(lambda px: float(px[0]) if px[0] else 0)
        imp_prob_tracker_df = pd.merge(imp_prob_tracker_df, live_price_df, how='left', on='Ticker')
        imp_prob_tracker_df['implied_probability'] = 1e2 * (
                    imp_prob_tracker_df['Price'] - imp_prob_tracker_df['DealDownside']) / (
                                                                 imp_prob_tracker_df['DealUpside'] -
                                                                 imp_prob_tracker_df['DealDownside'])

        imp_prob_tracker_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace Inf values
        imp_prob_tracker_df['count'] = imp_prob_tracker_df['implied_probability'].apply(
            lambda x: 1 if not pd.isna(x) else np.nan)
        grouped_funds_imp_prob = imp_prob_tracker_df[['Date', 'Fund', 'LongShort', 'implied_probability', 'count']]. \
            groupby(['Date', 'Fund', 'LongShort']).agg({'implied_probability': 'mean', 'count': 'sum'}).reset_index()
        imp_prob_tracker_df.drop(columns=['count'], inplace=True)

        grouped_funds_imp_prob['deal_type'] = grouped_funds_imp_prob['Fund'] + " " + grouped_funds_imp_prob['LongShort']
        grouped_funds_imp_prob = grouped_funds_imp_prob[['Date', 'deal_type', 'implied_probability', 'count']]

        # --------------- POTENTIAL LONG SHORT LEVEL IMPLIED PROBABILITY TRACKING --------------------------------------
        ess_potential_ls_df = pd.read_sql_query("SELECT * FROM " + settings.CURRENT_DATABASE +
                                                ".portfolio_optimization_esspotentiallongshorts where Date='" +
                                                today.strftime("%Y-%m-%d") + "'", con=con)

        catalyst_rating_dfs = ess_potential_ls_df[['alpha_ticker', 'catalyst', 'catalyst_tier', 'price',
                                                   'implied_probability', 'potential_long', 'potential_short']]

        ess_potential_ls_df = ess_potential_ls_df[['alpha_ticker', 'price', 'implied_probability',
                                                   'potential_long', 'potential_short']]

        # -------------- Section for Implied Probabilities Grouped by Catalyst and Tiers -------------------------------
        catalyst_rating_dfs['deal_type'] = catalyst_rating_dfs.apply(lambda x: x['catalyst'] + "-" +
                                                                               x['catalyst_tier'], axis=1)

        catalyst_rating_dfs['count'] = catalyst_rating_dfs['implied_probability'].apply(lambda y: 1 if not pd.isna(y) else np.nan)

        catalyst_implied_prob = catalyst_rating_dfs[['deal_type', 'implied_probability', 'count']].groupby('deal_type').agg({'implied_probability': 'mean','count': 'sum'}).reset_index()
        catalyst_implied_prob['Date'] = today

        def classify_ess_longshorts(row):
            classification = 'Universe (Unclassified)'
            if row['potential_long'] == 'Y':
                classification = 'Universe (Long)'
            if row['potential_short'] == 'Y':
                classification = 'Universe (Short)'

            return classification

        # Get the Whole ESS Universe Implied Probability

        universe_data = ['ESS IDEA Universe', ess_potential_ls_df['implied_probability'].mean(),
                         len(ess_potential_ls_df)]
        all_ess_universe_implied_probability = pd.DataFrame(columns=['deal_type', 'implied_probability', 'count'],
                                                            data=[universe_data])

        all_ess_universe_implied_probability['Date'] = today

        # Section for only Long Short Tagging...

        ess_potential_ls_df['LongShort'] = ess_potential_ls_df.apply(classify_ess_longshorts, axis=1)
        ess_potential_ls_df['count'] = ess_potential_ls_df['implied_probability'].apply(
            lambda x: 1 if not pd.isna(x) else np.nan)
        universe_long_short_implied_probabilities_df = ess_potential_ls_df[['LongShort', 'count',
                                                                            'implied_probability']].groupby(
            ['LongShort']).agg({'implied_probability': 'mean', 'count': 'sum'}).reset_index()

        universe_long_short_implied_probabilities_df['Date'] = today
        universe_long_short_implied_probabilities_df = universe_long_short_implied_probabilities_df. \
            rename(columns={'LongShort': 'deal_type'})

        universe_long_short_implied_probabilities_df = universe_long_short_implied_probabilities_df[
            ['Date', 'deal_type', 'implied_probability', 'count']]

        final_implied_probability_df = pd.concat([avg_imp_prob, all_ess_universe_implied_probability,
                                                  universe_long_short_implied_probabilities_df,
                                                  grouped_funds_imp_prob, catalyst_implied_prob])
        del final_implied_probability_df['Date']

        final_implied_probability_df['Date'] = today
        final_implied_probability_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace Inf values
        final_implied_probability_df.to_sql(index=False, name='portfolio_optimization_essuniverseimpliedprobability',
                                            schema=settings.CURRENT_DATABASE, con=con, if_exists='append')

        print('refresh_ess_long_shorts_and_implied_probability : Task Done')

        message = '~ _(Risk Automation)_ *Potential Long/Shorts & Implied Probabilities Refereshed* \n ' \
                  'Link for Potential Long/Short candidates: ' \
                  'http://192.168.0.16:8000/portfolio_optimization/ess_potential_long_shorts'
        if deals_with_exp_close_today:
            message += '\n\n*Following IDEAs need attention - Expected closing date is in PAST* ' \
                       '(_ESS Potential Long Shorts_):\n' + ', '.join(deals_with_exp_close_today)

        if implied_prob_attention_names:
            message += '\n\n*Following IDEAs need attention - Implied Prob is either 0 or greater than 100* ' \
                       '(_Implied prob. calculated through analyst up & downs_)' \
                       ':\n' + ', '.join(implied_prob_attention_names)

        # Post this Update to Slack
        # Format avg_imp_prob
        final_implied_probability_df = final_implied_probability_df[['deal_type', 'implied_probability', 'count']]
        final_implied_probability_df['implied_probability'] = final_implied_probability_df['implied_probability'].apply(
            lambda ip: str(np.round(ip, decimals=2)) + " %")
        final_implied_probability_df.columns = ['Deal Type', 'Implied Probability', 'Count']

        slack_message('ESS_IDEA_DATABASE_ERRORS.slack',
                      {'message': message,
                       'table': tabulate(final_implied_probability_df, headers='keys', tablefmt='pssql',
                                         numalign='right', showindex=False)},
                      channel=get_channel_name('ess_idea_db_logs'))
        dbutils.add_task_record()
    except Exception as e:
        print('Error in ESS Potential Long Short Tasks ... ' + str(e))
        slack_message('ESS_IDEA_DATABASE_ERRORS.slack', {'message': str(e)},
                      channel=get_channel_name('portal-task-errors'))
        dbutils.add_task_record(status=e)
    finally:
        print('Closing Connection to Relational Database Service....')
        con.close()



@shared_task
def get_arb_optimization_ranks():    # Task runs every morning at 7pm and Posts to Slack
    """ For the ARB Optimized Sleeve & Other M&A Sleeve, calculate the Gross RoR, Ann. RoR
    1. Gross RoR : (AllInSpread/PX_LAST) * 100
    2. Ann. RoR: (Gross RoR/Days to Close) * 365
    3. Risk (%): (NAV Impact)/(% of ARB AUM)     : % of Sleeve Current in positions database
    4. Expected Volatility: (Sqrt(days_to_close) * Sqrt(Gross RoR) * Sqrt(ABS(Risk(%))) / 10
    """

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                               + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)

    con = engine.connect()
    today = datetime.datetime.now().date()
    try:
        query = "SELECT DISTINCT tradegroup, ticker, SecType, AlphaHedge, DealValue AS deal_value, Sleeve AS sleeve, "\
                "Bucket AS bucket, CatalystTypeWIC AS catalyst, DealStatusCustom AS deal_status, "\
                "CatalystRating AS catalyst_rating,CASE WHEN SecType LIKE 'EQSWAP' THEN D_Exposure ELSE CurrentMktVal " \
                "END AS CurrentMktVal,Strike AS StrikePrice, PutCall, "\
                "FXCurrentLocalToBase AS FxFactor, "\
                "amount*factor AS QTY, ClosingDate AS closing_date, Target_Ticker AS target_ticker, " \
                "LongShort AS long_short, TargetLastPrice AS target_last_price,Price AS SecurityPrice, " \
                "AllInSpread AS all_in_spread, DealDownside AS deal_downside, " \
                "datediff(ClosingDate, curdate()) AS days_to_close, "\
                "CASE WHEN SecType LIKE 'EQSWAP' THEN (100*D_Exposure/aum) ELSE " \
                "PctOfSleeveCurrent END AS PctOfSleeveCurrent, aum FROM wic.daily_flat_file_db WHERE "\
                "Flat_file_as_of = (SELECT MAX(Flat_file_as_of) "\
                "FROM wic.daily_flat_file_db) AND LongShort IN ('Long', 'Short') "\
                "AND amount<>0 AND SecType IN ('EQ', 'EXCHOPT', 'EQSWAP') AND Fund = 'ARB' AND ticker NOT LIKE %s "\
                "AND sleeve = 'Merger Arbitrage' AND PctOfSleeveCurrent<>0;"

        # Create two Dataframes (One for adjusting RoRs with Hedges and another @ Tradegroup level for merging later...)
        df = pd.read_sql_query(query, con=con, params=('% CVR',))

        def calculate_usd_target_last_price(tg_value):
            sleeve_pct_df = df[df['tradegroup'] == tg_value]
            alpha_df = sleeve_pct_df[sleeve_pct_df['AlphaHedge'] == 'Alpha']
            if alpha_df.empty:
                return 0
            alpha_df = alpha_df.iloc[0]
            return alpha_df['target_last_price'] / alpha_df['FxFactor']
        df['target_last_price'] = df['tradegroup'].apply(calculate_usd_target_last_price)

        tradegroup_level_df = df.copy()
        del tradegroup_level_df['ticker']
        del tradegroup_level_df['target_ticker']
        del tradegroup_level_df['SecType']
        del tradegroup_level_df['AlphaHedge']
        del tradegroup_level_df['CurrentMktVal']
        del tradegroup_level_df['PutCall']
        del tradegroup_level_df['StrikePrice']
        del tradegroup_level_df['QTY']
        del tradegroup_level_df['FxFactor']
        del tradegroup_level_df['SecurityPrice']
        del tradegroup_level_df['PctOfSleeveCurrent']
        # Drop the duplicates
        tradegroup_level_df = tradegroup_level_df.drop_duplicates(keep='first')
        nav_impacts_df = pd.read_sql_query('SELECT TradeGroup as tradegroup, SUM(BASE_CASE_NAV_IMPACT_ARB) as '
                                           'BASE_CASE_NAV_IMPACT_ARB FROM ' +
                                           settings.CURRENT_DATABASE + '.risk_reporting_positionlevelnavimpacts'
                                                                       ' GROUP BY TradeGroup', con=con)

        # Get Excluded Deals from Formulae Linked Downsides Page

        excluded_deals_query = "SELECT TradeGroup as tradegroup from "+settings.CURRENT_DATABASE+\
                               ".risk_reporting_formulaebaseddownsides where IsExcluded='Yes'"
        excluded_deals_df = pd.read_sql_query(excluded_deals_query, con=con)

        df = df[~(df['tradegroup'].isin(excluded_deals_df['tradegroup'].unique()))]

        df['PnL'] = df.apply(calculate_pl_sec_impact, axis=1)
        # Delete the Security Price column
        del df['SecurityPrice']

        df['pnl_impact'] = 1e2*(df['PnL']/df['aum'])

        rors_df = df[['tradegroup', 'ticker', 'pnl_impact']].groupby(['tradegroup'])['pnl_impact'].sum().reset_index()

        def get_pct_of_sleeve_alpha(row):
            sleeve_pct_df = df[df['tradegroup'] == row]
            try:
                alpha_pct = sleeve_pct_df[sleeve_pct_df['AlphaHedge'] == 'Alpha']['PctOfSleeveCurrent'].sum()
                if not alpha_pct:
                    alpha_pct = 0
            except IndexError:
                alpha_pct = 0
            # Returns the Alpha Current Sleeve %
            return float(alpha_pct)

        rors_df['pct_of_sleeve_current'] = rors_df['tradegroup'].apply(get_pct_of_sleeve_alpha)

        # Calculate the RoR
        rors_df['gross_ror'] = 1e2*(rors_df['pnl_impact']/rors_df['pct_of_sleeve_current'])
        rors_df = pd.merge(rors_df, tradegroup_level_df, how='left', on=['tradegroup'])  # Adds Tradegroup level cols.

        rors_df['ann_ror'] = (rors_df['gross_ror']/rors_df['days_to_close'])*365
        rors_df = pd.merge(rors_df, nav_impacts_df, how='left', on='tradegroup')
        rors_df.rename(columns={'BASE_CASE_NAV_IMPACT_ARB': 'base_case_nav_impact'}, inplace=True)

        rors_df['base_case_nav_impact'] = rors_df['base_case_nav_impact'].astype(float)
        rors_df['pct_of_sleeve_current'] = rors_df['pct_of_sleeve_current'].astype(float)
        rors_df['risk_pct'] = 1e2 * (rors_df['base_case_nav_impact']/rors_df['pct_of_sleeve_current'])
        rors_df['risk_pct_unhedged'] = 1e2 * (1 - (rors_df['deal_downside']/rors_df['target_last_price']))

        rors_df['expected_vol'] = (np.sqrt(rors_df['days_to_close']) * np.sqrt(rors_df['gross_ror']) *
                                   np.sqrt(abs(rors_df['risk_pct'])))/10

        rors_df['date_updated'] = today

        rors_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Remove unwanted columns
        del rors_df['aum']

        rors_df[['gross_ror', 'ann_ror', 'base_case_nav_impact', 'risk_pct', 'risk_pct_unhedged', 'expected_vol']] = \
            rors_df[['gross_ror', 'ann_ror', 'base_case_nav_impact', 'risk_pct','risk_pct_unhedged', 'expected_vol']].fillna(value=0)
        rors_df['hedge_ror'] = np.nan
        ArbOptimizationUniverse.objects.filter(date_updated=today).delete()
        rors_df.to_sql(name='portfolio_optimization_arboptimizationuniverse', schema=settings.CURRENT_DATABASE,
                       if_exists='append', index=False, con=con)
        credit_df = get_credit_sleeve_optimization_df()
        if not credit_df.empty:
            credit_df.to_sql(name='portfolio_optimization_arboptimizationuniverse', schema=settings.CURRENT_DATABASE,
                             if_exists='append', index=False, con=con)
        else:
            slack_message('eze_uploads.slack', {'error': 'ERROR - Arb/credit hard opt - Credit Df is empty'},
                          channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN)
        slack_message('eze_uploads.slack', {'null_risk_limits':
                                            str("_(Risk Automation)_ Successfully calculated ARB, Credit RoRs. "
                                                "Visit http://192.168.0.16:8000/portfolio_optimization/wic_universe_rors")},
                      channel=get_channel_name('portal_downsides'), token=settings.SLACK_TOKEN)
        dbutils.add_task_record()
        return 'WIC Universe RoRs calculated!'
    except Exception as e:
        print('ARB/Credit Sleeve optimization failed', e)
        slack_message('eze_uploads.slack', {'error': 'ARB/Credit Sleeve optimization failed. ' + str(e)},
                      channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN)
        dbutils.add_task_record(status=e)
        return 'WIC Universe RORs exception'
    finally:
        print('Closing connection to Relational Database Service...')
        con.close()


def get_credit_sleeve_optimization_df():
    today = datetime.datetime.now().date()
    query = 'SELECT DISTINCT TradeGroup, Ticker, DealValue, Price, Analyst, OriginationDate, ClosingDate, Coupon, ' + \
            'D_Exposure, TargetLastPrice, amount, Factor, SecType, PutCall, CurrentMktVal, PctOfSleeveCurrent, '+ \
            'RiskLimit, BloombergID, CatalystTypeWIC, CatalystRating, AlphaHedge, Strike, Sleeve, Bucket, ' + \
            'datediff(ClosingDate, curdate()) as days_to_close, FXCurrentLocalToBase, Target_Ticker, LongShort, '+ \
            'AUM, DealDownside, DealUpside, DealStatusCustom from wic.daily_flat_file_db where Flat_file_as_of = ' + \
            '(Select max(Flat_file_as_of) from wic.daily_flat_file_db) and Sleeve="Credit Opportunities" ' + \
            'and Amount <> 0 and SecType in ("B", "EQ", "CVB", "EXCHOPT") and Fund = "TACO"'
    credit_df = pd.read_sql(query, connection)
    credit_df.rename(columns={'DealValue': 'deal_value', 'Price': 'SecurityPrice', 'Strike': 'StrikePrice',
                              'Sleeve': 'sleeve', 'FXCurrentLocalToBase': 'FxFactor', 'TradeGroup': 'tradegroup',
                              'Bucket': 'bucket', 'CatalystTypeWIC': 'catalyst', 'ClosingDate': 'closing_date',
                              'CatalystRating': 'catalyst_rating', 'Target_Ticker': 'target_ticker',
                              'LongShort': 'long_short', 'AUM': 'aum',
                              'DealDownside': 'deal_downside', 'DealUpside': 'deal_upside', 'Ticker': 'ticker',
                              'D_Exposure': 'd_exposure', 'DealStatusCustom': 'deal_status'}, inplace=True)

    credit_upside_df = pd.DataFrame.from_records(CreditDealsUpsideDownside.objects.all().values())
    if credit_upside_df.empty:
        slack_message('eze_uploads.slack', {'error': 'ERROR: Credit Upside Downside Table empty.'},
                                            channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN)
        return pd.DataFrame()
    excluded_credit_df = credit_upside_df[credit_upside_df['is_excluded'] == 'Yes']
    if not excluded_credit_df.empty:
        credit_df = credit_df[~(credit_df['tradegroup'].isin(excluded_credit_df['tradegroup'].unique()))]
    credit_df = pd.merge(credit_df, credit_upside_df[['tradegroup', 'ticker', 'upside']], on=['tradegroup', 'ticker'],
                         how='left')
    def convert_to_float(value):
        return convert_to_float_else_zero(value)
    credit_df['target_last_price'] = credit_df['TargetLastPrice'] / credit_df['FxFactor']
    credit_df['deal_upside'] = credit_df['deal_upside'].apply(convert_to_float)
    credit_df['target_last_price'] = credit_df['target_last_price'].apply(convert_to_float)
    credit_df['spread'] = credit_df['deal_upside'] - credit_df['target_last_price']
    credit_df['QTY'] = credit_df['amount'] * credit_df['Factor']
    credit_df['upside'] = credit_df['upside'].fillna(0.00)
    credit_df['upside'] = credit_df['upside'].replace('', 0.00)
    credit_df['upside'] = credit_df['upside'].astype(float)
    credit_df['PnL'] = credit_df.apply(calculate_cr_hedge_pl, axis=1) # CR Hedge PL
    credit_df['arb_hedged_pl'] = credit_df.apply(calculate_arb_hedged_pl, axis=1) # ARB Hedge PL
    credit_df['pnl_impact'] = 1e2 * (credit_df['PnL'] / credit_df['aum']) # CR Hedge NAV PL
    credit_df['arb_nav_pnl'] = 1e2 * (credit_df['arb_hedged_pl'] / credit_df['aum']) # ARB Hedge NAV PL
    merged_df = credit_df[['tradegroup', 'pnl_impact', 'arb_nav_pnl']].groupby(['tradegroup']).agg(
        {'pnl_impact': 'sum', 'arb_nav_pnl': 'sum'}).reset_index()

    def get_pct_sleeve_alpha(row):
        sleeve_pct_df = credit_df[credit_df['tradegroup'] == row]
        alpha_pct = sleeve_pct_df[sleeve_pct_df['AlphaHedge'] == 'Alpha']
        if alpha_pct.empty:
            alpha_pct = np.nan
        else:
            alpha_pct = convert_to_float_else_zero(alpha_pct['PctOfSleeveCurrent'].iloc[0])
        return alpha_pct

    credit_df.drop(columns=['pnl_impact', 'arb_nav_pnl'], inplace=True)
    credit_ror_df = pd.merge(credit_df, merged_df, on='tradegroup', how='left')
    credit_ror_df['pct_of_sleeve_current'] = credit_ror_df['tradegroup'].apply(get_pct_sleeve_alpha)
    credit_ror_df = credit_ror_df.dropna(subset=['pct_of_sleeve_current'])
    credit_ror_df['hedge_ror'] = credit_ror_df.apply(calculate_cr_hedge_ror, axis=1) # CR Hedge ROR
    credit_ror_df['delta_adj_pct_aum'] = 1e2 * credit_ror_df['d_exposure'] / credit_ror_df['aum']
    credit_ror_df['gross_spread_ror'] = credit_ror_df.apply(calculate_gross_spread_ror, axis=1)
    credit_ror_df['gross_carry_ror'] = credit_ror_df['Coupon'] * credit_ror_df['days_to_close'] / 365
    credit_ror_df['gross_ror'] = (credit_ror_df['hedge_ror'] + credit_ror_df['gross_spread_ror'] +
                                  credit_ror_df['gross_carry_ror'])
    credit_ror_df['ann_ror'] = (credit_ror_df['gross_ror'] / credit_ror_df['days_to_close']) * 365
    credit_ror_df['risk_pct_unhedged'] = (1e2 * (credit_ror_df['deal_downside'] - credit_ror_df['target_last_price']) /
                                          credit_ror_df['target_last_price'])
    credit_ror_df['risk_pct'] = 0
    hedge_df = credit_ror_df[credit_ror_df['AlphaHedge'] == 'Hedge'][['tradegroup', 'hedge_ror']]
    hedge_df = hedge_df.groupby('tradegroup').mean()
    credit_ror_df = credit_ror_df[credit_ror_df['AlphaHedge'] == 'Alpha']
    credit_ror_df.drop(columns=['hedge_ror'], inplace=True)
    credit_ror_df = pd.merge(credit_ror_df, hedge_df, on='tradegroup', how='left')
    credit_ror_df['expected_vol'] = (np.sqrt(credit_ror_df['days_to_close']).fillna(0) *
                                     np.sqrt(credit_ror_df['gross_ror']).fillna(0) *
                                     np.sqrt(abs(credit_ror_df['risk_pct_unhedged'])).fillna(0)) / 10
    credit_ror_df['date_updated'] = today
    credit_ror_df['base_case_nav_impact'] = 0
    credit_ror_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    credit_ror_df[['gross_ror', 'ann_ror', 'risk_pct_unhedged', 'expected_vol', 'hedge_ror']] = \
        credit_ror_df[['gross_ror', 'ann_ror', 'risk_pct_unhedged', 'expected_vol', 'hedge_ror']].fillna(value=0)
    credit_ror_df.drop(columns=['ticker', 'SecurityPrice', 'Analyst', 'OriginationDate', 'Coupon', 'TargetLastPrice',
                                'PutCall', 'CurrentMktVal', 'RiskLimit', 'PctOfSleeveCurrent', 'BloombergID', 'Factor',
                                'AlphaHedge', 'StrikePrice', 'FxFactor', 'SecType', 'd_exposure', 'amount', 'QTY',
                                'PnL', 'arb_hedged_pl', 'arb_nav_pnl', 'delta_adj_pct_aum', 'aum',
                                'gross_spread_ror', 'gross_carry_ror', 'deal_upside', 'deal_value'], inplace=True)
    credit_ror_df.rename(columns={'upside': 'deal_value', 'spread': 'all_in_spread'}, inplace=True)
    return credit_ror_df


@shared_task(bind=True)
def credit_hard_float_optimization(self, post_to_slack=True, record_progress=False):

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    curr_credit_query = "SELECT tradegroup, notes, is_excluded, rebal_multiples, rebal_target FROM " + \
                        settings.CURRENT_DATABASE + ".portfolio_optimization_credithardfloatoptimization WHERE " \
                        "date_updated = (SELECT MAX(date_updated) FROM " + settings.CURRENT_DATABASE + \
                        ".portfolio_optimization_credithardfloatoptimization)"
    curr_credit_df = pd.read_sql_query(curr_credit_query, connection)
    if record_progress:
        progress_recorder = ProgressRecorder(self)
        progress_recorder.set_progress(10, 100)
    manually_excluded_df = curr_credit_df[curr_credit_df['is_excluded'] == True]
    if not manually_excluded_df.empty:
        excluded_tradegroups = manually_excluded_df['tradegroup'].unique().tolist()
    else:
        excluded_tradegroups = []
    max_date_updated = ArbOptimizationUniverse.objects.latest('date_updated').date_updated
    hard_query = ArbOptimizationUniverse.objects.filter(sleeve='Credit Opportunities', catalyst='HARD',
                                                        date_updated=max_date_updated)
    hard_df = pd.DataFrame.from_records(hard_query.values())
    if record_progress:
        progress_recorder.set_progress(20, 100)
    hard_df.drop(columns=['days_to_close'], inplace=True)
    query = 'SELECT DISTINCT TradeGroup, Ticker, Price, ISIN, Analyst, OriginationDate, Coupon, Fund, ' + \
            'D_Exposure, amount, Factor, SecType, PutCall, CurrentMktVal, '+ \
            'RiskLimit, CatalystTypeWIC, AlphaHedge, Strike, Sleeve, Bucket, ' + \
            'datediff(ClosingDate, curdate()) as days_to_close, FXCurrentLocalToBase, Target_Ticker, LongShort, '+ \
            'AUM, DealDownside, DealUpside from wic.daily_flat_file_db where Flat_file_as_of = ' + \
            '(Select max(Flat_file_as_of) from wic.daily_flat_file_db) and Sleeve="Credit Opportunities" ' + \
            'and Amount <> 0 and SecType in ("B", "EQ", "CVB", "EXCHOPT") and Fund in ("TACO", "AED", "ARB") and ' + \
            'CatalystTypeWIC="HARD" and AlphaHedge="Alpha"'
    credit_df = pd.read_sql(query, connection)
    merge_df = pd.merge(hard_df, credit_df, how='left', left_on='tradegroup', right_on='TradeGroup')
    if record_progress:
        progress_recorder.set_progress(30, 100)
    isin_list = merge_df.ISIN.tolist()
    isin_list = [i + ' CORP' if (not pd.isna(i) and 'corp' not in i.lower()) else i for i in isin_list]
    api_host = bbgclient.bbgclient.get_next_available_host()
    px_ask_dict = bbgclient.bbgclient.get_secid2field(isin_list, 'tickers', ['PX_ASK'], req_type='refdata',
                                                      api_host=api_host)
    def get_px_last(row):
        isin = row['ISIN']
        if isin:
            isin = isin + ' CORP' if (not pd.isna(isin) and 'corp' not in isin.lower()) else isin
            px_ask = px_ask_dict.get(isin)
            px_ask = px_ask.get('PX_ASK') if px_ask else None
            px_ask = px_ask[0] if px_ask else 0
            return px_ask
        return 0
    merge_df['px_ask_price'] = merge_df.apply(get_px_last, axis=1)
    merge_df['net'] = merge_df['amount'] * merge_df['Factor']

    def calculate_excluded_ann_ror(row):
        if row['tradegroup'] in excluded_tradegroups:
            return True
        else:
            if (row['days_to_close'] < 22 and row['DealUpside'] < 0.1) or row['days_to_close'] < 14:
                return True
        return False
    merge_df['is_excluded'] = merge_df.apply(calculate_excluded_ann_ror, axis=1)
    merge_df['cr_break_pl'] = merge_df.apply(calculate_cr_break_pl, axis=1)
    merge_df['nav_impact'] = 1e2 * merge_df['cr_break_pl'] / merge_df['AUM']
    curr_credit_df.drop(columns=['is_excluded'], inplace=True)
    merge_df = pd.merge(merge_df, curr_credit_df, on='tradegroup', how='left')
    if record_progress:
        progress_recorder.set_progress(40, 100)
    # Calculate ARB Nav Impact
    arb_df = merge_df[merge_df['Fund'] == 'ARB']
    grouped_arb_df = arb_df[['tradegroup', 'nav_impact']].groupby('tradegroup').sum().reset_index()
    arb_df.drop(columns=['nav_impact'], inplace=True)
    arb_df = pd.merge(arb_df, grouped_arb_df, on='tradegroup', how='left')
    arb_df.rename(columns={'nav_impact': 'arb_pct_risk'}, inplace=True)
    arb_df['arb_pct_aum'] = (arb_df['amount'] * arb_df['Factor'] *
                                (arb_df['target_last_price'] / arb_df['FXCurrentLocalToBase']) / arb_df['AUM'])
    # Calculate TACO Nav Impact
    taco_df = merge_df[merge_df['Fund'] == 'TACO']
    grouped_taco_df = taco_df[['tradegroup', 'nav_impact']].groupby('tradegroup').sum().reset_index()
    taco_df.drop(columns=['nav_impact'], inplace=True)
    taco_df = pd.merge(taco_df, grouped_taco_df, on='tradegroup', how='left')
    taco_df.rename(columns={'nav_impact': 'taco_pct_risk'}, inplace=True)
    # Calculate AED Nav Impact
    aed_df = merge_df[merge_df['Fund'] == 'AED']
    grouped_aed_df = aed_df[['tradegroup', 'nav_impact']].groupby('tradegroup').sum().reset_index()
    aed_df.drop(columns=['nav_impact'], inplace=True)
    aed_df = pd.merge(aed_df, grouped_aed_df, on='tradegroup', how='left')
    aed_df['m_strat_pct_aum'] = (1e2 * aed_df['net'] * (aed_df['target_last_price'] / aed_df['FXCurrentLocalToBase']) /
                                 aed_df['AUM'])
    aed_df['rebal_target'] = aed_df.apply(lambda x: (x['rebal_multiples'] * x['pct_of_sleeve_current']) if not
                                                    pd.isna(x['rebal_multiples']) else x['m_strat_pct_aum'], axis=1)
    aed_df['weighted_gross_nav_potential'] = aed_df['m_strat_pct_aum'] * aed_df['gross_ror'] / 1e2
    aed_df = aed_df[['tradegroup', 'nav_impact', 'm_strat_pct_aum', 'weighted_gross_nav_potential', 'rebal_target']]
    merge_df.drop(columns=['nav_impact', 'rebal_target'], inplace=True)
    if record_progress:
        progress_recorder.set_progress(50, 100)
    merge_df = pd.merge(merge_df, aed_df, how='left', on='tradegroup')
    merge_df = pd.merge(merge_df, taco_df[['tradegroup', 'taco_pct_risk']], how='left', on='tradegroup')
    merge_df = pd.merge(merge_df, arb_df[['tradegroup', 'arb_pct_risk', 'arb_pct_aum']], how='left', on='tradegroup')
    merge_df = merge_df.sort_values(by=['catalyst_rating', 'ann_ror'], ascending=[True, False]).reset_index(drop=True)
    if record_progress:
        progress_recorder.set_progress(60, 100)
    excluded_df = merge_df[merge_df['is_excluded'] == True]
    non_excluded_df = merge_df[merge_df['is_excluded'] == False]
    non_excluded_df['non_excluded_pct_aum'] = abs(non_excluded_df['m_strat_pct_aum'])
    non_excluded_df["curr_rtn_wt_duration"] = non_excluded_df['days_to_close'].\
            mul(non_excluded_df['weighted_gross_nav_potential']).fillna(0).cumsum().\
            div(non_excluded_df['weighted_gross_nav_potential'].fillna(0).cumsum())
    non_excluded_df['weighted_nav_cumsum'] = non_excluded_df['weighted_gross_nav_potential'].fillna(0).cumsum()
    non_excluded_df['non_excluded_pct_aum_cumsum'] = non_excluded_df['rebal_target'].fillna(0).cumsum()
    non_excluded_df['curr_rwd_ror'] = (non_excluded_df['weighted_nav_cumsum'] / \
        non_excluded_df['non_excluded_pct_aum_cumsum']) / non_excluded_df['curr_rtn_wt_duration'] * 365 * 100
    non_excluded_df['curr_cwd_ror'] = (non_excluded_df['ann_ror'].fillna(0) * non_excluded_df['rebal_target'].fillna(0)).cumsum() / \
        non_excluded_df['rebal_target'].fillna(0).cumsum()
    non_excluded_df['mstrat_weighted_ror'] = non_excluded_df.apply(calculate_mstrat_weighted_ror, axis=1)
    if record_progress:
        progress_recorder.set_progress(80, 100)
    excluded_df['non_excluded_pct_aum_cumsum'] = None
    excluded_df['weighted_nav_cumsum'] = None
    excluded_df['non_excluded_pct_aum'] = None
    excluded_df['curr_rtn_wt_duration'] = None
    excluded_df['curr_rwd_ror'] = None
    excluded_df['curr_cwd_ror'] = None
    excluded_df['mstrat_weighted_ror'] = None
    merge_df = pd.concat([excluded_df, non_excluded_df])
    final_df = merge_df[merge_df['Fund'] == 'TACO'].sort_values(by=['catalyst_rating', 'ann_ror'],
                                                                ascending=[True, False]).reset_index(drop=True)
    if record_progress:
        progress_recorder.set_progress(90, 100)
    final_df = final_df[['tradegroup', 'sleeve', 'catalyst', 'catalyst_rating', 'target_last_price', 'px_ask_price',
                         'DealUpside', 'Coupon', 'closing_date', 'days_to_close', 'gross_ror', 'ann_ror', 'hedge_ror',
                         'risk_pct_unhedged', 'nav_impact', 'pct_of_sleeve_current', 'm_strat_pct_aum', 'is_excluded',
                         'weighted_gross_nav_potential', 'non_excluded_pct_aum', 'curr_rtn_wt_duration', 'curr_rwd_ror',
                         'curr_cwd_ror', 'mstrat_weighted_ror', 'target_ticker', 'expected_vol', 'notes', 'deal_status',
                         'taco_pct_risk', 'arb_pct_risk', 'arb_pct_aum', 'rebal_target', 'rebal_multiples']]
    final_df.rename(columns={'DealUpside': 'deal_upside', 'Coupon': 'coupon'}, inplace=True)
    final_df['date_updated'] = datetime.datetime.now().date()
    if not final_df.empty:
        if len(final_df.tradegroup.unique()) != len(final_df.tradegroup):
            slack_message('eze_uploads.slack',
                          {'null_risk_limits': str("_(Risk Automation)_ <@ssuizhu> <@akubal>" \
                                                   "*ERROR in Credit HardOpt* Duplicate tradegroups")},
                          channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN)
            dbutils.add_task_record(status="Error in Credit Hard Opt: Duplicate tradegroups")
            return 'Error in Credit Hard Opt: Duplicate tradegroups'
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()
    CreditHardFloatOptimization.objects.filter(date_updated=datetime.datetime.now().date()).delete()
    final_df.to_sql(name='portfolio_optimization_credithardfloatoptimization', schema=settings.CURRENT_DATABASE,
                    if_exists='append', index=False, con=con)
    con.close()
    if post_to_slack:
        slack_message('eze_uploads.slack',
                      {'null_risk_limits': str("_(Risk Automation)_ CREDIT Hard Catalyst Optimization Completed... "
                                               "Visit http://192.168.0.16:8000/portfolio_optimization/credit_hard_optimization")},
                      channel=get_channel_name('portal_downsides'), token=settings.SLACK_TOKEN)
    if record_progress:
        progress_recorder.set_progress(100, 100)
    dbutils.add_task_record()


@shared_task(bind=True)
def arb_hard_float_optimization(self, post_to_slack=True, record_progress=False):
    """ Purpose of this task is to take Hard M&A Deals in ARB and list scenarios to show Firm % of Float if Mstarts
        get to 1x AUM of ARB and 2x AUM of ARB Fund. Run this task after ARB Rate of Returns task... """

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return


    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                               + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()
    if record_progress:
        progress_recorder = ProgressRecorder(self)
        progress_recorder.set_progress(10, 100)
    try:
        api_host = bbgclient.bbgclient.get_next_available_host()
        max_date = "(SELECT MAX(date_updated) from "+settings.CURRENT_DATABASE + \
                   ".portfolio_optimization_arboptimizationuniverse)"  # RoRs
        comments_df_query = "SELECT tradegroup, notes, rebal_multiples, rebal_target, rebal_multiples_taq, " \
                            "rebal_target_taq, is_excluded FROM "+settings.CURRENT_DATABASE + \
                            ".portfolio_optimization_hardfloatoptimization WHERE date_updated = " \
                            "(SELECT MAX(date_updated) FROM " + settings.CURRENT_DATABASE + \
                            ".portfolio_optimization_hardfloatoptimization)"

        comments_df = pd.read_sql_query(comments_df_query, con=con)   # Comments & Rebal Mult,Target

        manually_excluded_df = comments_df[comments_df['is_excluded'] == 1]
        arb_df = pd.read_sql_query("SELECT * FROM "+settings.CURRENT_DATABASE+".portfolio_optimization_arboptimizationuniverse WHERE "
                                   "sleeve = 'Merger Arbitrage' and date_updated="+max_date, con=con)

        cols_to_work_on = ['tradegroup', 'sleeve', 'catalyst', 'catalyst_rating', 'closing_date', 'target_last_price',
                           'deal_value', 'all_in_spread', 'days_to_close', 'gross_ror', 'ann_ror', 'risk_pct', 'risk_pct_unhedged',
                           'expected_vol', 'deal_status']

        arb_df = arb_df[cols_to_work_on]

        excluded_deals_query = "SELECT TradeGroup as tradegroup from "+settings.CURRENT_DATABASE+\
                               ".risk_reporting_formulaebaseddownsides where IsExcluded='Yes'"
        excluded_deals_df = pd.read_sql_query(excluded_deals_query, con=con)

        arb_df = arb_df[~(arb_df['tradegroup'].isin(excluded_deals_df['tradegroup'].unique()))]

        shares_query = "SELECT TradeGroup,Target_Ticker,TargetLastPrice/FXCurrentLocalToBase as TargetLastPrice, Fund, " \
                       "SUM(amount*factor) AS TotalQty, aum, " \
                       "100*(D_Exposure)/aum " \
                       "AS Current_Pct_ofAUM FROM wic.daily_flat_file_db WHERE " \
                       "Flat_file_as_of = (SELECT MAX(flat_file_as_of) FROM wic.daily_flat_file_db) AND " \
                       "CatalystTypeWIC = 'HARD' AND amount<>0 AND SecType IN ('EQ', 'EQSWAP') AND AlphaHedge ='Alpha' AND " \
                       "LongShort='Long' AND TradeGroup IS NOT NULL AND Fund IN " \
                       "('ARB', 'MACO', 'MALT', 'CAM', 'LG', 'LEV', 'AED', 'EVNT', 'PRELUDE')  and ticker not like %s " \
                       "GROUP BY TradeGroup, Fund, aum;"

        current_shares_df = pd.read_sql_query(shares_query, con=con, params=('% CVR',))

        current_shares_df['Target_Ticker'] = current_shares_df['Target_Ticker'].apply(
            lambda x: (re.sub(' +', ' ', x.upper().replace('SWAP', '') + ' EQUITY')) if pd.notna(x) else x)

        arb_shares_df = current_shares_df[current_shares_df['Fund'] == 'ARB']  # Slice shares for ARB
        arb_shares_df.rename(columns={'Current_Pct_ofAUM':'ARB_Pct_of_AUM'}, inplace=True)
        progress_recorder.set_progress(30, 100) if record_progress else None
        # Slice AED and LG separately
        aed_shares_df = current_shares_df[current_shares_df['Fund'] == 'AED']
        taq_shares_df = current_shares_df[current_shares_df['Fund'] == 'EVNT']
        aed_aum = aed_shares_df['aum'].unique()[0]
        taq_aum = taq_shares_df['aum'].unique()[0]

        # Following Inline function returns Multistrat Ration w.r.t ARB
        def get_mstrat_quantity(row, fund):
            if fund == 'AED':
                aum = aed_aum
            else:
                aum = taq_aum
            quantity = (0.01*row['ARB_Pct_of_AUM']*aum)/row['TargetLastPrice']
            return quantity

        # Get the Quantity if MStarts asuume 1x AUM of ARB (in their respective Funds)
        arb_shares_df['AED Qty 1x'] = arb_shares_df.apply(lambda x: get_mstrat_quantity(x, 'AED'), axis=1)
        arb_shares_df['TAQ Qty 1x'] = arb_shares_df.apply(lambda x: get_mstrat_quantity(x, 'TAQ'), axis=1)

        # Get 2x quantitity of Mstrats go to 2x
        arb_shares_df['AED Qty 2x'] = arb_shares_df['AED Qty 1x'].apply(lambda x: x*2)
        arb_shares_df['TAQ Qty 2x'] = arb_shares_df['TAQ Qty 1x'].apply(lambda x: x*2)

        # Get Current Firmwide Quantity of Shares excluding Mstrats (TAQ and AED)
        current_qty = current_shares_df[~(current_shares_df['Fund'].isin(['AED', 'TAQ']))][['TradeGroup', 'TotalQty']]

        # Add 1x and 2x Quantity Columns
        aed_1x_shares = arb_shares_df[['TradeGroup', 'AED Qty 1x']]
        taq_1x_shares = arb_shares_df[['TradeGroup', 'TAQ Qty 1x']]
        aed_2x_shares = arb_shares_df[['TradeGroup', 'AED Qty 2x']]
        taq_2x_shares = arb_shares_df[['TradeGroup', 'TAQ Qty 2x']]

        # Rename
        aed_1x_shares.columns = ['TradeGroup', 'TotalQty']
        taq_1x_shares.columns = ['TradeGroup', 'TotalQty']
        aed_2x_shares.columns = ['TradeGroup', 'TotalQty']
        taq_2x_shares.columns = ['TradeGroup', 'TotalQty']

        shares_1x = pd.concat([current_qty, aed_1x_shares, taq_1x_shares])
        shares_2x = pd.concat([current_qty, aed_2x_shares, taq_2x_shares])

        # Get Firmwide Shares if Mstart go to 1x and 2x
        firmwide_shares_1x = shares_1x.groupby('TradeGroup').sum().reset_index()
        firmwide_shares_2x = shares_2x.groupby('TradeGroup').sum().reset_index()

        firmwide_shares_1x.columns = ['TradeGroup', 'TotalQty_1x']
        firmwide_shares_2x.columns = ['TradeGroup', 'TotalQty_2x']
        firmwide_current_shares = current_shares_df[['TradeGroup', 'Target_Ticker', 'TotalQty']].\
            groupby(['TradeGroup', 'Target_Ticker']).sum().reset_index()

        all_shares = pd.merge(firmwide_current_shares, firmwide_shares_1x, on='TradeGroup')
        all_shares = pd.merge(all_shares, firmwide_shares_2x, on='TradeGroup')

        unique_target_tickers = list(all_shares['Target_Ticker'].unique())

        # Get the Current Float for all tickers
        current_floats = bbgclient.bbgclient.get_secid2field(unique_target_tickers,'tickers',
                                                              ['EQY_FLOAT'], req_type='refdata', api_host=api_host)

        all_shares['FLOAT'] = all_shares['Target_Ticker'].apply(lambda x:
                                                                parse_fld(current_floats,'EQY_FLOAT', x)
                                                                if not pd.isnull(x) else None)
        all_shares['FLOAT'] = all_shares['FLOAT'] * 1000000

        # Calculates the Current % of Float for current portfolio, 1x Mstart and 2x Mstrat
        all_shares['Current % of Float'] = 1e2*(all_shares['TotalQty']/all_shares['FLOAT'])

        all_shares['Firm % of Float if Mstart 1x'] = 1e2*(all_shares['TotalQty_1x']/all_shares['FLOAT'])
        all_shares['Firm % of Float if Mstart 2x'] = 1e2*(all_shares['TotalQty_2x']/all_shares['FLOAT'])

        taq_current_shares = current_shares_df[current_shares_df['Fund'].isin(['TAQ'])]
        aed_current_shares = current_shares_df[current_shares_df['Fund'].isin(['AED'])]
        arb_current_shares = current_shares_df[current_shares_df['Fund'].isin(['ARB'])]
        progress_recorder.set_progress(50, 100) if record_progress else None
        # Below function to get the AUM Multiplier i.e times of ARBs AUM we currenly in AED and TAQ

        def get_aum_multiplier(row, fund):
            aum_multiplier = 0
            df_arb = arb_current_shares[arb_current_shares['TradeGroup'] == row['TradeGroup']]
            if len(df_arb) == 0:
                return np.NAN

            if fund == 'TAQ':
                df_ = taq_current_shares[taq_current_shares['TradeGroup'] == row['TradeGroup']]
                if len(df_) > 0:
                    aum_multiplier = df_['Current_Pct_ofAUM'].iloc[0]/df_arb['Current_Pct_ofAUM'].iloc[0]
            else:
                df_ = aed_current_shares[aed_current_shares['TradeGroup'] == row['TradeGroup']]
                if len(df_) > 0:
                    aum_multiplier = df_['Current_Pct_ofAUM'].iloc[0]/df_arb['Current_Pct_ofAUM'].iloc[0]

            return aum_multiplier

        all_shares['AED AUM Mult'] = all_shares.apply(lambda x: get_aum_multiplier(x, 'AED'), axis=1)
        all_shares['TAQ AUM Mult'] = all_shares.apply(lambda x: get_aum_multiplier(x, 'TAQ'), axis=1)

        # Get % of AUMs
        def get_aed_pct_of_aum(row, fund):
            if fund == 'ARB':
                aum_df_ = arb_current_shares
            elif fund == 'EVNT':
                aum_df_ = taq_shares_df
            else:
                aum_df_ = aed_shares_df
            return_value = 0
            aum_ = aum_df_[aum_df_['TradeGroup'] == row['TradeGroup']]
            if len(aum_) > 0:
                return aum_['Current_Pct_ofAUM'].iloc[0]
            return return_value

        all_shares['aed_pct_of_aum'] = all_shares.apply(lambda x: get_aed_pct_of_aum(x, 'AED'), axis=1)
        all_shares['taq_pct_of_aum'] = all_shares.apply(lambda x: get_aed_pct_of_aum(x, 'TAQ'), axis=1)
        all_shares['arb_pct_of_aum'] = all_shares.apply(lambda x: get_aed_pct_of_aum(x, 'ARB'), axis=1)

        all_shares.columns = ['tradegroup', 'target_ticker', 'total_qty', 'total_qty_1x', 'total_qty_2x', 'eqy_float',
                              'current_pct_of_float', 'firm_pct_float_mstrat_1x', 'firm_pct_float_mstrat_2x',
                              'aed_aum_mult', 'taq_aum_mult', 'aed_pct_of_aum', 'taq_pct_of_aum', 'arb_pct_of_aum']


        # Merge ARB_DF (Rate of Returns with Float DF)
        final_hard_opt_df = pd.merge(arb_df, all_shares, how='left', on='tradegroup')

        final_hard_opt_df['date_updated'] = datetime.datetime.now().date()

        # Delete unwanted columns
        final_hard_opt_df = pd.merge(final_hard_opt_df, comments_df,how='left', on='tradegroup')

        # Get the NAV Impacts for Risk Multiples
        nav_impacts_query = "Select Tradegroup as tradegroup, OUTLIER_NAV_IMPACT_ARB as arb_outlier_risk, " \
                            "OUTLIER_NAV_IMPACT_AED as aed_outlier_risk, OUTLIER_NAV_IMPACT_TAQ as taq_outlier_risk " \
                            " FROM " + settings.CURRENT_DATABASE + ".risk_reporting_dailynavimpacts"
        #
        # nav_impacts_query = "SELECT TradeGroup tradegroup, SUM(OUTLIER_NAV_IMPACT_ARB) AS arb_outlier_risk, "\
        #                     "SUM(OUTLIER_NAV_IMPACT_AED) AS aed_outlier_risk, "\
        #                     "SUM(OUTLIER_NAV_IMPACT_LG) AS lg_outlier_risk FROM "\
        #                     + settings.CURRENT_DATABASE + ".risk_reporting_positionlevelnavimpacts GROUP BY TradeGroup"

        nav_impacts_df = pd.read_sql_query(nav_impacts_query, con=con)
        numeric_cols = ['arb_outlier_risk', 'aed_outlier_risk', 'taq_outlier_risk']
        nav_impacts_df[numeric_cols] = nav_impacts_df[numeric_cols].apply(pd.to_numeric)
        final_hard_opt_df = pd.merge(final_hard_opt_df, nav_impacts_df, on='tradegroup', how='left')

        final_hard_opt_df['aed_risk_mult'] = final_hard_opt_df['aed_outlier_risk'] / final_hard_opt_df['arb_outlier_risk']

        final_hard_opt_df['taq_risk_mult'] = final_hard_opt_df['taq_outlier_risk'] / final_hard_opt_df['arb_outlier_risk']
        progress_recorder.set_progress(70, 100) if record_progress else None
        HardFloatOptimization.objects.filter(date_updated=datetime.datetime.now().date()).delete()

        # Adjust for the Rebal Targets
        final_hard_opt_df['rebal_target'] = final_hard_opt_df.apply(lambda x:
                                                                    np.round((x['rebal_multiples'] * x['arb_pct_of_aum']
                                                                              ), decimals=2) if not
                                                                    pd.isna(x['rebal_multiples'])
                                                                    else x['aed_pct_of_aum'], axis=1)

        final_hard_opt_df['rebal_target_taq'] = final_hard_opt_df.apply(lambda x:
                                                                    np.round((x['rebal_multiples_taq'] * x['arb_pct_of_aum']
                                                                              ), decimals=2) if not
                                                                    pd.isna(x['rebal_multiples_taq'])
                                                                    else x['taq_pct_of_aum'], axis=1)

        # Section for Time Weighted Rate of Return Calculation...
        def exclude_from_ror(row):
            if row['tradegroup'] in list(manually_excluded_df['tradegroup'].unique()):
                return True
            else:
                if ((row['days_to_close'] < 22) and (row['all_in_spread'] < 0.1)) or (row['days_to_close'] < 6):
                    return True

            return False

        final_hard_opt_df = final_hard_opt_df.sort_values(by=['catalyst', 'catalyst_rating', 'ann_ror'],
                                                          ascending=[True, True, False])
        final_hard_opt_df['is_excluded'] = final_hard_opt_df.apply(exclude_from_ror, axis=1)

        final_hard_opt_df_excluded = final_hard_opt_df[final_hard_opt_df['is_excluded'] == True]
        final_hard_opt_df = final_hard_opt_df[final_hard_opt_df['is_excluded'] == False]

        final_hard_opt_df['weighted_gross_nav_potential'] = (final_hard_opt_df['gross_ror'] * final_hard_opt_df['rebal_target'])/100

        final_hard_opt_df['weighted_gross_nav_potential_taq'] = (final_hard_opt_df['gross_ror'] * final_hard_opt_df['rebal_target_taq'])/100

        # Calculate the Cumulative Sum
        final_hard_opt_df['weighted_nav_cumsum'] = final_hard_opt_df['weighted_gross_nav_potential'].cumsum()
        final_hard_opt_df['weighted_nav_cumsum_taq'] = final_hard_opt_df['weighted_gross_nav_potential_taq'].cumsum()
        # Get the Mstrat % of AUM
        final_hard_opt_df['non_excluded_pct_aum_cumsum'] = final_hard_opt_df['rebal_target'].cumsum()
        final_hard_opt_df['non_excluded_pct_aum_cumsum_taq'] = final_hard_opt_df['rebal_target_taq'].cumsum()

        # Get the Current Rtn Weight Duration
        final_hard_opt_df["curr_rtn_wt_duration"] = final_hard_opt_df['days_to_close'].mul(final_hard_opt_df['weighted_gross_nav_potential']).cumsum().div(final_hard_opt_df['weighted_gross_nav_potential'].cumsum())

        final_hard_opt_df["curr_rtn_wt_duration_taq"] = final_hard_opt_df['days_to_close'].\
            mul(final_hard_opt_df['weighted_gross_nav_potential_taq']).cumsum().\
            div(final_hard_opt_df['weighted_gross_nav_potential_taq'].cumsum())

        # Get the RWD, ROR
        final_hard_opt_df['curr_rwd_ror'] = final_hard_opt_df['weighted_nav_cumsum']/final_hard_opt_df['non_excluded_pct_aum_cumsum']/\
                                    final_hard_opt_df['curr_rtn_wt_duration']*360
        final_hard_opt_df['curr_rwd_ror'] = 1e2 * final_hard_opt_df['curr_rwd_ror']

        # TAQ
        final_hard_opt_df['curr_rwd_ror_taq'] = final_hard_opt_df['weighted_nav_cumsum_taq']/final_hard_opt_df['non_excluded_pct_aum_cumsum_taq']/\
                                    final_hard_opt_df['curr_rtn_wt_duration_taq']*360
        final_hard_opt_df['curr_rwd_ror_taq'] = 1e2 * final_hard_opt_df['curr_rwd_ror_taq']

        # Get the Capital Weighted Return (Calculated the weighted Average (x1y1 + x2y2 + ...+ xnyn)/(y1+y2+..+yn)

        final_hard_opt_df["curr_cwd_ror"] = final_hard_opt_df['ann_ror'].\
            mul(final_hard_opt_df['non_excluded_pct_aum_cumsum']).cumsum().\
            div(final_hard_opt_df['non_excluded_pct_aum_cumsum'].cumsum())

        final_hard_opt_df["curr_cwd_ror_taq"] = final_hard_opt_df['ann_ror'].\
            mul(final_hard_opt_df['non_excluded_pct_aum_cumsum_taq']).cumsum().\
            div(final_hard_opt_df['non_excluded_pct_aum_cumsum_taq'].cumsum())

        # Calculate the Weighted RoR from Capital weighted & time weighted
        final_hard_opt_df['aed_weighted_ror'] = final_hard_opt_df[['curr_rwd_ror', 'curr_cwd_ror']].mean(axis=1)
        final_hard_opt_df['taq_weighted_ror'] = final_hard_opt_df[['curr_rwd_ror_taq', 'curr_cwd_ror_taq']].\
            mean(axis=1)

        #  Merge back Excluded deals after adding new columns
        final_hard_opt_df_excluded['weighted_nav_cumsum'] = None
        final_hard_opt_df_excluded['non_excluded_pct_aum_cumsum'] = None
        final_hard_opt_df_excluded['curr_rtn_wt_duration'] = None
        final_hard_opt_df_excluded['curr_rwd_ror'] = None

        final_hard_opt_df_excluded['weighted_nav_cumsum_taq'] = None
        final_hard_opt_df_excluded['non_excluded_pct_aum_cumsum_taq'] = None
        final_hard_opt_df_excluded['curr_rtn_wt_duration_taq'] = None
        final_hard_opt_df_excluded['curr_rwd_ror_taq'] = None

        final_hard_opt_df = pd.concat([final_hard_opt_df, final_hard_opt_df_excluded])

        # Replace Infinity values
        progress_recorder.set_progress(85, 100) if record_progress else None
        final_hard_opt_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace Inf values

        if not final_hard_opt_df.empty:
            if len(final_hard_opt_df.tradegroup.unique()) != len(final_hard_opt_df.tradegroup):
                slack_message('eze_uploads.slack',
                              {'null_risk_limits': str("_(Risk Automation)_ <@ssuizhu> <@akubal>" \
                                                       "*ERROR in ARB Hard Opt* Duplicate tradegroups")},
                              channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN)
                return 'Error in Arb Hard Opt: Duplicate tradegroups'
        HardFloatOptimization.objects.filter(date_updated=datetime.datetime.now().date()).delete()
        final_hard_opt_df.to_sql(name='portfolio_optimization_hardfloatoptimization', schema=settings.CURRENT_DATABASE,
                                 if_exists='append', index=False, con=con)

        hard_optimized_summary(post_to_slack)
        if post_to_slack:
            slack_message('eze_uploads.slack', {'null_risk_limits':
                                                str("_(Risk Automation)_ ARB Hard Catalyst Optimization Completed... "
                                                    "Visit http://192.168.0.16:8000/portfolio_optimization/arb_hard_optimization"
                                                    )},
                          channel=get_channel_name('portal_downsides'), token=settings.SLACK_TOKEN)

        progress_recorder.set_progress(100, 100) if record_progress else None
        dbutils.add_task_record()
    except Exception as e:
        import traceback
        traceback.print_exc()
        if post_to_slack:
            slack_message('eze_uploads.slack', {'null_risk_limits':
                                                str("_(Risk Automation)_ *ERROR in HardOpt!*... ") + str(e)
                                                },
                          channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN)
        dbutils.add_task_record(status=e)
    finally:
        print('Closing Connection to Relational Database Service...')
        con.close()


@shared_task
def hard_optimized_summary(post_to_slack=True):
    """ Task runs after the above task is completed. Creates a Summary of the Hard-Optimized schedule """
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                               + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()
    try:
        #  Get current data
        current_invested_query = "SELECT TradeGroup, Fund, CurrentMktVal_Pct, Sleeve FROM wic.daily_flat_file_db "\
                                 "WHERE Flat_file_as_of = (SELECT MAX(flat_file_as_of) FROM wic.daily_flat_file_db) "\
                                 "AND Fund IN ('ARB', 'AED', 'TAQ') AND LongShort='Long' AND amount<>0 AND " \
                                 "Ticker NOT LIKE '%%CASH%%' AND AlphaHedge IN ('Alpha')"
        current_invested_whole_fund = pd.read_sql_query(current_invested_query, con=con)
        current_invested_arb = current_invested_whole_fund[current_invested_whole_fund['Sleeve'] == 'Merger Arbitrage']

        arb_no_of_deals = current_invested_arb[current_invested_arb['Fund'] == 'ARB']['TradeGroup'].nunique()
        arb_pct_invested = current_invested_arb[current_invested_arb['Fund'] == 'ARB']['CurrentMktVal_Pct'].sum()

        aed_no_of_deals = current_invested_arb[current_invested_arb['Fund'] == 'AED']['TradeGroup'].nunique()
        aed_pct_invested = current_invested_arb[current_invested_arb['Fund'] == 'AED']['CurrentMktVal_Pct'].sum()

        taq_no_of_deals = current_invested_arb[current_invested_arb['Fund'] == 'TAQ']['TradeGroup'].nunique()
        taq_pct_invested = current_invested_arb[current_invested_arb['Fund'] == 'TAQ']['CurrentMktVal_Pct'].sum()

        # Summary based on Hard Optimization and Rebalanced Targets
        rebal_query = "SELECT * FROM "+ settings.CURRENT_DATABASE+".portfolio_optimization_hardfloatoptimization " \
                      "WHERE date_updated = (SELECT MAX(date_updated) FROM " + settings.CURRENT_DATABASE + \
                      ".portfolio_optimization_hardfloatoptimization)"

        rebal_query_df = pd.read_sql_query(rebal_query, con=con)
        # Remove Excluded ones
        rebal_query_df = rebal_query_df[rebal_query_df['is_excluded'] == False]
        # Get Hard-1 Optimized RoRs
        hard_one_df = rebal_query_df[((rebal_query_df['catalyst'] == 'HARD') & (rebal_query_df['catalyst_rating'] == '1'))]
        average_optimized_rors = hard_one_df['ann_ror'].mean()

        weighted_arb_ror = (hard_one_df['ann_ror']*hard_one_df['arb_pct_of_aum']).sum() * 0.01
        weighted_aed_ror = rebal_query_df['aed_weighted_ror'].min()

        weighted_taq_ror = rebal_query_df['taq_weighted_ror'].min()

        # Following metrics based on the Adjustable column
        hard_aed = rebal_query_df[rebal_query_df['catalyst'] == 'HARD']
        hard_aed_pct_invested = hard_aed['rebal_target'].sum()
        hard_taq_pct_invested = hard_aed['rebal_target_taq'].sum()

        # Get the whole AED Fund % invested based on the Rebalanced Targets

        aed_whole_fund = current_invested_whole_fund[current_invested_whole_fund['Fund'] == 'AED']
        taq_whole_fund = current_invested_whole_fund[current_invested_whole_fund['Fund'] == 'TAQ']

        # aed_df = aed_whole_fund[~aed_whole_fund['TradeGroup'].isin(rebal_query_df['tradegroup'].unique())]
        #
        # rebalanced_aed_pct_invested = rebal_query_df[['tradegroup', 'rebal_target']]
        # rebalanced_aed_pct_invested['Fund'] = 'AED'
        # rebalanced_aed_pct_invested.rename(columns={'rebal_target': 'CurrentMktVal_Pct', 'tradegroup': 'TradeGroup'},
        #                                    inplace=True)
        #
        # # Concatenate the 2 dataframes
        # aed_rebalanced_df = pd.concat([rebalanced_aed_pct_invested, aed_df])

        aed_fund_pct_invested_rebalanced = aed_whole_fund['CurrentMktVal_Pct'].sum()
        taq_fund_pct_invested_rebalanced = taq_whole_fund['CurrentMktVal_Pct'].sum()
        now = datetime.datetime.now().date()
        HardOptimizationSummary.objects.filter(date_updated=now).delete()

        HardOptimizationSummary(date_updated=now, average_optimized_rors=average_optimized_rors,
                                weighted_arb_rors=weighted_arb_ror, weighted_aed_ror=weighted_aed_ror,
                                arb_number_of_deals=arb_no_of_deals, arb_pct_invested=arb_pct_invested,
                                aed_number_of_deals=aed_no_of_deals, aed_currently_invested=aed_pct_invested,
                                aed_hard_pct_invested=hard_aed_pct_invested,
                                aed_fund_pct_invested=aed_fund_pct_invested_rebalanced,
                                weighted_taq_ror=weighted_taq_ror, taq_number_of_deals=taq_no_of_deals,
                                taq_currently_invested=taq_pct_invested, taq_hard_pct_invested=hard_taq_pct_invested,
                                taq_fund_pct_invested=taq_fund_pct_invested_rebalanced).save()

    except Exception as e:
        print(e)
        if post_to_slack:
            slack_message('eze_uploads.slack', {'null_risk_limits':
                                                str("_(Risk Automation)_ *ERROR in HardOpt (Summary)!*... ") + str(e)
                                                },
                          channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN)
    finally:
        con.close()
        print('Closing connection to Relational Database Service')


@shared_task
def update_pnl_potentials():
    """ Task to Save the current PnL Potential in the Database and Refresh the PNL Potentials
        by looking at the deals in the portfolio with closing date always greater than today and less than end_date
        in Pnl Potential Dates Model
    """

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)

    con = engine.connect()
    try:
        df_dict, dict_values = format_data()
        scenario_response_df = df_dict.get('scenario_response_df')
        scenario_processing_df = df_dict.get('scenario_processing_df')
        ess_achievement_returns_df = df_dict.get('ess_achievement_returns_df')
        aed_ess_df = df_dict.get('aed_ess_df')
        aed_df = df_dict.get('aed_df')
        aed_credit_df = df_dict.get('aed_credit_df')
        ess_required_return = dict_values.get('ess_required_return')
        implied_prob_deduct = dict_values.get('implied_prob_deduct')
        now_date = datetime.datetime.now().date().strftime('%Y-%m-%d')
        aed_ess_df.drop(columns=['sleeve'], inplace=True)
        aed_ess_df.rename(columns={'AED NAV': 'aed_nav', 'Current MktVal %': 'current_mkt_val_pct',
                                   'Tradegroup': 'tradegroup', 'PT UP': 'pt_up', 'PT WIC': 'pt_wic',
                                   'PT Down': 'pt_down', 'Model Up': 'model_up', 'Model WIC': 'model_wic',
                                   'Model Down': 'model_down', 'Alpha Ticker': 'alpha_ticker', 'PX LAST':'px_last',
                                   'Pnl Potential 100%': 'pnl_potential_100', 'Pnl Potential 50%': 'pnl_potential_50',
                                   'Pnl Potential 0%': 'pnl_potential_0'}, inplace=True)

        aed_ess_df['date_updated'] = now_date
        PnlPotentialESSConstituentsHistory.objects.filter(date_updated=now_date).delete()
        aed_ess_df.to_sql(con=con, name='portfolio_optimization_pnlpotentialessconstituentshistory', index=False,
                          schema=settings.CURRENT_DATABASE, if_exists='append')

        # Process the Summary DataFrame
        valid_scenarios = [x for x in scenario_response_df.columns.values if x not in ['Index', 'sleeve']]
        summary_data = []
        for scenario in valid_scenarios:
            temp_df = scenario_response_df[['Index', 'sleeve', scenario]]
            for row in temp_df.iterrows():
                slv_to_insert = row[1]['sleeve']
                scn_to_insert = scenario
                cut_to_insert = row[1]['Index']
                value_to_insert = row[1][scenario]
                summary_data.append([now_date, slv_to_insert, scn_to_insert, cut_to_insert, value_to_insert])

            # Repeat for Scenario Processing Dataframe
            temp_df = scenario_processing_df[['Index', 'sleeve', scenario]]
            for row in temp_df.iterrows():
                slv_to_insert = row[1]['sleeve']
                scn_to_insert = scenario
                cut_to_insert = row[1]['Index']
                value_to_insert = row[1][scenario]
                summary_data.append([now_date, slv_to_insert, scn_to_insert, cut_to_insert, value_to_insert])

            # Repeat for ESS Processing DF
            temp_df = ess_achievement_returns_df[['Index', 'sleeve', scenario]]
            for row in temp_df.iterrows():
                slv_to_insert = row[1]['sleeve']
                scn_to_insert = scenario
                cut_to_insert = row[1]['Index']
                value_to_insert = row[1][scenario]
                summary_data.append([now_date, slv_to_insert, scn_to_insert, cut_to_insert, value_to_insert])

        summary_df = pd.DataFrame(columns=['date_updated', 'sleeve', 'scenario_name', 'cut_name', 'value'],
                                  data=summary_data)

        summary_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        PnlPotentialDailySummary.objects.filter(date_updated=now_date).delete()
        summary_df.to_sql(con=con, name='portfolio_optimization_pnlpotentialdailysummary', index=False,
                          schema=settings.CURRENT_DATABASE, if_exists='append')

        # Store the Historical Consitutnents

        bulk_create(PnlPotentialScenarios, PnlPotentialScenariosHistory, now_date)
        bulk_create(PnlPotentialIncremental, PnlPotentialIncrementalHistory, now_date)
        bulk_create(PnlPotentialExclusions, PnlPotentialExclusionsHistory, now_date)
        bulk_create(PnlPotentialDate, PnlPotentialDateHistory, now_date)
        bulk_create(ArbCreditPnLPotentialDrilldown, ArbCreditPnLPotentialDrilldownHistory, now_date)

        slack_message('eze_uploads.slack',
                      {'null_risk_limits': str("_(Risk Automation)_ PnL Potentials (Summary) ran successfully on " + \
                                           now_date)},
                      channel=get_channel_name('portal-task-reports'), token=settings.SLACK_TOKEN)
        dbutils.add_task_record()
    except Exception as e:
        slack_message('eze_uploads.slack', {'null_risk_limits':
                                            str("_(Risk Automation)_ *ERROR in PnL Potentials (Summary)!*... ") + str(e)
                                            },
                      channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN)
        dbutils.add_task_record(status=e)
    finally:
        con.close()


def bulk_create(from_model, to_model, now_date):
    to_model.objects.filter(date_updated=now_date).delete()
    queryset = (from_model.objects.all().values())
    for obj in queryset: del obj['id']
    new_objects = [to_model(**values) for values in queryset]
    to_model.objects.bulk_create(new_objects)


@shared_task
def update_pnl_ess_constituents():
    """ Task to Save the current PnL Potential in the Database and Refresh the PNL Potentials
        by looking at the deals in the portfolio with closing date always greater than today and less than end_date
        in Pnl Potential Dates Model
    """

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    update_pnl_ess_constituents_function()


def update_pnl_ess_constituents_function(post_to_slack=True):
    """
    This function is also used in views for Pnl Potential Page for refreshing the ESS values whenever a new ESS Idea
    is added.
    Return response only when the function is called from the view.
    If called from task, do not return anything but post on Slack.
    """
    response = 'failed'
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)

    con = engine.connect()
    try:
        df_dict, dict_values = format_data()
        aed_ess_df = df_dict.get('aed_ess_df')
        now_date = datetime.datetime.now().date().strftime('%Y-%m-%d')
        aed_ess_df.drop(columns=['sleeve'], inplace=True)
        aed_ess_df.rename(columns={'AED NAV': 'aed_nav', 'Current MktVal %': 'current_mkt_val_pct',
                                   'Tradegroup': 'tradegroup', 'PT UP': 'pt_up', 'PT WIC': 'pt_wic',
                                   'PT Down': 'pt_down', 'Model Up': 'model_up', 'Model WIC': 'model_wic',
                                   'Model Down': 'model_down', 'Alpha Ticker': 'alpha_ticker', 'PX LAST':'px_last',
                                   'Pnl Potential 100%': 'pnl_potential_100', 'Pnl Potential 50%': 'pnl_potential_50',
                                   'Pnl Potential 0%': 'pnl_potential_0'}, inplace=True)

        PnlPotentialESSConstituents.objects.all().delete()
        aed_ess_df.to_sql(con=con, name='portfolio_optimization_pnlpotentialessconstituents', index=False,
                          schema=settings.CURRENT_DATABASE, if_exists='append')
        if post_to_slack:
            slack_message('eze_uploads.slack',
                          {'null_risk_limits':
                           str("_(Risk Automation)_ PnL Potentials ESS Constituents updated on " + now_date)},
                          channel=get_channel_name('portal-task-reports'), token=settings.SLACK_TOKEN)
        response = 'success'
        dbutils.add_task_record(task_name='portfolio_optimization.tasks.update_pnl_ess_constituents')
    except Exception as e:
        if post_to_slack:
            slack_message('eze_uploads.slack',
                          {'null_risk_limits':
                           str("_(Risk Automation)_ *ERROR in PnL Potentials ESS Constituents!*... ") + str(e)},
                          channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN)
        dbutils.add_task_record(task_name='portfolio_optimization.tasks.update_pnl_ess_constituents',status=e)
    finally:
        con.close()
        if not post_to_slack:
            return response


@shared_task
def update_pnl_potential_arb_credit_drilldown():
    """ This task is used for the custom position sizing for P&L potential. Adds new positions to the Drilldown
        Runs everyday at 9pm after flat file is consumed....Reports to portal_task_reports
    """

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    flag = 0
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD +
                           "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()
    old_df = pd.DataFrame()
    try:
        wic_universe_df = pd.read_sql_query("SELECT * FROM " + settings.CURRENT_DATABASE + \
                                            ".portfolio_optimization_arboptimizationuniverse WHERE date_updated = " \
                                            "(SELECT MAX(date_updated) FROM " + settings.CURRENT_DATABASE +
                                            ".portfolio_optimization_arboptimizationuniverse)", con=con)

        aed_df = pd.read_sql_query("SELECT AUM AS `aed_nav`, tradegroup, sleeve, CASE WHEN SecType = 'EQSWAP' THEN "
                                   "ROUND(SUM(100*(D_Exposure/AUM)),4) ELSE SUM(CurrentMktVal_Pct) END "
                                   "AS current_mkt_val_pct " \
                                   "FROM wic.daily_flat_file_db WHERE Flat_file_as_of = (SELECT MAX(Flat_file_as_of) "
                                   "FROM wic.daily_flat_file_db) AND fund LIKE 'AED' AND amount<>0 AND AlphaHedge "
                                   "LIKE 'Alpha' GROUP BY AUM, tradegroup, sleeve", con=con)

        present_arb_credit_df = pd.read_sql_query('SELECT * FROM ' + settings.CURRENT_DATABASE + \
                                                  '.portfolio_optimization_arbcreditpnlpotentialdrilldown',
                                                  con=con)
        present_arb_credit_df.drop_duplicates(inplace=True)
        old_df = present_arb_credit_df.copy()
        present_arb_credit_df = present_arb_credit_df[['tradegroup', 'sleeve', 'customized_mkt_val_pct',
                                                       'is_customized']]
        aed_df = pd.merge(aed_df, wic_universe_df, how='left', on=['tradegroup', 'sleeve'])
        aed_df['implied_probability'] = (1e2 * ((aed_df['target_last_price'].fillna(0) - aed_df['deal_downside'].fillna(0)) /
                                                (aed_df['deal_value'].fillna(0) - aed_df['deal_downside'].fillna(0)))).fillna(0)
        aed_df.replace([np.inf, -np.inf], 0, inplace=True)
        aed_df = aed_df[['aed_nav', 'tradegroup', 'sleeve', 'current_mkt_val_pct', 'id', 'bucket', 'catalyst',
                         'catalyst_rating', 'closing_date', 'long_short', 'target_last_price', 'deal_value',
                         'all_in_spread', 'deal_downside', 'days_to_close', 'pct_of_sleeve_current', 'gross_ror',
                         'implied_probability']]

        arb_credit_df = aed_df[aed_df['sleeve'].isin(['Merger Arbitrage', 'Credit Opportunities'])]
        # if present_df is empty then do not merge
        if not present_arb_credit_df.empty:
            arb_credit_df = pd.merge(arb_credit_df, present_arb_credit_df, how='left', on=['tradegroup', 'sleeve'])
        else:
            # assume custom position exactly same as current position
            arb_credit_df['customized_mkt_val_pct'] = arb_credit_df['current_mkt_val_pct']

        # Fill Customized for New positions
        arb_credit_df['customized_mkt_val_pct'].fillna(arb_credit_df['current_mkt_val_pct'], inplace=True)
        arb_credit_df['is_customized'].fillna(0, inplace=True)
        arb_credit_df['customized_mkt_val_pct'] = arb_credit_df.apply(
            lambda x: x['customized_mkt_val_pct'] if x['is_customized'] else x['current_mkt_val_pct'], axis=1)

        # Calculate the P&L Potentials
        arb_credit_df = arb_credit_df[~pd.isna(arb_credit_df['closing_date'])]
        arb_credit_df['closing_date'] = arb_credit_df['closing_date'].apply(pd.to_datetime, format="%Y-%m-%d")
        arb_credit_df['days_to_close'] = (arb_credit_df['closing_date'].dt.date -  datetime.date.today()).dt.days
        arb_credit_df['pnl_potential'] = arb_credit_df['customized_mkt_val_pct'] * \
                                         arb_credit_df['gross_ror'] * arb_credit_df['aed_nav'] / 10000

        ArbCreditPnLPotentialDrilldown.objects.all().delete()
        flag = 1
        # Insert updated dataframe
        arb_credit_df.to_sql(con=con, if_exists='append', schema=settings.CURRENT_DATABASE,
                             name='portfolio_optimization_arbcreditpnlpotentialdrilldown', index=False, chunksize=100)
        # post to slack
        slack_message('eze_uploads.slack', {'null_risk_limits':
                                            str("_(Risk Automation)_ *ARB & Credit P&L Potentials Updated*")},
                      channel=get_channel_name('portal-task-reports'),
                      token=settings.SLACK_TOKEN)
        dbutils.add_task_record()
    except Exception as e:
        print(e)
        if flag:
            old_df.to_sql(con=con, if_exists='append', schema=settings.CURRENT_DATABASE,
                          name='portfolio_optimization_arbcreditpnlpotentialdrilldown', index=False, chunksize=100)
            # post to slack
        slack_message('eze_uploads.slack',
                      {'null_risk_limits': str("_(Risk Automation)_ *ERROR in ARB & Credit P&L Potentials *... ")
                                           + "Restored to previous database." + str(e)},
                      channel=get_channel_name('portal-task-errors'),
                      token=settings.SLACK_TOKEN)
        dbutils.add_task_record(e)
    finally:
        con.close()
