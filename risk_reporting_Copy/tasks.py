# coding: utf-8
import datetime
import io
import logging
import traceback
from locale import atof
import os
import sys
import time

from django.contrib.contenttypes.models import ContentType

import bbgclient
from celery import shared_task
import django
from django import db
from django.conf import settings
from django.db import connections as django_db_connections, connection as django_db_connection, connection
from django_slack import slack_message
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import dbutils
import holiday_utils
from email_utilities import send_email, get_attachment_by_subject, copy_latest_file
from funds_snapshot.models import FundSleeveAttribution
from risk.mna_deal_bloomberg_utils import get_data_from_bloomberg_by_bg_id
from realtime_pnl_impacts import utils, views
from realtime_pnl_impacts.models import PnlMonitors
from risk.models import Downside_Trendlines, MA_Deals, MaDownsideRegression, MaDownsidePeerSource, \
    MaDownsideRegressionModel
from risk.tasks import get_peerdownside_by_tradegroup
from risk_reporting.update_credit_deals_tasks import update_credit_deals
from risk_reporting.models import (CreditDealsUpsideDownside, DailyNAVImpacts, PositionLevelNAVImpacts,
                                   FormulaeBasedDownsides, HistoricalFormulaeBasedDownsides, DealDownside,
                                   HistoricalDealDownside, LinearRegressionDownside)
from risk_reporting.deal_downside.downside_calculations import update_deal_downsides, update_downside_from_model
from .views import get_security_info_dataframe, get_deal_info_dataframe
from slack_utils import get_channel_name
from .RiskReportingCustomExceptions import TradeGroupMismatchException
from .ess_utils import get_ess_nav_impacts

import PyPDF2
import re

logger = logging.getLogger(__name__)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WicPortal_Django.settings")
django.setup()

SLEEVE_DICT = {
    'Credit Opportunities': 'CREDIT',
    'Equity Special Situations': 'ESS',
    'Merger Arbitrage': 'M&A',
    'Opportunistic': 'OPP',
    'Break': 'UNLISTED/CASH',
}

PNL_FUND_LIST = ['ARB', 'MALT', 'AED', 'LG', 'TACO', 'ARBETF', 'PRELUDE', 'EVNT']


def get_todays_date_yyyy_mm_dd():
    return datetime.datetime.now().date().strftime('%Y-%m-%d')


def remove_risk_limit_CVR_error(df):
    ''' filter the tradegroups with "Other MNA" tags from formulaeBasedDownsides & NAV due to price/risk_limit incorrect values
    output: list [  , ] or df'''
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)

    con = engine.connect()
    # check cvr and retrieve Last price from db
    ticker = '% CVR%'
    bucket = 'Other M&A'
    # query = 'SELECT TradeGroup FROM wic.daily_flat_file_db where Flat_file_as_of = (select max(flat_file_as_of) ' \
    #         'from wic.daily_flat_file_db) and Bucket = ' + bucket + ' AND Ticker like "' +ticker+ '" LIMIT 1;'
    flatfile_df = pd.read_sql_query('Select tradegroup from ' +
                                    'wic.daily_flat_file_db where flat_file_as_of = (select max(flat_file_as_of) from wic.daily_flat_file_db) AND ' +
                                    'bucket = "' + bucket + '" ',
                                    # 'AND ticker like "' +ticker+ '" ',
                                    con=connection)
    flatfile_error_list = list(flatfile_df['tradegroup'])
    if len(df):
        res = df[~df['TradeGroup'].isin(flatfile_error_list)]
    else:
        res = flatfile_error_list

    return res


def replace_CVR_price(con, ticker):
    '''filter out the wrong price for CVR tickers to better calc risk limit and downsides'''
    query = "SELECT LastPrice FROM " + settings.CURRENT_DATABASE + ".risk_reporting_arbnavimpacts where FundCode not like 'WED' AND Ticker like '" + ticker + "' LIMIT 1; "
    lastprice_df = pd.read_sql_query(query, con=con)
    lastprice = list(lastprice_df['LastPrice'])[0]
    return lastprice
    #  input line as   last_price = replace_CVR_price(con,ticker)


@shared_task
def refresh_base_case_and_outlier_downsides():
    """ Refreshes the base case and outlier downsides every 20 minutes for dynamically linked downsides """

    def adjust_for_london_stock(row):
        px_last = row['PX_LAST']
        if ' LN EQUITY' in row['Underlying'] or ' SJ EQUITY' in row['Underlying']:
            return float(px_last) * 0.01 if px_last else px_last
        return px_last

    def fill_null_prices(row):
        # Fill Null prices from Flat File Database
        if not row['PX_LAST'] or row['PX_LAST'] == 'None' or pd.isnull(row['PX_LAST']) or pd.isna(row['PX_LAST']):
            ticker = ' '.join(row['Underlying'].split(' ')[0:2])  # remove 'Equity' part for flat file matching
            # retrieve Last price from db
            query = 'SELECT Price FROM wic.daily_flat_file_db where Flat_file_as_of = (select max(flat_file_as_of) ' \
                    'from wic.daily_flat_file_db) and Ticker like "' + ticker + '" LIMIT 1;'
            rs = con.execute(query)
            last_price = 0
            for price in rs:
                last_price = price[0]
            return last_price

        return row['PX_LAST']

    def populate_last_prices(row):
        if pd.isnull(row['PX_LAST']) or not row['PX_LAST'] or row['PX_LAST'] == 'None' or pd.isna(row['PX_LAST']):
            return row['LastPrice']  # Return the Last Price fetched from the OMS

        return row['PX_LAST']

    def update_regression_downside(row) -> float:
        try:
            deal = MA_Deals.objects.get(deal_name=row['TradeGroup'])
            deal_downsides = DealDownside.objects.filter(deal=deal, content_type=ContentType.objects.get_for_model(
                LinearRegressionDownside))
            for regression_downside in deal_downsides:
                if regression_downside.content_object.is_selected:
                    return update_downside_from_model(regression_downside)
            logger.error(f'No Regression Model Selected for Deal id {deal}: ' + row['TradeGroup'])
        except Exception as e:
            logger.error(traceback.format_exc())
        return 0
    def update_base_case_reference_price(row):
        if row['BaseCaseDownsideType'] == 'Break Spread':
            return row['DealValue']  # Reference Price should be the deal value
        if row['BaseCaseDownsideType'] == 'Last Price':
            return row['LastPrice']  # Reference Price is refreshed Last price...

        if row['BaseCaseDownsideType'] == 'Premium/Discount':
            return row['LastPrice']  # Reference Price is refreshed Last price...
        if row['BaseCaseDownsideType'] == 'Peer Index':
            proxy_name = row['BaseCaseReferenceDataPoint']
            underlying = row['Underlying']
            tradegroup = row['TradeGroup']
            # find matching peer source with the proxy name
            try:
                return get_peerdownside_by_tradegroup(tradegroup, proxy_name, underlying)
            except Exception as e:
                logging.error(e)
        if row['BaseCaseDownsideType'] == 'CIX Index':
            # Get the Price from Live Price Dataframe
            peer_index = row['cix_ticker']
            price_df = live_price_df[live_price_df['Underlying'] == peer_index]
            price = np.NaN
            if not price_df.empty:
                price = price_df.iloc[0]['PX_LAST']
            return price

        if row['BaseCaseDownsideType'] == 'Reference Security':
            return float(
                live_price_df[live_price_df['Underlying'] == row['BaseCaseReferenceDataPoint']]['PX_LAST'].iloc[0])

        if row['BaseCaseDownsideType'] == 'Regression Peer':  # query for regression data
            return update_regression_downside(row)

        # Else just return the original BaseCaseReferencePrice
        return row['BaseCaseReferencePrice']

    def update_outlier_reference_price(row):
        if row['OutlierDownsideType'] == 'Break Spread':
            return row['DealValue']  # Reference Price should be the deal value
        if row['OutlierDownsideType'] == 'Last Price':
            return row['LastPrice']  # Reference Price is refreshed Last price...

        if row['OutlierDownsideType'] == 'Premium/Discount':
            return row['LastPrice']  # Reference Price is refreshed Last price...
        if row['OutlierDownsideType'] == 'Peer Index':
            proxy_name = row['OutlierReferenceDataPoint']
            underlying = row['Underlying']
            tradegroup = row['TradeGroup']
            # find matching peer source with the proxy name
            try:
                return get_peerdownside_by_tradegroup(tradegroup, proxy_name, underlying)
            except Exception as e:
                logging.error(e)
        if row['OutlierDownsideType'] == 'CIX Index':
            # Get the Price from Live Price Dataframe
            peer_index = row['cix_ticker']
            price_df = live_price_df[live_price_df['Underlying'] == peer_index]
            price = np.NaN
            if not price_df.empty:
                price = price_df.iloc[0]['PX_LAST']
            return price

        if row['OutlierDownsideType'] == 'Reference Security':
            return float(
                live_price_df[live_price_df['Underlying'] == row['OutlierReferenceDataPoint']]['PX_LAST'].iloc[0])

        if row['OutlierDownsideType'] == 'Regression Peer':  # query for regression data
            return update_regression_downside(row)
        # Else just return the original OutlierReferencePrice
        return row['OutlierReferencePrice']

    def match_base_case(row, base_cln, outlier_cln):
        # Only meant for Outlier
        if row['OutlierDownsideType'] == 'Match Base Case':
            return row[base_cln]
        else:
            return row[outlier_cln]

    def update_base_case_downsides(row):
        try:
            if row['BaseCaseDownsideType'] in ['Fundamental Valuation', 'Unaffected Downside']:
                base_case = row['base_case']
                base_case = None if base_case == 'None' else base_case
                return base_case
            if row['BaseCaseOperation'] in ['None', None, '']:
                base_case_ref_price = row['BaseCaseReferencePrice']
                base_case_ref_price = None if base_case_ref_price == 'None' else base_case_ref_price
                return base_case_ref_price
            base_case = eval(str(row['BaseCaseReferencePrice']) + str(row['BaseCaseOperation']) +
                             str(row['BaseCaseCustomInput']))
            if row['BaseCaseDownsideType'] == 'CIX Index':
                return base_case / 100
            return base_case / 100 if row['BaseCaseDownsideType'] == 'Peer Index' else base_case
        except Exception as e:
            print(e)
            base_case = row['base_case']
            base_case = None if base_case == 'None' else base_case
            return base_case

    def update_outlier_downsides(row):
        try:
            if row['OutlierDownsideType'] in ['Fundamental Valuation', 'Unaffected Downside']:
                outlier = row['outlier']
                outlier = None if outlier == 'None' else outlier
                return outlier
            if row['OutlierDownsideType'] == 'Match Base Case':
                base_case = row['base_case']
                base_case = None if base_case == 'None' else base_case
                return base_case
            if row['OutlierOperation'] in ['None', None, '']:
                outlier_ref_price = row['OutlierReferencePrice']
                outlier_ref_price = None if outlier_ref_price == 'None' else outlier_ref_price
                return outlier_ref_price

            outlier_px = eval(str(row['OutlierReferencePrice']) + str(row['OutlierOperation']) +
                              str(row['OutlierCustomInput']))

            if row['OutlierDownsideType'] == 'CIX Index':
                return outlier_px / 100

            return outlier_px / 100 if row['OutlierDownsideType'] == 'Peer Index' else outlier_px
        except Exception as e:
            print(e)
            outlier = row['outlier']
            outlier = None if outlier == 'None' else outlier
            return outlier

    def get_current_mkt_val(row):
        if row['SecType'] != 'EXCHOPT':
            return row['QTY'] * row['LastPrice']
        if row['SecType'] == 'EXCHOPT':
            # print(row['OptionLastPrice'])
            option_price = row['OptionLastPrice']
            return row['QTY'] * float(option_price) if option_price else 0

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    for name, info in django_db_connections.databases.items():  # Close the DB connections
        django_db_connection.close()
    has_errors = True
    # Create new Engine and Close Connections here
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)

    con = engine.connect()
    is_excluded = 'No'
    query = f'SELECT * FROM {settings.CURRENT_DATABASE}.risk_reporting_formulaebaseddownsides WHERE IsExcluded="{is_excluded}"'
    formulae_based_downsides = pd.read_sql_query(query, con=con)
    # remove offending tradegroups
    formulae_based_downsides = remove_risk_limit_CVR_error(formulae_based_downsides)
    old_formulae_tradegroups = formulae_based_downsides['TradeGroup'].nunique()
    # Update the Last Prices of Each Deal
    api_host = bbgclient.bbgclient.get_next_available_host()

    all_unique_tickers = list(formulae_based_downsides['Underlying'].unique())
    all_unique_base_case_reference_data_points = list(
        formulae_based_downsides[~(formulae_based_downsides['BaseCaseReferenceDataPoint'].isin(
            [np.nan, None, '', 'NONE', 'None']))]['BaseCaseReferenceDataPoint'].unique())

    all_unique_outlier_reference_data_points = list(
        formulae_based_downsides[~(formulae_based_downsides['OutlierReferenceDataPoint'].isin(
            [np.nan, None, '', 'NONE', 'None']))]['OutlierReferenceDataPoint'].unique())

    all_unique_tickers += all_unique_base_case_reference_data_points + all_unique_outlier_reference_data_points

    live_price_df = pd.DataFrame.from_dict(
        bbgclient.bbgclient.get_secid2field(all_unique_tickers, 'tickers', ['PX_LAST'], req_type='refdata',
                                            api_host=api_host), orient='index').reset_index()
    live_price_df['PX_LAST'] = live_price_df['PX_LAST'].apply(lambda x: x[0])
    live_price_df.columns = ['Underlying', 'PX_LAST']

    # Live Price DF already consists of  prices for Peer Groups.
    live_price_df['PX_LAST'] = live_price_df.apply(adjust_for_london_stock, axis=1)
    live_price_df['PX_LAST'] = live_price_df.apply(fill_null_prices, axis=1)

    # Merge Live Price Df
    formulae_based_downsides = pd.merge(formulae_based_downsides, live_price_df, how='left', on=['Underlying'])
    formulae_based_downsides['PX_LAST'] = formulae_based_downsides.apply(populate_last_prices, axis=1)

    # Delete the old LastPrice
    del formulae_based_downsides['LastPrice']
    formulae_based_downsides.rename(columns={'PX_LAST': 'LastPrice'}, inplace=True)

    # Got the Latest Last Prices now iterate and refresh the ReferenceDataPoint based on DownsideType
    formulae_based_downsides['BaseCaseReferencePrice'] = formulae_based_downsides.apply(
        update_base_case_reference_price, axis=1)

    match_base_case_rows = [('BaseCaseReferencePrice', 'OutlierReferencePrice'),
                            ('BaseCaseReferenceDataPoint', 'OutlierReferenceDataPoint'),
                            ('BaseCaseOperation', 'OutlierOperation'), ('BaseCaseCustomInput', 'OutlierCustomInput'),
                            ('base_case', 'outlier')
                            ]

    for base_column, outlier_column in match_base_case_rows:
        formulae_based_downsides[outlier_column] = formulae_based_downsides.apply(match_base_case, axis=1,
                                                                                  args=(base_column, outlier_column))

    formulae_based_downsides['OutlierReferencePrice'] = formulae_based_downsides.apply(update_outlier_reference_price,
                                                                                       axis=1)

    # Reference Prices are Refreshed now recalculate the Base case and outlier downsides
    formulae_based_downsides['base_case'] = formulae_based_downsides.apply(update_base_case_downsides, axis=1)
    formulae_based_downsides['outlier'] = formulae_based_downsides.apply(update_outlier_downsides, axis=1)

    old_formulaes = pd.read_sql_query('SELECT * FROM ' + settings.CURRENT_DATABASE +
                                      '.risk_reporting_formulaebaseddownsides WHERE IsExcluded="No"', con=con)
    # remove offending tradegroups to match formulae_based_downsides
    old_formulaes = remove_risk_limit_CVR_error(old_formulaes)
    # Base Case and Outliers are now updated! Delete the old table and insert new ones
    try:
        if old_formulaes['TradeGroup'].nunique() != formulae_based_downsides['TradeGroup'].nunique():
            raise TradeGroupMismatchException
        time.sleep(10)
        FormulaeBasedDownsides.objects.all().delete()
        formulae_based_downsides.to_sql(name='risk_reporting_formulaebaseddownsides', con=con, if_exists='append',
                                        index=False, schema=settings.CURRENT_DATABASE)
        print('Refreshed Base Case and Outlier Downsides successfully...')
        has_errors = False
    except TradeGroupMismatchException:
        slack_message('generic.slack',
                      {'message': 'ERROR in refresh_base_case_and_outlier_downsides: Restored Downside formulae ' +
                                  '...TradeGroup mismatch detected...'},
                      channel=get_channel_name('portal-task-errors'),
                      token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')

    except Exception as e:
        current_time = datetime.datetime.now().strftime('%Y%m%d.%H%M%S')
        old_formulaes.to_csv('old_formulaes.' + current_time + '.csv')
        formulae_based_downsides.to_csv('new_formulaes.' + current_time + '.csv')
        logger.error(traceback.format_exc(),
                     extra={'old_formulaes': old_formulaes, 'new_formulaes': formulae_based_downsides})
        slack_message('generic.slack',
                      {'message': 'ERROR in refresh_base_case_and_outlier_downsides: Restored Downside formulae ' +
                                  'state to previous...!' + str(e)[1:len(str(e)) if len(str(e)) < 1000 else 1000]},
                      channel=get_channel_name('portal-task-errors'),
                      token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')
        FormulaeBasedDownsides.objects.all().delete()
        old_formulaes.to_sql(name='risk_reporting_formulaebaseddownsides', con=con, if_exists='append', index=False,
                             schema=settings.CURRENT_DATABASE)
    """
    Update the target_downside and acquirer_upside for MA Deals.
    If TargetAcquirer = 'Target' -> ma_deals.target_downside = formulae_downside.outlier
    If TargetAcquirer = 'Acquirer' -> ma_deals.acquirer_upside = formulae_downside.outlier
    """
    try:
        if not has_errors:
            ma_deals_target_query = ''
            target_deals = set()
            target_underlying = set()
            acq_deals = set()
            ma_deals_acq_query = ''
            for index, row in formulae_based_downsides.iterrows():
                target_acquirer = row['TargetAcquirer']
                if not target_acquirer:
                    target_acquirer = ""
                outlier = row['outlier']
                tradegroup = row['TradeGroup']
                try:
                    outlier = float(outlier) if outlier else 0.0
                except ValueError:
                    outlier = 0
                target_acquirer = target_acquirer.lower()
                if target_acquirer and target_acquirer == 'target':
                    underlying = row['Underlying']
                    underlying = underlying.lower().replace("equity", "").strip() if underlying else underlying
                    temp_query = 'when deal_name="' + tradegroup + '" and LOWER(target_ticker) LIKE "' + underlying + \
                                 '%%" then ' + str(outlier) + ' '
                    ma_deals_target_query += temp_query
                    target_deals.add(tradegroup)
                    target_underlying.add(row['Underlying'])
                elif target_acquirer and target_acquirer == 'acquirer':
                    temp_query = 'when deal_name="' + tradegroup + '" then ' + str(outlier) + ' '
                    ma_deals_acq_query += temp_query
                    acq_deals.add(tradegroup)
            ma_deals_target_query = 'UPDATE ' + settings.CURRENT_DATABASE + '.risk_ma_deals SET target_downside = ( case ' + ma_deals_target_query + \
                                    ' end) where deal_name in ' + str(tuple(target_deals)) + \
                                    ' and '
            target_deal_query = ''
            for x in target_underlying:
                target_deal_query += 'LOWER(target_ticker) like "' + \
                                     (x.lower().replace("equity", "").strip() if x else x) + '%%" or '
            ma_deals_target_query += target_deal_query[:-4] + ';'
            ma_deals_acq_query = 'UPDATE ' + settings.CURRENT_DATABASE + '.risk_ma_deals SET acquirer_upside = ( case ' + ma_deals_acq_query + \
                                 ' end) where deal_name in ' + str(tuple(acq_deals)) + ';'
            con.execute(ma_deals_target_query)
            con.execute(ma_deals_acq_query)
    except Exception as e:
        print('Concise Error: ' + str(e)[1:len(str(e)) if len(str(e)) < 1000 else 1000])
        slack_message('generic.slack',
                      {'message': 'ERROR in refresh_base_case_and_outlier_downsides while syncing target_downside / ' +
                                  'acquirer_upside for MA Deals. ' + str(e)[
                                                                     1:len(str(e)) if len(str(e)) < 1000 else 1000]},
                      channel=get_channel_name('portal-task-errors'),
                      token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')

    try:
        api_host = bbgclient.bbgclient.get_next_available_host()
        # Populate all the deals
        query = "SELECT * FROM " + settings.CURRENT_DATABASE + ".risk_reporting_arbnavimpacts where FundCode not like 'WED'"
        nav_impacts_positions_df = pd.read_sql_query(query, con=con)
        # remove offending tradegroups
        nav_impacts_positions_df = remove_risk_limit_CVR_error(nav_impacts_positions_df)

        # Todo Improve this
        nav_impacts_positions_df = nav_impacts_positions_df[~(nav_impacts_positions_df['FundCode'].isin(
            ['INDEX1', 'INDEX2', 'ETF1', 'ETF2', 'INDEX3', 'ETF3', 'WIC', 'INDEX2018', 'ETF4', 'ETF2018', 'RAGE',
             'ARBETF', 'TACO']))]

        nav_impacts_positions_df.drop(columns=['LastPrice', 'RiskLimit'], inplace=True)

        ytd_performances = utils.get_arbitrage_performance(prefix='ytd')
        ytd_performances = ytd_performances[['tradegroup', 'fund', 'pnl_bps']].copy()
        ytd_performances = ytd_performances.groupby(['tradegroup', 'fund']).sum().reset_index()
        time.sleep(1)
        ytd_performances.rename(columns={'tradegroup': 'TradeGroup', 'fund': 'FundCode', 'pnl_bps': 'PnL_BPS'},
                                inplace=True)
        # Convert Underlying Ticker to format Ticker Equity
        nav_impacts_positions_df['Underlying'] = nav_impacts_positions_df['Underlying'].apply(
            lambda x: x + " EQUITY" if "EQUITY" not in x else x)
        forumale_linked_downsides = pd.read_sql_query('SELECT * FROM ' + settings.CURRENT_DATABASE +
                                                      '.risk_reporting_formulaebaseddownsides',
                                                      con=con)
        time.sleep(2)
        # Filter IsExcluded ones
        forumale_linked_downsides = forumale_linked_downsides[forumale_linked_downsides['IsExcluded'] == 'No']
        forumale_linked_downsides = forumale_linked_downsides[['TradeGroup', 'Underlying', 'RiskLimit', 'base_case',
                                                               'outlier', 'LastUpdate', 'LastPrice']]

        # Query Options Last Prices
        options_df = nav_impacts_positions_df[nav_impacts_positions_df['SecType'] == 'EXCHOPT']
        all_unique_tickers = options_df['Ticker'].apply(lambda x: x + " EQUITY").unique()

        options_live_price_df = pd.DataFrame.from_dict(
            bbgclient.bbgclient.get_secid2field(all_unique_tickers, 'tickers', ['PX_LAST'], req_type='refdata',
                                                api_host=api_host), orient='index').reset_index()
        options_live_price_df['PX_LAST'] = options_live_price_df['PX_LAST'].apply(lambda x: x[0])
        options_live_price_df.columns = ['Ticker', 'OptionLastPrice']

        merged_df = pd.merge(nav_impacts_positions_df, forumale_linked_downsides, how='inner',
                             on=['TradeGroup', 'Underlying'])

        # Now merge with Options live Price Dataframe on Ticker
        merged_df['Ticker'] = merged_df['Ticker'].apply(lambda x: x + ' EQUITY')
        merged_df = pd.merge(merged_df, options_live_price_df, how='left', on='Ticker')

        # merged_df = pd.merge(merged_df, ytd_performances, on=['TradeGroup', 'FundCode'], how='left')

        merged_df.drop(columns=['PM_BASE_CASE', 'Outlier'], inplace=True)
        merged_df.rename(columns={'base_case': 'PM_BASE_CASE', 'outlier': 'Outlier'}, inplace=True)
        nav_impacts_positions_df = merged_df.copy()
        nav_impacts_positions_df = nav_impacts_positions_df[
            (nav_impacts_positions_df['PM_BASE_CASE'] != 'None') &
            (nav_impacts_positions_df['PM_BASE_CASE'] != '') &
            (~pd.isna(nav_impacts_positions_df['PM_BASE_CASE'])) &
            (nav_impacts_positions_df['Outlier'] != 'None') &
            (nav_impacts_positions_df['Outlier'] != '') &
            (~pd.isna(nav_impacts_positions_df['Outlier']))]

        float_cols = ['DealTermsCash', 'DealTermsStock', 'DealValue', 'NetMktVal', 'FxFactor', 'Capital',
                      'BaseCaseNavImpact', 'RiskLimit',
                      'OutlierNavImpact', 'QTY', 'NAV', 'PM_BASE_CASE', 'Outlier', 'StrikePrice', 'LastPrice']

        nav_impacts_positions_df[float_cols] = nav_impacts_positions_df[float_cols].replace('', np.nan).fillna(
            0).astype(float)

        nav_impacts_positions_df['CurrMktVal'] = nav_impacts_positions_df.apply(get_current_mkt_val, axis=1)
        # Calculate the Impacts
        nav_impacts_positions_df['PL_BASE_CASE'] = nav_impacts_positions_df.apply(calculate_pl_base_case, axis=1)
        nav_impacts_positions_df['BASE_CASE_NAV_IMPACT'] = nav_impacts_positions_df.apply(
            calculate_base_case_nav_impact,
            axis=1)
        # Calculate Outlier Impacts
        nav_impacts_positions_df['OUTLIER_PL'] = nav_impacts_positions_df.apply(calculate_outlier_pl, axis=1)
        nav_impacts_positions_df['OUTLIER_NAV_IMPACT'] = nav_impacts_positions_df.apply(calculate_outlier_nav_impact,
                                                                                        axis=1)

        def adjust_with_ytd_performance(row, compare_to):
            if row['PnL_BPS'] < 0:
                return row[compare_to] + row['PnL_BPS']
            return row[compare_to]

        nav_impacts_positions_df = nav_impacts_positions_df.round({'BASE_CASE_NAV_IMPACT': 2, 'OUTLIER_NAV_IMPACT': 2})
        nav_impacts_sum_df = nav_impacts_positions_df.groupby(['TradeGroup', 'FundCode', 'RiskLimit']).agg(
            {'BASE_CASE_NAV_IMPACT': 'sum', 'OUTLIER_NAV_IMPACT': 'sum'}).reset_index()

        nav_impacts_sum_df = pd.merge(nav_impacts_sum_df, ytd_performances, how='left', on=['TradeGroup', 'FundCode'])
        nav_impacts_sum_df['BASE_CASE_NAV_IMPACT'] = nav_impacts_sum_df.apply(lambda x:
                                                                              adjust_with_ytd_performance
                                                                              (x, compare_to=
                                                                              'BASE_CASE_NAV_IMPACT'), axis=1)
        nav_impacts_sum_df['OUTLIER_NAV_IMPACT'] = nav_impacts_sum_df.apply(lambda x:
                                                                            adjust_with_ytd_performance
                                                                            (x, compare_to=
                                                                            'OUTLIER_NAV_IMPACT'), axis=1)
        nav_impacts_sum_df.drop(columns='PnL_BPS', inplace=True)

        nav_impacts_sum_df = pd.pivot_table(nav_impacts_sum_df, index=['TradeGroup', 'RiskLimit'], columns='FundCode',
                                            aggfunc=np.sum,
                                            fill_value='')

        nav_impacts_sum_df.columns = ["_".join((i, j)) for i, j in nav_impacts_sum_df.columns]
        nav_impacts_sum_df.reset_index(inplace=True)

        DailyNAVImpacts.objects.all().delete()
        nav_impacts_sum_df.to_sql(con=con, if_exists='append', index=False, name='risk_reporting_dailynavimpacts',
                                  schema=settings.CURRENT_DATABASE)

        impacts = DailyNAVImpacts.objects.all()
        impacts_df = pd.DataFrame.from_records(impacts.values())

        def get_last_update_downside(row):
            try:
                last_update = forumale_linked_downsides[forumale_linked_downsides['TradeGroup'] ==
                                                        row['TradeGroup']]['LastUpdate'].max()
            except Exception:
                last_update = None
            return last_update

        impacts_df['LastUpdate'] = impacts_df.apply(get_last_update_downside, axis=1)

        # NAV Impacts @ Position Level

        nav_impacts_positions_df = nav_impacts_positions_df.groupby(
            ['FundCode', 'TradeGroup', 'Ticker', 'PM_BASE_CASE', 'Outlier', 'LastPrice']).agg(
            {'BASE_CASE_NAV_IMPACT': 'sum', 'OUTLIER_NAV_IMPACT': 'sum'})

        nav_impacts_positions_df = pd.pivot_table(nav_impacts_positions_df,
                                                  index=['TradeGroup', 'Ticker', 'PM_BASE_CASE',
                                                         'Outlier', 'LastPrice'], columns=['FundCode'],
                                                  aggfunc=np.sum,
                                                  fill_value='')

        nav_impacts_positions_df.columns = ["_".join((i, j)) for i, j in nav_impacts_positions_df.columns]
        nav_impacts_positions_df.reset_index(inplace=True)
        nav_impacts_positions_df['CALCULATED_ON'] = datetime.datetime.now()
        PositionLevelNAVImpacts.objects.all().delete()
        nav_impacts_positions_df.to_sql(name='risk_reporting_positionlevelnavimpacts', con=con,
                                        if_exists='append', index=False, schema=settings.CURRENT_DATABASE)

    except Exception as e:
        print('Concise Error: ' + str(e)[1:len(str(e)) if len(str(e)) < 1000 else 1000])
        exc_type, exc_obj, exc_tb = sys.exc_info()
        slack_message('generic.slack',
                      {'message': 'ERROR: ' + str(e) + ' : ' + str(exc_type) + ' : ' + str(exc_tb.tb_lineno)},
                      channel=get_channel_name('portal-task-errors'),
                      token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')

    con.close()


# Following NAV Impacts Utilities
def calculate_pl_base_case(row):
    if row['SecType'] != 'EXCHOPT':
        return (row['PM_BASE_CASE'] * row['FxFactor'] * row['QTY']) - (row['CurrMktVal'] * row['FxFactor'])

    if row['PutCall'] == 'CALL':
        if row['StrikePrice'] <= row['PM_BASE_CASE']:
            x = (row['PM_BASE_CASE'] - row['StrikePrice']) * (row['QTY']) * row['FxFactor']
        else:
            x = 0
    elif row['PutCall'] == 'PUT':
        if row['StrikePrice'] >= row['PM_BASE_CASE']:
            x = (row['StrikePrice'] - row['PM_BASE_CASE']) * (row['QTY']) * row['FxFactor']
        else:
            x = 0
    return -row['CurrMktVal'] + x


def calculate_base_case_nav_impact(row):
    return (row['PL_BASE_CASE'] / row['NAV']) * 100


def calculate_outlier_pl(row):
    if row['SecType'] != 'EXCHOPT':
        return (row['Outlier'] * row['FxFactor'] * row['QTY']) - (row['CurrMktVal'] * row['FxFactor'])

    if row['PutCall'] == 'CALL':
        if row['StrikePrice'] <= row['Outlier']:
            x = (row['Outlier'] - row['StrikePrice']) * (row['QTY']) * row['FxFactor']
        else:
            x = 0
    elif row['PutCall'] == 'PUT':
        if row['StrikePrice'] >= row['Outlier']:
            x = (row['StrikePrice'] - row['Outlier']) * row['QTY'] * row['FxFactor']
        else:
            x = 0

    return -row['CurrMktVal'] + x


def calculate_outlier_nav_impact(row):
    return (row['OUTLIER_PL'] / row['NAV']) * 100


@shared_task
def email_nav_impacts_report():
    """ Daily NAV Impacts Report run at 6.45am """

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    try:
        engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                               + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
        con = engine.connect()
        df = pd.read_sql_query('SELECT * FROM ' + settings.CURRENT_DATABASE + '.risk_reporting_dailynavimpacts',
                               con=con)
        time.sleep(3)
        downsides_df = pd.read_sql_query(
            'SELECT TradeGroup, base_case,outlier, max(LastUpdate) as LastUpdate FROM '
            + settings.CURRENT_DATABASE + '.risk_reporting_formulaebaseddownsides WHERE IsExcluded = \'No\''
                                          ' GROUP BY TradeGroup', con=con)
        time.sleep(3)
        con.close()
        arb_ytd_pnl = utils.get_arbitrage_performance(prefix='ytd')
        arb_ytd_pnl = arb_ytd_pnl[arb_ytd_pnl['fund'] == 'ARB'][['tradegroup', 'ytd_dollar']].copy()
        arb_ytd_pnl = arb_ytd_pnl.groupby(['tradegroup']).sum().reset_index()

        time.sleep(3)
        arb_ytd_pnl.rename(columns={'tradegroup': 'TradeGroup', 'ytd_dollar': '(ARB) YTD $ P&L'}, inplace=True)
        daily_nav_impacts = df.copy()
        daily_nav_impacts.drop(['BASE_CASE_NAV_IMPACT_TAQ', 'OUTLIER_NAV_IMPACT_TAQ'], axis=1, errors='ignore',
                               inplace=True)

        daily_nav_impacts.columns = ['id', 'TradeGroup', 'RiskLimit', 'Base Case (AED)', 'Base Case (ARB)',
                                     'Base Case (CAM)', 'Base Case (LEV)', 'Base Case (LG)', 'Base Case (MACO)',
                                     'Outlier (AED)', 'Outlier (ARB)', 'Outlier (CAM)',
                                     'Outlier (LEV)', 'Outlier (LG)', 'Outlier (MACO)',
                                     'Base Case (MALT)', 'Outlier (MALT)', 'Last Update', 'Base Case (WED)',
                                     'Base Case (WIC)', 'Outlier (WED)', 'Outlier (WIC)', 'Base Case (PRELUDE)',
                                     'Outlier (PRELUDE)', 'Base Case (EVNT)', 'Outlier (EVNT)']

        daily_nav_impacts = daily_nav_impacts[
            ['id', 'TradeGroup', 'RiskLimit', 'Base Case (AED)', 'Base Case (ARB)', 'Base Case (CAM)',
             'Base Case (LEV)', 'Base Case (LG)', 'Base Case (MACO)', 'Outlier (AED)',
             'Outlier (ARB)', 'Outlier (CAM)', 'Outlier (LEV)', 'Outlier (LG)', 'Outlier (MACO)',
             'Base Case (MALT)', 'Outlier (MALT)', 'Base Case (PRELUDE)', 'Outlier (PRELUDE)',
             'Base Case (EVNT)', 'Outlier (EVNT)'
             ]]

        precision_cols = ['Base Case (AED)', 'Base Case (ARB)', 'Base Case (CAM)', 'Base Case (LEV)',
                          'Base Case (LG)', 'Base Case (MACO)', 'Outlier (AED)', 'Outlier (ARB)',
                          'Outlier (CAM)', 'Outlier (LEV)', 'Outlier (LG)', 'Outlier (MACO)',
                          'Base Case (MALT)', 'Outlier (MALT)', 'Base Case (PRELUDE)', 'Outlier (PRELUDE)',
                          'Base Case (EVNT)', 'Outlier (EVNT)']

        def round_df(val):
            try:
                return np.round(float(val), decimals=2) if val is not None else val
            except ValueError:
                return val

        for col in precision_cols:
            daily_nav_impacts[col] = daily_nav_impacts[col].apply(round_df)
            daily_nav_impacts[col] = daily_nav_impacts[col].apply(pd.to_numeric)

        daily_nav_impacts = daily_nav_impacts[['TradeGroup', 'RiskLimit', 'Base Case (ARB)', 'Base Case (MACO)',
                                               'Base Case (MALT)', 'Base Case (AED)', 'Base Case (CAM)',
                                               'Base Case (LG)', 'Base Case (LEV)', 'Base Case (PRELUDE)',
                                               'Base Case (EVNT)',
                                               'Outlier (ARB)', 'Outlier (MACO)', 'Outlier (MALT)', 'Outlier (AED)',
                                               'Outlier (CAM)', 'Outlier (LG)', 'Outlier (LEV)', 'Outlier (PRELUDE)',
                                               'Outlier (EVNT)']]

        def excel_formatting(row):
            ret = ["color:green" for _ in row.index]
            # Color Risk Limit and TradeGroup
            ret[row.index.get_loc("RiskLimit")] = "color:red"
            ret[row.index.get_loc("TradeGroup")] = "color:black"

            if abs(row['RiskLimit']) <= abs(row['Base Case (ARB)']):
                ret[row.index.get_loc("Base Case (ARB)")] = "color:red"

            if abs(row['RiskLimit']) <= abs(row['Base Case (MACO)']):
                ret[row.index.get_loc("Base Case (MACO)")] = "color:red"

            if abs(row['RiskLimit']) <= abs(row['Base Case (MALT)']):
                ret[row.index.get_loc("Base Case (MALT)")] = "color:red"

            if abs(row['RiskLimit']) <= abs(row['Base Case (PRELUDE)']):
                ret[row.index.get_loc("Base Case (PRELUDE)")] = "color:red"

            if abs(row['RiskLimit']) <= abs(row['Base Case (EVNT)']):
                ret[row.index.get_loc("Base Case (EVNT)")] = "color:red"

            if abs(row['RiskLimit']) <= abs(row['Outlier (ARB)']):
                ret[row.index.get_loc("Outlier (ARB)")] = "color:red"

            if abs(row['RiskLimit']) <= abs(row['Outlier (MACO)']):
                ret[row.index.get_loc("Outlier (MACO)")] = "color:red"

            if abs(row['RiskLimit']) <= abs(row['Outlier (MALT)']):
                ret[row.index.get_loc("Outlier (MALT)")] = "color:red"

            if abs(row['RiskLimit']) <= abs(row['Outlier (PRELUDE)']):
                ret[row.index.get_loc("Outlier (PRELUDE)")] = "color:red"

            if abs(row['RiskLimit']) <= abs(row['Outlier (EVNT)']):
                ret[row.index.get_loc("Outlier (EVNT)")] = "color:red"

            # Multi Strat is 2x Risk
            if abs(row['RiskLimit']) <= 2 * abs(row['Base Case (AED)']):
                ret[row.index.get_loc("Base Case (AED)")] = "color:red"

            if abs(row['RiskLimit']) <= 2 * abs(row['Base Case (CAM)']):
                ret[row.index.get_loc("Base Case (CAM)")] = "color:red"

            if abs(row['RiskLimit']) <= 2 * abs(row['Base Case (LG)']):
                ret[row.index.get_loc("Base Case (LG)")] = "color:red"

            if abs(row['RiskLimit']) <= 2 * abs(row['Outlier (AED)']):
                ret[row.index.get_loc("Outlier (AED)")] = "color:red"

            if abs(row['RiskLimit']) <= 2 * abs(row['Outlier (CAM)']):
                ret[row.index.get_loc("Outlier (CAM)")] = "color:red"

            if abs(row['RiskLimit']) <= 2 * abs(row['Outlier (LG)']):
                ret[row.index.get_loc("Outlier (LG)")] = "color:red"

            # Leveraged is 3x Risk
            if abs(row['RiskLimit']) <= 3 * abs(row['Base Case (LEV)']):
                ret[row.index.get_loc("Base Case (LEV)")] = "color:red"

            if abs(row['RiskLimit']) <= 3 * abs(row['Outlier (LEV)']):
                ret[row.index.get_loc("Outlier (LEV)")] = "color:red"

            return ret

        daily_nav_impacts = daily_nav_impacts.style.apply(excel_formatting, axis=1)

        downsides_df.columns = ['TradeGroup', 'Downside Base Case', 'Outlier', 'Last Downside Revision']

        df = df[['TradeGroup', 'RiskLimit', 'BASE_CASE_NAV_IMPACT_ARB', 'OUTLIER_NAV_IMPACT_ARB']]

        df = pd.merge(df, downsides_df, on='TradeGroup')

        df.columns = ['TradeGroup', 'RiskLimit', '(Base Case) NAV Impact', '(Outlier) NAV Impact',
                      'Downside Base Case', 'Outlier', 'Last Update']

        df = df[df['(Base Case) NAV Impact'] != '']
        df['(Base Case) NAV Impact'] = df['(Base Case) NAV Impact'].apply(lambda x: np.round(float(x), decimals=2))

        df = df[df['(Outlier) NAV Impact'] != '']
        df['(Outlier) NAV Impact'] = df['(Outlier) NAV Impact'].apply(lambda x: np.round(float(x), decimals=2))

        downsides_not_updated = df[pd.isna(df['Downside Base Case'])]['TradeGroup'].tolist()
        extra_message = '' if len(downsides_not_updated) == 0 else \
            '<br><br> Please update downsides for these Tradegroups: ' + ', '.join(downsides_not_updated)
        df = df[~(pd.isna(df['Downside Base Case']))]
        df = df[~(pd.isna(df['Outlier']))]
        df['Downside Base Case'] = df['Downside Base Case'].apply(lambda x: np.round(float(x), decimals=2))
        df['Outlier'] = df['Outlier'].apply(lambda x: np.round(float(x), decimals=2))

        def get_base_case_impact_over_limit(row):
            if abs(row['RiskLimit']) <= abs(row['(Base Case) NAV Impact']):
                return np.round((row['RiskLimit'] - row['(Base Case) NAV Impact']), decimals=2)

            return np.round((row['RiskLimit'] - row['(Base Case) NAV Impact']), decimals=2)

        def get_outlier_impact_over_limit(row):
            if abs(row['RiskLimit']) <= abs(row['(Outlier) NAV Impact']):
                return np.round((row['RiskLimit'] - row['(Outlier) NAV Impact']), decimals=2)

            return np.round((row['RiskLimit'] - row['(Outlier) NAV Impact']), decimals=2)

        df['(BaseCase)Impact Over Limit'] = df.apply(get_base_case_impact_over_limit, axis=1)
        df['(Outlier)Impact Over Limit'] = df.apply(get_outlier_impact_over_limit, axis=1)

        df1 = df.sort_values(by=['(BaseCase)Impact Over Limit'], ascending=False)
        df_over_limit = df1[df1['(BaseCase)Impact Over Limit'] >= 0]
        df_under_limit = df1[df1['(BaseCase)Impact Over Limit'] < 0]
        df_under_limit = df_under_limit.sort_values(by=['(BaseCase)Impact Over Limit'], ascending=False)
        df = pd.concat([df_over_limit, df_under_limit])
        df['(Outlier)Impact Over Limit'] = df['(Outlier)Impact Over Limit'].apply(lambda x: str(x) + '%')
        df['(BaseCase)Impact Over Limit'] = df['(BaseCase)Impact Over Limit'].apply(lambda x: str(x) + '%')
        df = pd.merge(df, arb_ytd_pnl, how='left', on='TradeGroup')

        df['(ARB) YTD $ P&L'] = format_with_commas(df, '(ARB) YTD $ P&L')

        # Get last Synced time
        last_calculated_on = PositionLevelNAVImpacts.objects.latest('CALCULATED_ON').CALCULATED_ON

        def export_excel(export_df):
            with io.BytesIO() as buffer:
                writer = pd.ExcelWriter(buffer)
                export_df.to_excel(writer)
                writer.save()
                return buffer.getvalue()

        def color_negative_red(val):
            if val == '':
                return ''
            value = float(val.split('%')[0])
            if value >= 0:
                color = 'red'
            else:
                color = 'black'

            return 'color: %s' % color

        del df['Downside Base Case']
        del df['Outlier']
        df = df[['TradeGroup', 'RiskLimit', '(Base Case) NAV Impact', '(BaseCase)Impact Over Limit',
                 '(Outlier) NAV Impact', '(Outlier)Impact Over Limit', 'Last Update', '(ARB) YTD $ P&L']]

        df = df.style.applymap(color_negative_red,
                               subset=['(Outlier)Impact Over Limit', '(BaseCase)Impact Over Limit']).set_table_styles([
            {'selector': 'tr:hover td', 'props': [('background-color', 'yellow')]},
            {'selector': 'th, td', 'props': [('border', '1px solid black'),
                                             ('padding', '4px'),
                                             ('text-align', 'center')]},
            {'selector': 'th', 'props': [('font-weight', 'bold')]},
            {'selector': '', 'props': [('border-collapse', 'collapse'),
                                       ('border', '1px solid black')]}
        ])
        sum_df = pd.DataFrame()
        try:
            sum_df, details_df = get_ess_nav_impacts()
        except Exception as e:
            print(e)

        if not sum_df.empty:
            # Take a subset of columns
            main_cols = ['tradegroup', 'fundamental_downside', 'cix_implied_downside', 'nav_risk_AED',
                         'cix_nav_risk_AED']
            numeric_cols = ['cix_implied_downside', 'nav_risk_AED', 'cix_nav_risk_AED']
            impact_cols = ['nav_risk_AED', 'cix_nav_risk_AED']

            def try_conversion(value):
                ret = np.NaN
                try:
                    ret = float(value)
                except ValueError as ve:
                    pass
                return np.round(ret, decimals=2)

            for eachCol in numeric_cols:
                sum_df[eachCol] = sum_df[eachCol].apply(lambda x: try_conversion(x))

            sum_df = sum_df.sort_values(by='nav_risk_AED', ascending=False)

            for eachCol in impact_cols:
                sum_df[eachCol] = sum_df[eachCol].apply(lambda x: str(x) + '%' if not pd.isna(x) else x)

            sum_df = sum_df[main_cols].fillna('')
            renamed_cols = ['TradeGroup', 'Fundamental Downside', 'CIX Implied Downside', 'Base Case NAV Impact AED',
                            'CIX NAV Impact AED']
            sum_df.columns = renamed_cols

            sum_df = sum_df.style.applymap(color_negative_red,
                                           subset=['Base Case NAV Impact AED', 'CIX NAV Impact AED']).set_table_styles([

                {'selector': 'tr:hover td', 'props': [('background-color', 'yellow')]},
                {'selector': 'th, td', 'props': [('border', '1px solid black'),
                                                 ('padding', '4px'),
                                                 ('text-align', 'center')]},
                {'selector': 'th', 'props': [('font-weight', 'bold')]},
                {'selector': '', 'props': [('border-collapse', 'collapse'),
                                           ('border', '1px solid black')]}
            ])

        html = """ \
                <html>
                  <head>
                  </head>
                  <body>
                    <p>Synchronization Timestamp: {0}</p>
                    <a href="http://192.168.0.16:8000/risk_reporting/merger_arb_risk_attributes">
                    Click to visit Realtime NAV Impacts Page</a>{1}<br><br>
                    {2}
                    <br>
                    <h3>NAV Impacts</h3>
                    {3}
                  </body>
                </html>
        """.format(last_calculated_on, extra_message, df.hide_index().render(index=False),
                   sum_df.hide_index().render(index=False))

        exporters = {'Merger Arb NAV Impacts (' + get_todays_date_yyyy_mm_dd() + ').xlsx': export_excel,
                     'NAV Impacts (' + get_todays_date_yyyy_mm_dd() + ').xlsx': export_excel}
        subject = '(Risk Automation) Merger Arb & NAV Impacts - ' + get_todays_date_yyyy_mm_dd()
        send_email(from_addr=settings.EMAIL_HOST_USER, pswd=settings.EMAIL_HOST_PASSWORD,
                   recipients=['iteam@wicfunds.com'],
                   subject=subject, from_email='dispatch@wicfunds.com', html=html,
                   EXPORTERS=exporters, dataframe=[daily_nav_impacts, sum_df], multiple=True
                   )
        dbutils.add_task_record()

    except Exception as e:
        import traceback
        print('Error Occured....')
        traceback.print_exc()

        dbutils.add_task_record(status=e)


@shared_task
def email_daily_formulae_linked_downsides():
    """ Daily Formulae Reports run at 7pm """

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    try:
        engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                               + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
        con = engine.connect()
        downsides_df = pd.read_sql_query(
            'SELECT * FROM ' + settings.CURRENT_DATABASE + '.risk_reporting_formulaebaseddownsides ', con=con)
        credit_deals_up_down_df = pd.DataFrame.from_records(CreditDealsUpsideDownside.objects.all().values())
        time.sleep(3)

        alert_message = ''
        downsides_df = downsides_df[downsides_df['IsExcluded'] == 'No']
        null_risk_limits = downsides_df[(downsides_df['RiskLimit'] == 0) | (pd.isna(downsides_df['RiskLimit'])) |
                                        (downsides_df['RiskLimit'].astype(str) == '') |
                                        (downsides_df['RiskLimit'].astype(str) == 'None')]['TradeGroup'].unique()

        null_base_case_downsides = \
            downsides_df[(downsides_df['base_case'] == 0) | (pd.isna(downsides_df['base_case'])) |
                         (downsides_df['base_case'] == '') |
                         (downsides_df['base_case'] == 'None')]['TradeGroup'].unique()
        null_outlier_downsides = downsides_df[(downsides_df['outlier'] == 0) | (pd.isna(downsides_df['outlier'])) |
                                              (downsides_df['outlier'] == '') |
                                              (downsides_df['outlier'] == 'None')]['TradeGroup'].unique()
        credit_deals_null_up_downside = credit_deals_up_down_df[((credit_deals_up_down_df['downside'] == '') |
                                                                 (credit_deals_up_down_df['upside'] == ''))]
        credit_deals_null_downside_tradegroups = credit_deals_null_up_downside['tradegroup'].unique()

        if len(null_risk_limits) > 0:
            alert_message += '<strong>Following have Undefined or Zero Risk Limits</strong>: <div class="bg-warning">' + \
                             ' , '.join(null_risk_limits) + "</div>"

        if len(null_base_case_downsides) > 0:
            alert_message += '<strong><br><br> Following have Undefined or Zero Base case</strong>: ' \
                             '<div class="bg-warning">' + ' , '.join(null_base_case_downsides) + "</div>"

        if len(null_outlier_downsides) > 0:
            alert_message += '<strong><br><br> Following have Undefined or Zero Outlier</strong>: ' \
                             '<div class="bg-warning">' + ' , '.join(null_outlier_downsides) + "</div>"

        if len(credit_deals_null_downside_tradegroups) > 0:
            alert_message += '<strong><br><br>CREDIT DEALS</strong><br>' \
                             '<strong>Following Credit Deals TradeGroups have NULL upside/downside</strong>:' \
                             '<div class="bg-warning">' + ' , '.join(credit_deals_null_downside_tradegroups) + "</div>"

        def export_excel(df):
            with io.BytesIO() as buffer:
                writer = pd.ExcelWriter(buffer)
                df.to_excel(writer)
                writer.save()
                return buffer.getvalue()

        html = """ \
                <html>
                  <head>
                  </head>
                  <body>
                    PFA Daily Backup for Formulae Linked Downsides and Credit Deals<br><br>
                    {0}
                  </body>
                </html>
        """.format(alert_message)

        exporters = {'FormulaeLinkedDownsides (' + get_todays_date_yyyy_mm_dd() + ').xlsx': export_excel,
                     'CreditDealsUpsideDownsides (' + get_todays_date_yyyy_mm_dd() + ').xlsx': export_excel}

        subject = '(Risk Automation) FormulaeLinkedDownsides & Credit Deals - ' + get_todays_date_yyyy_mm_dd()
        send_email(from_addr=settings.EMAIL_HOST_USER, pswd=settings.EMAIL_HOST_PASSWORD,
                   recipients=['risk@wicfunds.com'], subject=subject,
                   from_email='dispatch@wicfunds.com', html=html, EXPORTERS=exporters,
                   dataframe=[downsides_df, credit_deals_up_down_df], multiple=True)
        dbutils.add_task_record()
    except Exception as e:
        print('Error Occured....')
        print(e)
        dbutils.add_task_record(status=e)


def get_ytd_key(period_dict):
    if period_dict:
        for key in period_dict.keys():
            if period_dict.get(key) == 'YTD':
                return key
    return None


def round_bps(value):
    try:
        return float(value * 0.01)
    except ValueError:
        return value


def get_bps_value(row):
    return row['P&L(bps)'].get(get_ytd_key(row['Period']))


def hover(hover_color="#ffff99"):
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])


def style_funds(x):
    return ['font-size: 125%; font-weight: bold; border: 1px solid black' if v == 'Loss Budgets' else '' for v in
            x.index]


def calculate_status(row):
    if row['max_amount'] == 0 and row['min_amount'] == 0:
        row['status'] = 'CLOSED'
    else:
        row['status'] = 'ACTIVE'
    return row


@shared_task
def email_pl_target_loss_budgets():
    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    # check if previous daily tasks are done
    # if not clear_to_run(['funds_snapshot.tasks.cache_sleeve_attribution']):
    #     slack_message('generic.slack',
    #                   {'message': 'email_pl_target_loss_budgets skipped due to previous tasks missing/failing'},
    #                   channel=get_channel_name('portal-task-errors'),
    #                   token=settings.SLACK_TOKEN,
    #                   name='ESS_IDEA_DB_ERROR_INSPECTOR')
    #     return

    loss_budgets, ytd_return_sleeve_df, ytd_dollar_sleeve_df = calculate_pnl_budgets(send_email=True)
    loss_budgets = loss_budgets.drop(columns=['Last Updated'])
    ytd_dollar_sleeve_df['Sleeve'] = ytd_dollar_sleeve_df['Sleeve'].fillna('UNLISTED')
    pivoted = pd.pivot_table(loss_budgets, columns=['Fund'], aggfunc=lambda x: x, fill_value='')
    pivoted = pivoted[PNL_FUND_LIST]
    pivoted = pivoted.reindex(['AUM',
                               'Ann Gross P&L Target %',
                               'Gross YTD Return',
                               'YTD P&L % of Target',
                               'Time Passed',
                               'Ann Gross P&L Target $',
                               'Gross YTD P&L',
                               '',
                               'Loss Budget',
                               'YTD Total Loss % of Budget',
                               'Time Passed',
                               'Ann Loss Budget $',
                               'YTD Closed Deal Losses',
                               'YTD Active Deal Losses', ])
    df1 = pivoted.iloc[:8].copy()
    df2 = pivoted.iloc[8:].copy()
    df3 = pd.DataFrame([list(pivoted.columns.values)], columns=list(pivoted.columns.values))
    df1 = df1.append(df3)
    df1.index.values[5] = '* Ann Gross P&L Target $'
    df1.index.values[8] = 'Loss Budgets'
    df1 = df1.append(df2)
    df1.index.values[9] = 'Ann Loss Budget %'
    df1.index.values[0] = 'Investable Assets'
    df1.index.values[4] = 'Time Passed%'
    df1.index.values[11] = 'Time Passed %'
    df1.index.values[12] = '* Ann Loss Budget $'
    df1 = df1.replace(np.nan, '', regex=True)

    def excel_formatting(row):
        bold = False
        if row.index.str.contains('Sleeve_').any():
            if row['Sleeve_'] == 'Total' or row['Sleeve_'] == 'Sleeve_':
                bold = True
        if bold:
            ret = ["color:green; font-weight:bold" for _ in row.index]
        else:
            ret = ["color:green" for _ in row.index]
        if row.index.str.contains('Sleeve_').any():
            if bold:
                ret[row.index.get_loc("Sleeve_")] = "color:black; font-weight:bold"
            else:
                ret[row.index.get_loc("Sleeve_")] = "color:black"
        if row.index.str.contains('TradeGroup_').any():
            ret[row.index.get_loc("TradeGroup_")] = "color:black"
        if row.index.str.contains('Catalyst_').any():
            ret[row.index.get_loc("Catalyst_")] = "color:black"

        columns = row[row.index.str.startswith('Total YTD')].index.to_list()

        for column in columns:
            if isinstance(row[column], (int, float)) and row[column] < 0:
                if bold:
                    ret[row.index.get_loc(column)] = "color:red; font-weight:bold"
                else:
                    ret[row.index.get_loc(column)] = "color:red"
            elif isinstance(row[column], (str)) and row[column] == column:
                ret[row.index.get_loc(column)] = "color:black; font-weight:bold"

        return ret

    def sleeve_excel_formatting(row):
        bold = False
        if row.index.str.contains('Sleeve').any():
            if row['Sleeve'] == 'Total':
                bold = True
        if bold:
            ret = ["color:green; font-weight:bold" for _ in row.index]
        else:
            ret = ["color:green" for _ in row.index]
        columns = row.index.tolist()
        for column in columns:
            if row.index.str.contains(column).any():
                if column == 'Sleeve':
                    ret[row.index.get_loc("Sleeve")] = "color:black; font-weight:bold"
                elif isinstance(row[column], (int, float)) and row[column] < 0:
                    if bold:
                        ret[row.index.get_loc(column)] = "color:red; font-weight:bold"
                    else:
                        ret[row.index.get_loc(column)] = "color:red"
                elif isinstance(row[column], (str)) and row[column] == column:
                    ret[row.index.get_loc(column)] = "color:black; font-weight:bold"
                elif isinstance(row[column], (str)) and row[column].find("%") > -1:
                    value = row[column]
                    value = float(value.replace("%", "").replace(",", ""))
                    if value >= 0:
                        if bold:
                            ret[row.index.get_loc(column)] = "color:green; font-weight:bold"
                        else:
                            ret[row.index.get_loc(column)] = "color:green"
                    else:
                        if bold:
                            ret[row.index.get_loc(column)] = "color:red; font-weight:bold"
                        else:
                            ret[row.index.get_loc(column)] = "color:red"

        return ret

    styles = [
        hover(),
        dict(selector="th", props=[("font-size", "125%"), ("text-align", "center")]),
        dict(selector="tr", props=[("text-align", "center")]),
        dict(selector="caption", props=[("caption-side", "bottom")]),
        {'selector': 'tr:hover td', 'props': [('background-color', 'green')]},
        {'selector': 'th, td', 'props': [('border', '1px solid black'), ('padding', '4px'), ('text-align', 'center')]},
        {'selector': 'th', 'props': [('font-weight', 'bold')]},
        {'selector': '', 'props': [('border-collapse', 'collapse'), ('border', '1px solid black'), ('text-align',
                                                                                                    'center')]}
    ]
    # df1.drop(columns=['TAQ'], inplace=True)
    styled_html = (df1.style.apply(style_funds).set_table_styles(styles).set_caption(
        "PL Targets & Loss Budgets (" + get_todays_date_yyyy_mm_dd() + ")"))

    def export_excel(df_list):
        with io.BytesIO() as buffer:
            writer = pd.ExcelWriter(buffer)
            workbook = writer.book
            sheet_names = ['TradeGroup P&L', 'Sleeve P&L %', 'Sleeve P&L $', 'Fund P&L Monitor']
            for i, df in enumerate(df_list):
                sheet_name = sheet_names[i]
                worksheet = workbook.add_worksheet(sheet_name)
                writer.sheets[sheet_name] = worksheet
                if sheet_name == 'Fund P&L Monitor':
                    df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, index=True)
                elif sheet_name == 'Sleeve P&L %':
                    worksheet.write(0, 0, 'Profit PL % Breakdown by Sleeves')
                    df[0].to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=0, index=False)
                    worksheet.write(10, 0, 'Loss PL % Breakdown by Sleeves')
                    df[1].to_excel(writer, sheet_name=sheet_name, startrow=11, startcol=0, index=False)
                elif sheet_name == 'Sleeve P&L $':
                    worksheet.write(0, 0, 'Profit PL $ Breakdown by Sleeves')
                    df[0].to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=0, index=False)
                    worksheet.write(12, 0, 'Loss PL $ Breakdown by Sleeves')
                    df[1].to_excel(writer, sheet_name=sheet_name, startrow=13, startcol=0, index=False)
                else:
                    df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, index=False)
                worksheet.set_column('A:L', 20)
            format1 = workbook.add_format({'num_format': '#,###'})
            worksheet = writer.sheets['TradeGroup P&L']
            worksheet.set_column('A:C', 20)
            worksheet.set_column('D:M', 20, format1)
            writer.save()
            return buffer.getvalue()

    final_live_df, final_daily_pnl, position_level_pnl, last_updated, fund_level_live, final_position_level_ytd_pnl, \
    fund_drilldown_details, final_position_level_itd_pnl, final_live_itd_df = views.get_data()

    report_columns = final_live_df[['TradeGroup_', 'Sleeve_', 'Catalyst_']]
    fund_columns = final_live_df.filter(like='Total YTD PnL')  # generic way to fetch all fund columns
    final_live_df = pd.concat([report_columns, fund_columns], axis=1)

    final_live_df_columns = list(final_live_df.columns.values)
    final_live_df_columns.remove('TradeGroup_')
    final_live_df_columns.remove('Sleeve_')
    final_live_df_columns.remove('Catalyst_')
    fund_list = PNL_FUND_LIST
    final_live_df.reset_index(inplace=True)
    cash_index = final_live_df[final_live_df['TradeGroup_'] == 'CASH'].index
    if not cash_index.empty:
        # Change Sleeve for CASH TradeGroup to CASH
        final_live_df.at[cash_index, 'Sleeve_'] = 'CASH'
    unique_sleeves = final_live_df['Sleeve_'].unique()
    ytd_return_unique_funds = ytd_return_sleeve_df.Fund.unique().tolist()
    ytd_return_unique_sleeves = ytd_return_sleeve_df.Sleeve.unique().tolist()
    ytd_dollar_unique_sleeves = ytd_dollar_sleeve_df.Sleeve.unique().tolist()
    if 'Risk' in ytd_return_unique_sleeves:
        ytd_return_unique_sleeves.remove('Risk')
    if 'Forwards' in ytd_return_unique_sleeves:
        ytd_return_unique_sleeves.remove('Forwards')

    profit_sleeve_ytd = pd.DataFrame(columns=['Sleeve'] + fund_list)
    loss_sleeve_ytd = pd.DataFrame(columns=['Sleeve'] + fund_list)
    profit_sleeve_ytd_perc = pd.DataFrame(columns=['Sleeve'] + ytd_return_unique_funds)
    loss_sleeve_ytd_perc = pd.DataFrame(columns=['Sleeve'] + ytd_return_unique_funds)
    profit_dollar_sleeve_df = ytd_dollar_sleeve_df[ytd_dollar_sleeve_df['Gross YTD Dollar'] > 0]. \
        groupby(['Fund', 'Sleeve']).agg('sum').reset_index()
    loss_dollar_sleeve_df = ytd_dollar_sleeve_df[ytd_dollar_sleeve_df['Gross YTD Dollar'] < 0]. \
        groupby(['Fund', 'Sleeve']).agg('sum').reset_index()

    for sleeve in ytd_return_unique_sleeves:
        new_sleeve = SLEEVE_DICT.get(sleeve, sleeve)
        profit_row_perc_dict = {'Sleeve': new_sleeve}
        for fund in ytd_return_unique_funds:
            gross_ytd_return_index = ytd_return_sleeve_df[
                (ytd_return_sleeve_df['Fund'] == fund) & (ytd_return_sleeve_df['Sleeve'] == sleeve)].index
            if not gross_ytd_return_index.empty:
                gross_ytd_return_index = gross_ytd_return_index[0]
                gross_ytd_return = ytd_return_sleeve_df.at[gross_ytd_return_index, 'Gross YTD Return']
                if not np.isnan(gross_ytd_return):
                    loss_budget_row_index = loss_budgets[loss_budgets['Fund'] == fund].index
                    if not loss_budget_row_index.empty:
                        loss_budget_row_index = loss_budget_row_index[0]
                        ann_gross_pl_target_perc = float(float(
                            loss_budgets.at[loss_budget_row_index, 'Ann Gross P&L Target %'].replace("%", "")) * 0.01)
                        ytd_sleeve_perc_target = float(gross_ytd_return / ann_gross_pl_target_perc)
                else:
                    ytd_sleeve_perc_target = 0.00
            else:
                ytd_sleeve_perc_target = 0.00

            profit_row_perc_dict[fund] = float(ytd_sleeve_perc_target * 0.01)
        profit_sleeve_ytd_perc = profit_sleeve_ytd_perc.append(profit_row_perc_dict, ignore_index=True)

    for sleeve in ytd_dollar_unique_sleeves:
        new_sleeve = SLEEVE_DICT.get(sleeve, sleeve)
        profit_row_dollar_dict = {'Sleeve': new_sleeve}
        loss_row_perc_dict = {'Sleeve': new_sleeve}
        loss_row_dollar_dict = {'Sleeve': new_sleeve}
        for fund in ytd_return_unique_funds:
            profit_dollar_sleeve_index = profit_dollar_sleeve_df[
                (profit_dollar_sleeve_df['Fund'] == fund) & (profit_dollar_sleeve_df['Sleeve'] == new_sleeve)].index
            if not profit_dollar_sleeve_index.empty:
                profit_dollar_sleeve_index = profit_dollar_sleeve_index[0]
                profit_dollar = profit_dollar_sleeve_df.at[profit_dollar_sleeve_index, 'Gross YTD Dollar']
                if np.isnan(profit_dollar):
                    profit_dollar = 0.00
            else:
                profit_dollar = 0.00

            loss_dollar_sleeve_index = loss_dollar_sleeve_df[
                (loss_dollar_sleeve_df['Fund'] == fund) & (loss_dollar_sleeve_df['Sleeve'] == new_sleeve)].index
            if not loss_dollar_sleeve_index.empty:
                loss_dollar_sleeve_index = loss_dollar_sleeve_index[0]
                loss_dollar = loss_dollar_sleeve_df.at[loss_dollar_sleeve_index, 'Gross YTD Dollar']
                if not np.isnan(loss_dollar):
                    loss_budget_row_index = loss_budgets[loss_budgets['Fund'] == fund].index
                    if not loss_budget_row_index.empty:
                        loss_budget_row_index = loss_budget_row_index[0]
                        ann_loss_budget_dollar = int(
                            float(loss_budgets.at[loss_budget_row_index, 'Ann Loss Budget $'].replace(",", "")))
                        loss_row_perc_dict[fund] = float((loss_dollar / ann_loss_budget_dollar) * 100)
                else:
                    loss_dollar = 0.00
                    loss_row_perc_dict[fund] = 0.00
            else:
                loss_dollar = 0.00
                loss_row_perc_dict[fund] = 0.00

            profit_row_dollar_dict[fund] = int(profit_dollar)
            loss_row_dollar_dict[fund] = int(loss_dollar)
        profit_sleeve_ytd = profit_sleeve_ytd.append(profit_row_dollar_dict, ignore_index=True)
        loss_sleeve_ytd = loss_sleeve_ytd.append(loss_row_dollar_dict, ignore_index=True)
        loss_sleeve_ytd_perc = loss_sleeve_ytd_perc.append(loss_row_perc_dict, ignore_index=True)

    # Calculate total of the columns in all dataframes
    total_profit_dict = {'Sleeve': 'Total'}
    total_loss_dict = {'Sleeve': 'Total'}
    total_loss_perc_dict = {'Sleeve': 'Total'}

    for fund in fund_list:
        total_profit_dict[fund] = profit_sleeve_ytd[fund].sum()
        total_loss_dict[fund] = loss_sleeve_ytd[fund].sum()
        total_loss_perc_dict[fund] = str(round(loss_sleeve_ytd_perc[fund].sum(), 2)) + "%"
        loss_sleeve_ytd_perc[fund] = format_with_percentage_decimal(loss_sleeve_ytd_perc, fund)
    profit_sleeve_ytd = profit_sleeve_ytd.append(total_profit_dict, ignore_index=True)
    loss_sleeve_ytd = loss_sleeve_ytd.append(total_loss_dict, ignore_index=True)
    loss_sleeve_ytd_perc = loss_sleeve_ytd_perc.append(total_loss_perc_dict, ignore_index=True)

    total_profit_perc_dict = {'Sleeve': 'Total'}
    for fund in ytd_return_unique_funds:
        total_profit_perc_dict[fund] = str(round(profit_sleeve_ytd_perc[fund].sum(), 2)) + "%"
        profit_sleeve_ytd_perc[fund] = format_with_percentage_decimal(profit_sleeve_ytd_perc, fund)
    profit_sleeve_ytd_perc = profit_sleeve_ytd_perc.append(total_profit_perc_dict, ignore_index=True)

    # Display in the following Fund order
    column_order = ['Sleeve', 'ARB', 'MALT', 'AED', 'LG', 'TACO', 'EVNT']
    loss_sleeve_ytd_perc = loss_sleeve_ytd_perc[column_order]
    profit_sleeve_ytd_perc = profit_sleeve_ytd_perc[column_order]
    loss_sleeve_ytd = loss_sleeve_ytd[column_order]
    profit_sleeve_ytd = profit_sleeve_ytd[column_order]

    # Sort according to the ['M&A', 'CREDIT', 'ESS', 'OPP'] sleeve order
    profit_sleeve_ytd = sort_by_sleeve(profit_sleeve_ytd, 'Sleeve')
    loss_sleeve_ytd = sort_by_sleeve(loss_sleeve_ytd, 'Sleeve')
    profit_sleeve_ytd_perc = sort_by_sleeve(profit_sleeve_ytd_perc, 'Sleeve')
    loss_sleeve_ytd_perc = sort_by_sleeve(loss_sleeve_ytd_perc, 'Sleeve')

    # Revert the changing of Sleeve name for CASH TradeGroup
    final_live_df.at[cash_index, 'Sleeve_'] = 0
    final_live_df.drop(columns=['index'], inplace=True)

    # Sort the dataframe according to Sleeve followed by TradeGroup in alphabetical order
    sleeve_sorting = ['M&A', 'CREDIT', 'ESS', 'OPP']
    for sleeve in final_live_df.Sleeve_.unique():
        if isinstance(sleeve, str):
            sleeve = sleeve.strip()
        if sleeve not in sleeve_sorting:
            sleeve_sorting.append(sleeve)
    final_live_df['Sleeve_'] = pd.Categorical(final_live_df['Sleeve_'], sleeve_sorting)
    final_live_df = final_live_df.sort_values(by=['Sleeve_', 'TradeGroup_'], ascending=[True, True])

    # Replace '0' with '-' and convert all numbers to int
    for column in final_live_df_columns:
        final_live_df[column] = final_live_df[column].astype(int)
        final_live_df = final_live_df.replace({column: 0}, '-')

    final_live_df = final_live_df.style.apply(excel_formatting, axis=1)
    profit_sleeve_ytd = profit_sleeve_ytd.style.apply(sleeve_excel_formatting, axis=1)
    loss_sleeve_ytd = loss_sleeve_ytd.style.apply(sleeve_excel_formatting, axis=1)
    profit_sleeve_ytd_perc = profit_sleeve_ytd_perc.style.apply(sleeve_excel_formatting, axis=1)
    loss_sleeve_ytd_perc = loss_sleeve_ytd_perc.style.apply(sleeve_excel_formatting, axis=1)

    exporters = {'PL Targets & Loss Budgets (' + get_todays_date_yyyy_mm_dd() + ').xlsx': export_excel}
    subject = 'PL Targets & Loss Budgets - ' + get_todays_date_yyyy_mm_dd()

    # Send MStrat Drawdowns
    styled_ess_mstrat_df, original_ess_mstrat_df = ess_multistrat_drawdown_monitor()
    html = """ \
                <html>
                  <head>
                  </head>
                  <body>
                    <p>PL Targets & Loss Budgets ({date})</p>
                    <a href="http://192.168.0.16:8000">Click to visit Realtime PL Targets & Loss Budgets Page</a>
                    <br>
                    <a href="http://192.168.0.16:8000/realtime_pnl_impacts/live_tradegroup_pnl">
                        Click to visit Realtime TradeGroup PL Page
                    </a>
                    <br><br>
                    {table}
                    <br>
                    <p>* Above data has been calculated using Average YTD Investable Assets</p>
                    <br><br>
                    
                  </body>
                </html>
        """.format(table=styled_html.render(), date=get_todays_date_yyyy_mm_dd(),
                   essmstratdrawdown=styled_ess_mstrat_df.hide_index().render(index=False))

    send_email(from_addr=settings.EMAIL_HOST_USER, pswd=settings.EMAIL_HOST_PASSWORD,
               recipients=['iteam@wicfunds.com'],
               subject=subject, from_email='dispatch@wicfunds.com', html=html,
               EXPORTERS=exporters,
               dataframe=[final_live_df, [profit_sleeve_ytd_perc, loss_sleeve_ytd_perc],
                          [profit_sleeve_ytd, loss_sleeve_ytd], df1])

    dbutils.add_task_record()


def push_data_to_table(df, send_email=False):
    df = df.rename(columns={'Fund': 'fund', 'YTD Active Deal Losses': 'ytd_active_deal_losses',
                            'YTD Closed Deal Losses': 'ytd_closed_deal_losses', 'Loss Budget': 'ann_loss_budget_perc',
                            'AUM': 'investable_assets', 'Gross YTD P&L': 'gross_ytd_pnl', 'Time Passed': 'time_passed',
                            'Ann Gross P&L Target %': 'ann_gross_pnl_target_perc', 'Last Updated': 'last_updated',
                            'Ann Gross P&L Target $': 'ann_gross_pnl_target_dollar',
                            'YTD P&L % of Target': 'ytd_pnl_perc_target',
                            'Ann Loss Budget $': 'ann_loss_budget_dollar', 'Gross YTD Return': 'gross_ytd_return',
                            'YTD Total Loss % of Budget': 'ytd_total_loss_perc_budget'})
    df = df.applymap(str)
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD +
                           "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()
    if send_email:
        PnlMonitors.objects.filter(last_updated__date=datetime.date.today()).delete()
    df.to_sql(con=con, name='realtime_pnl_impacts_pnlmonitors', schema=settings.CURRENT_DATABASE,
              if_exists='append', chunksize=10000, index=False)
    con.close()


def calculate_pnl_budgets(send_email=False):
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD +
                           "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()

    raw_df = utils.get_arbitrage_performance(prefix='ytd')
    flat_file_df = pd.read_sql('select TradeGroup as tradegroup, Fund as fund, LongShort as long_short, max(amount) '
                               'as max_amount, min(amount) as min_amount from wic.daily_flat_file_db where '
                               'Flat_file_as_of = (select max(Flat_file_as_of) from wic.daily_flat_file_db) and Fund is not null '
                               'group by TradeGroup, Fund, LongShort;', con=con)

    latest_date = FundSleeveAttribution.objects.latest('date').date
    sleeve_df = pd.DataFrame.from_records(FundSleeveAttribution.objects.filter(
        date=latest_date, period='YTD').values('fund', 'pnl_bps', 'sleeve'))
    sleeve_df.rename(columns={'fund': 'Fund', 'sleeve': 'Sleeve', 'pnl_bps': 'Gross YTD Return'}, inplace=True)
    return_targets_df = pd.read_sql(
        'SELECT DISTINCT t.fund as Fund, t.profit_target FROM ' + settings.CURRENT_DATABASE +
        '.realtime_pnl_impacts_pnlprofittarget t INNER JOIN (SELECT fund, MAX(last_updated)'
        ' AS Max_last_updated FROM ' + settings.CURRENT_DATABASE +
        '.realtime_pnl_impacts_pnlprofittarget GROUP BY fund) groupedt ON t.fund = '
        'groupedt.fund AND t.last_updated = groupedt.Max_last_updated;', con=con)

    loss_budget_df = pd.read_sql('SELECT DISTINCT t.fund as Fund, t.loss_budget as `Loss Budget` FROM ' +
                                 settings.CURRENT_DATABASE + '.realtime_pnl_impacts_pnllossbudget t INNER JOIN (SELECT '
                                                             'fund, MAX(last_updated) AS Max_last_updated FROM ' + settings.CURRENT_DATABASE +
                                 '.realtime_pnl_impacts_pnllossbudget GROUP BY fund) groupedt ON t.fund = groupedt.fund'
                                 ' AND t.last_updated = groupedt.Max_last_updated;', con=con)
    con.close()
    return_targets_df.rename(columns={'profit_target': 'Ann Gross P&L Target %'}, inplace=True)

    if 'id' in raw_df.columns.values:
        raw_df.drop(columns=['id'], inplace=True)

    flat_file_df = flat_file_df.apply(calculate_status, axis=1)
    df_active_null = raw_df[(raw_df['status'] == 'ACTIVE') | (raw_df['status'].isnull())].copy()
    df_closed = raw_df[raw_df['status'] == 'CLOSED'].copy()
    df_active_null = df_active_null.drop(columns=['status'])

    flat_file_df['long_short'] = flat_file_df['long_short'].str.upper()
    flat_file_df['tradegroup'] = flat_file_df['tradegroup'].str.upper()
    new_status_df = pd.merge(df_active_null, flat_file_df[['tradegroup', 'fund', 'long_short', 'status']],
                             on=['tradegroup', 'fund', 'long_short'], how='left')
    df = df_closed.append(new_status_df, ignore_index=True)
    df["status"] = df["status"].fillna('CLOSED')

    gross_ytd_return_sleeve_df = sleeve_df[['Fund', 'Sleeve', 'Gross YTD Return']].copy()
    gross_ytd_dollar_sleeve_df = df[['fund', 'sleeve', 'tradegroup', 'ytd_dollar']].copy()
    cash_index = gross_ytd_dollar_sleeve_df[gross_ytd_dollar_sleeve_df['tradegroup'] == 'CASH'].index
    if not cash_index.empty:
        for index in cash_index:
            gross_ytd_dollar_sleeve_df.at[index, 'sleeve'] = 'CASH'
    gross_ytd_dollar_sleeve_df.drop(columns=['tradegroup'], inplace=True)
    gross_ytd_dollar_sleeve_df.rename(columns={'fund': 'Fund', 'sleeve': 'Sleeve', 'ytd_dollar': 'Gross YTD Dollar'},
                                      inplace=True)
    new_sleeve_df = sleeve_df[['Fund', 'Gross YTD Return']].copy()

    new_sleeve_df = new_sleeve_df.groupby(['Fund']).sum()
    new_sleeve_df['Gross YTD Return'] = new_sleeve_df['Gross YTD Return'].apply(round_bps)
    new_sleeve_df = new_sleeve_df.reset_index()

    pd.set_option('float_format', '{:2}'.format)
    active_tradegroups = df[df['status'] == 'ACTIVE'].copy()
    closed_tradegroups = df[df['status'] == 'CLOSED'].copy()
    fund_active_losers = active_tradegroups[active_tradegroups['ytd_dollar'] < 0][['fund', 'ytd_dollar']]. \
        groupby(['fund']).agg('sum').reset_index()

    for fund in PNL_FUND_LIST:
        if fund_active_losers[fund_active_losers['fund'] == fund].empty:
            fund_active_losers = fund_active_losers.append(pd.DataFrame([[fund, 0]], columns=['fund', 'ytd_dollar']))
    fund_active_losers.reset_index(drop=True, inplace=True)
    fund_active_losers.columns = ['Fund', 'YTD Active Deal Losses']
    investable_assets_df = df[['fund', 'fund_aum']].drop_duplicates()
    investable_assets_df.columns = ['Fund', 'AUM']
    fund_realized_losses = closed_tradegroups[closed_tradegroups['ytd_dollar'] < 0][['fund', 'ytd_dollar']]. \
        groupby(['fund']).agg('sum').reset_index()
    for fund in PNL_FUND_LIST:
        if fund_realized_losses[fund_realized_losses['fund'] == fund].empty:
            fund_realized_losses = fund_realized_losses.append(
                pd.DataFrame([[fund, 0]], columns=['fund', 'ytd_dollar']))
    fund_realized_losses.reset_index(drop=True, inplace=True)
    fund_realized_losses.columns = ['Fund', 'YTD Closed Deal Losses']
    fund_pnl = pd.merge(fund_active_losers, fund_realized_losses, on=['Fund'])

    merged_df = pd.merge(fund_pnl, loss_budget_df, on=['Fund'])
    merged_df = pd.merge(merged_df, return_targets_df, on=['Fund'])
    merged_df = pd.merge(merged_df, investable_assets_df, on=['Fund'])
    average_aum_df = get_average_ytd_aum()
    merged_df = pd.merge(merged_df, average_aum_df, on=['Fund'], how='left')
    float_cols = ['YTD Active Deal Losses', 'YTD Closed Deal Losses', 'Loss Budget', 'Ann Gross P&L Target %', 'AUM']
    merged_df[float_cols] = merged_df[float_cols].astype(float)
    merged_df['Ann Gross P&L Target $'] = merged_df['Ann Gross P&L Target %'] * merged_df['Average YTD AUM'] * 0.01
    gross_ytd_pnl = df[['fund', 'ytd_dollar']].groupby('fund').agg('sum').reset_index()
    gross_ytd_pnl.columns = ['Fund', 'Gross YTD P&L']
    merged_df = pd.merge(merged_df, gross_ytd_pnl, on='Fund')
    merged_df = pd.merge(merged_df, new_sleeve_df, on='Fund', how='left')
    merged_df['YTD P&L % of Target'] = (merged_df['Gross YTD Return'] / merged_df['Ann Gross P&L Target %']) * 100
    merged_df.fillna(0, inplace=True)
    loss_budgets = merged_df[['Fund', 'YTD Active Deal Losses', 'YTD Closed Deal Losses', 'Loss Budget', 'AUM',
                              'Gross YTD P&L', 'Ann Gross P&L Target %', 'Gross YTD Return',
                              'Ann Gross P&L Target $', 'YTD P&L % of Target', 'Average YTD AUM']].copy()

    loss_budgets['Ann Loss Budget $'] = loss_budgets['Loss Budget'] * loss_budgets['Average YTD AUM'] * 0.01

    current_year = datetime.date.today().year
    ytd = datetime.date(current_year, 1, 1)
    now = datetime.datetime.today().date()
    days_passed = (now - ytd).days
    time_passed_in_percentage = np.round((days_passed / 365.0), decimals=2) * 100
    loss_budgets['Time Passed'] = "{:.2f}%".format(time_passed_in_percentage)

    loss_budgets['Ann Gross P&L Target %'] = format_with_percentage_decimal(loss_budgets, 'Ann Gross P&L Target %')
    loss_budgets['Loss Budget'] = format_with_percentage_decimal(loss_budgets, 'Loss Budget')
    loss_budgets['YTD Total Loss % of Budget'] = ((loss_budgets['YTD Active Deal Losses'] +
                                                   loss_budgets['YTD Closed Deal Losses']) /
                                                  loss_budgets['Ann Loss Budget $']) * 100

    # Rounding off to 2 decimal places for % values
    loss_budgets['YTD Active Deal Losses'] = format_with_commas(loss_budgets, 'YTD Active Deal Losses')
    loss_budgets['YTD Closed Deal Losses'] = format_with_commas(loss_budgets, 'YTD Closed Deal Losses')
    loss_budgets['AUM'] = format_with_commas(loss_budgets, 'AUM')
    loss_budgets['Gross YTD Return'] = format_with_percentage_decimal(loss_budgets, 'Gross YTD Return')
    loss_budgets['YTD P&L % of Target'] = format_with_percentage_decimal(loss_budgets, 'YTD P&L % of Target')
    loss_budgets['Ann Loss Budget $'] = format_with_commas(loss_budgets, 'Ann Loss Budget $')
    loss_budgets['YTD Total Loss % of Budget'] = format_with_percentage_decimal(loss_budgets,
                                                                                'YTD Total Loss % of Budget')
    loss_budgets['Gross YTD P&L'] = format_with_commas(loss_budgets, 'Gross YTD P&L')
    loss_budgets['Ann Gross P&L Target $'] = format_with_commas(loss_budgets, 'Ann Gross P&L Target $')
    loss_budgets['Last Updated'] = datetime.datetime.now()
    loss_budgets.drop(columns=['Average YTD AUM'], inplace=True)
    push_data = loss_budgets
    push_data_to_table(push_data, send_email=send_email)
    return loss_budgets, gross_ytd_return_sleeve_df, gross_ytd_dollar_sleeve_df


@shared_task
def calculate_realtime_pnl_budgets():
    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD +
                           "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()

    pnl_budgets = pd.read_sql('select * from ' + settings.CURRENT_DATABASE + '.realtime_pnl_impacts_pnlmonitors ' + \
                              'where last_updated = (Select max(last_updated) from ' + settings.CURRENT_DATABASE + \
                              '.realtime_pnl_impacts_pnlmonitors where DATE(last_updated) = CURDATE());', con=con)
    gross_ytd_return_df = pd.read_sql('select fund, gross_ytd_return from ' + settings.CURRENT_DATABASE + \
                                      '.realtime_pnl_impacts_pnlmonitors where last_updated = (Select ' + \
                                      'min(last_updated) from ' + settings.CURRENT_DATABASE + \
                                      '.realtime_pnl_impacts_pnlmonitors where DATE(last_updated) = CURDATE());',
                                      con=con)
    con.close()
    if pnl_budgets.empty or gross_ytd_return_df.empty:
        slack_message('generic.slack',
                      {'message': 'ERROR: Realtime Loss Budgets (Dashboard) email not sent at 8:00 AM. ' + \
                                  'Missing today\'s data, hence can not refresh PnL monitors.'},
                      channel=get_channel_name('portal-task-errors'),
                      token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')

    if not gross_ytd_return_df.empty and not pnl_budgets.empty:

        if 'id' in pnl_budgets.columns.values:
            pnl_budgets.drop(columns=['id'], inplace=True)

        final_live_df, final_daily_pnl, position_level_pnl, last_updated, fund_level_live, \
        final_position_level_ytd_pnl, fund_drilldown_details, final_position_level_itd_pnl, final_live_itd_df = views.get_data()
        fund_level_df = views.calculate_roc_nav_fund_level_live(fund_level_live)
        fund_level_df = fund_level_df[['Fund', 'Contribution_to_NAV']]
        fund_level_df = fund_level_df.groupby(['Fund']).sum()
        fund_level_df['fund'] = fund_level_df.index
        fund_level_df.reset_index(drop=True, inplace=True)
        fund_level_df['Contribution_to_NAV'] = fund_level_df['Contribution_to_NAV'] * 0.01
        ytd_return_merge = pd.merge(gross_ytd_return_df, fund_level_df, on='fund', how='left')
        ytd_return_merge['gross_ytd_return'] = ytd_return_merge['gross_ytd_return'].str.replace('%', '').apply(atof)
        ytd_return_merge['gross_ytd_return'] = ytd_return_merge['gross_ytd_return'].fillna(0) + ytd_return_merge[
            'Contribution_to_NAV'].fillna(0)
        ytd_return_merge.rename(columns={'gross_ytd_return': 'new_ytd_return'}, inplace=True)
        ytd_return_merge.drop(columns=['Contribution_to_NAV'], inplace=True)

        live_ytd_pnl = pd.Series([])
        ytd_live_pnl_sum = pd.Series([])
        ytd_pnl_df = pd.DataFrame()
        columns = final_live_df.columns.values
        for i, column in enumerate(columns):
            if "Total YTD PnL_" in column:
                live_ytd_pnl[i] = column.split("_")[-1]
                ytd_live_pnl_sum[i] = final_live_df[column].sum()
        ytd_pnl_df['fund'] = live_ytd_pnl
        ytd_pnl_df['Live P&L'] = ytd_live_pnl_sum
        realtime_pl_budget_df = pd.merge(pnl_budgets, ytd_pnl_df, on=['fund'], how='left')
        realtime_pl_budget_df['gross_ytd_pnl'] = realtime_pl_budget_df['Live P&L']
        realtime_pl_budget_df['investable_assets'] = realtime_pl_budget_df['investable_assets'].str.replace(',',
                                                                                                            '').apply(
            atof)
        realtime_pl_budget_df = pd.merge(realtime_pl_budget_df, ytd_return_merge, on='fund', how='left')
        realtime_pl_budget_df.drop(columns=['gross_ytd_return'], inplace=True)
        realtime_pl_budget_df.rename(columns={'new_ytd_return': 'gross_ytd_return'}, inplace=True)
        realtime_pl_budget_df['ann_gross_pnl_target_perc'] = realtime_pl_budget_df[
            'ann_gross_pnl_target_perc'].str.replace('%', '').apply(atof)
        realtime_pl_budget_df['ytd_pnl_perc_target'] = realtime_pl_budget_df['gross_ytd_return'] / \
                                                       realtime_pl_budget_df['ann_gross_pnl_target_perc'] * 100
        realtime_pl_budget_df['gross_ytd_return'] = format_with_percentage_decimal(realtime_pl_budget_df,
                                                                                   'gross_ytd_return')
        realtime_pl_budget_df['investable_assets'] = format_with_commas(realtime_pl_budget_df, 'investable_assets')
        realtime_pl_budget_df['gross_ytd_pnl'] = format_with_commas(realtime_pl_budget_df, 'gross_ytd_pnl')
        realtime_pl_budget_df['ann_gross_pnl_target_perc'] = format_with_percentage_decimal(realtime_pl_budget_df,
                                                                                            'ann_gross_pnl_target_perc')
        realtime_pl_budget_df['ytd_pnl_perc_target'] = format_with_percentage_decimal(realtime_pl_budget_df,
                                                                                      'ytd_pnl_perc_target')
        realtime_pl_budget_df.drop(columns=['Live P&L'], inplace=True)
        realtime_pl_budget_df['last_updated'] = datetime.datetime.now()
        push_data = realtime_pl_budget_df
        push_data_to_table(push_data, send_email=False)


def format_with_commas(df, column):
    """
    Format the numeric values in a specific column of a DataFrame with thousand commas.
    Non-numeric values in the column are left unchanged.

    Parameters
    ----------
    df : pd.DataFrame Input DataFrame.
    column_name : str Column to format.

    Returns
    -------
    pd.Series Column with formatted values.
    """

    def format_value(val):
        if isinstance(val, (int, float)):
            return "{:,.0f}".format(val) if isinstance(val, int) else "{:,.2f}".format(val)
        return val

    return df[column].apply(format_value)


def format_with_percentage_decimal(df, column):
    return df.apply(lambda x: "{:,.2f}%".format(x[column]), axis=1)


def get_average_ytd_aum():
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD +
                           "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()
    current_date = datetime.date.today()
    current_year = current_date.year
    if current_date.day == 1 and current_date.month == 1:
        # There will be not data for new year on January 1st, hence take previous year data
        current_year = current_year - 1

    average_aum_df = pd.read_sql('select fund as Fund, avg(aum) as `Average YTD AUM` from wic.daily_flat_file_db '
                                 'where year(Flat_file_as_of) = "' + str(
        current_year) + '" group by fund order by fund',
                                 con=con)
    con.close()
    return average_aum_df


def sort_by_sleeve(given_df, sleeve_column):
    sleeve_sorting = ['M&A', 'CREDIT', 'ESS', 'OPP']
    for sleeve in given_df[sleeve_column].unique():
        if isinstance(sleeve, str):
            sleeve = sleeve.strip()
        if sleeve not in sleeve_sorting:
            sleeve_sorting.append(sleeve)
    given_df['Sleeve'] = pd.Categorical(given_df['Sleeve'], sleeve_sorting)
    given_df = given_df.sort_values(by=['Sleeve'])
    return given_df


# Refresh Credit Deals UP/Down (Every 29 minutes)
@shared_task
def refresh_credit_deals_upside_downside():
    return  # temporary fix for stopping credit deal running

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    try:
        do_revert = True
        db.connections.close_all()
        credit_deals_df = pd.DataFrame.from_records(CreditDealsUpsideDownside.objects.all().values())
        current_credit_deals_df = credit_deals_df.copy()
        formuale_df = pd.DataFrame.from_records(
            FormulaeBasedDownsides.objects.all().values('Underlying', 'outlier', 'DealValue'))
        spread_index_list = credit_deals_df[credit_deals_df['upside_type'] == 'Calculate from SIX'][
            'spread_index'].tolist()
        bloomberg_id_list = credit_deals_df['bloomberg_id'].tolist()
        ticker_bbg_id_list = list(spread_index_list + bloomberg_id_list)
        fields = ['PX_LAST']
        result = get_data_from_bloomberg_by_bg_id(ticker_bbg_id_list, fields)
        credit_deals_df['last_price'] = credit_deals_df['bloomberg_id'].map(result)
        credit_deals_df['last_price'] = credit_deals_df['last_price'].apply(get_px_last_value)
        credit_deals_df['last_price'] = credit_deals_df['last_price'].astype(float)
        credit_deals_df['last_price'] = credit_deals_df['last_price'].fillna(0.0)
        credit_deals_df['spread_px_last'] = credit_deals_df['spread_index'].map(result)
        credit_deals_df['spread_px_last'] = credit_deals_df['spread_px_last'].apply(get_px_last_value)
        credit_deals_df['spread_px_last'] = credit_deals_df['spread_px_last'].astype(float)
        credit_deals_df['spread_px_last'] = credit_deals_df['spread_px_last'].fillna(0.0)
        credit_deals_df['upside_type'] = credit_deals_df['upside_type'].replace(r'', np.nan)
        credit_deals_df['upside_type'] = credit_deals_df['upside_type'].fillna('Fundamental Valuation')
        credit_deals_df['equity_ticker'] = credit_deals_df.apply(lambda row: row['ticker'] + ' EQUITY' if 'equity' not \
                                                                                                          in row[
                                                                                                              'ticker'].lower() else
        row['ticker'].upper(),
                                                                 axis=1)
        credit_deals_df = pd.merge(credit_deals_df, formuale_df, left_on='equity_ticker', right_on='Underlying',
                                   how='left')

        credit_deals_df['upside'] = credit_deals_df.apply(lambda row: row['last_price'] + row['spread_px_last'] if \
            row['upside_type'] == 'Calculate from SIX' else row['upside'],
                                                          axis=1)
        credit_deals_df['upside'] = credit_deals_df.apply(lambda row: row['last_price'] if row['upside_type'] == \
                                                                                           'Last Price' else row[
            'upside'], axis=1)
        credit_deals_df['upside'] = credit_deals_df.apply(lambda row: row['DealValue'] if row['upside_type'] == \
                                                                                          'Match ARB' else row[
            'upside'], axis=1)

        credit_deals_df['downside'] = credit_deals_df.apply(lambda row: row['last_price'] if row['downside_type'] == \
                                                                                             'Last Price' else row[
            'downside'], axis=1)
        credit_deals_df['downside'] = credit_deals_df.apply(lambda row: row['outlier'] if row['downside_type'] == \
                                                                                          'Match ARB' else row[
            'downside'], axis=1)
        credit_deals_df['deal_value'] = credit_deals_df.apply(lambda row: row['DealValue'] if row['downside_type'] == \
                                                                                              'Match ARB' or row[
                                                                                                  'upside_type'] == 'Match ARB' else \
            row['deal_value'], axis=1)

        credit_deals_df['last_refreshed'] = datetime.datetime.now()
        credit_deals_df.drop(columns=['id'], inplace=True)
        credit_deals_df.reset_index(inplace=True)
        credit_deals_df.rename(columns={'index': 'id'}, inplace=True)
        credit_deals_df.drop(columns=['equity_ticker', 'spread_px_last', 'Underlying', 'outlier', 'DealValue'],
                             inplace=True)
        if 'id' in credit_deals_df.columns.values.tolist():
            del credit_deals_df['id']
        credit_deals_df = credit_deals_df.drop_duplicates()
        credit_deals_df.reset_index(inplace=True)
        credit_deals_df.rename(columns={'index': 'id'}, inplace=True)
        try:
            CreditDealsUpsideDownside.objects.all().delete()
            do_revert = True
            engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" +
                                   settings.WICFUNDS_DATABASE_PASSWORD + "@" + settings.WICFUNDS_DATABASE_HOST + "/" +
                                   settings.WICFUNDS_DATABASE_NAME)
            con = engine.connect()

            credit_deals_df.to_sql(con=con, name='risk_reporting_creditdealsupsidedownside',
                                   schema=settings.CURRENT_DATABASE, if_exists='append', chunksize=10000, index=False)
        except Exception as e:
            if do_revert and not current_credit_deals_df.empty:
                CreditDealsUpsideDownside.objects.all().delete()
                current_credit_deals_df.to_sql(con=con, name='risk_reporting_creditdealsupsidedownside',
                                               schema=settings.CURRENT_DATABASE, if_exists='append', chunksize=10000,
                                               index=False)
            slack_message('generic.slack',
                          {
                              'message': 'ERROR: Credit Deals upside/downside DB Refresh (every 29 minutes) had an error.' +
                                         'The database has been restored to the previous data. Exception: ' + str(e)},
                          channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN,
                          name='ESS_IDEA_DB_ERROR_INSPECTOR')
        finally:
            con.close()
    except Exception as e:
        print('Credit Deals Upside/Downside Update failed', e)
        slack_message('generic.slack',
                      {'message': 'ERROR: Credit Deals up/downside refresh (every 29 minutes) had an error.' + str(e)},
                      channel=get_channel_name('portal-task-errors'),
                      token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')


def get_px_last_value(value):
    try:
        if isinstance(value, (dict)):
            px_last_list = value.get('PX_LAST')
            if px_last_list and len(px_last_list) > 0:
                return px_last_list[0]
        return 0.00
    except Exception as e:
        return 0.00


# Credit Up/Down daily Update
@shared_task
def update_credit_deals_upside_downside_once_daily():
    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    update_credit_deals()


# Automated File Dropping to EZE
@shared_task
def drop_arb_downsides_to_eze():
    """ Runs at 6pm Mon-Fri """

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    try:
        path = settings.DEAL_INFO_EZE_UPLOAD_PATH
        deal_info_df = get_deal_info_dataframe()
        deal_info_df.to_csv(path, index=False)
        success = '_(Risk Automation)_ *Successfully Uploaded DealInfo.csv to Eze Uploads (Eze/Upload Files/)*'
        error = "."
        slack_channel = 'portal-task-reports'
    except Exception as e:
        error = '_(Risk Automation)_ *Error in Uploading DealInfo.csv* -> ' + str(e)
        success = 'ERROR! Please upload files manually'
        slack_channel = 'portal-task-errors'
    slack_message('eze_uploads.slack', {'success': success, 'error': error}, channel=get_channel_name(slack_channel),
                  token=settings.SLACK_TOKEN)

    # Now process for SecurityInfo.csv
    try:
        path = settings.SECURITY_INFO_EZE_UPLOAD_PATH
        security_info_df = get_security_info_dataframe()
        security_info_df.to_csv(path, index=False)
        success = '_(Risk Automation)_ *Successfully Uploaded SecurityInfo.csv to Eze Uploads (Eze/Upload Files/)*'
        error = "."
        slack_channel = 'portal-task-reports'
    except Exception as e:
        error = '_(Risk Automation)_ *Error in Uploading SecurityInfo.csv* -> ' + str(e)
        success = "ERROR! Please upload files manually"
        slack_channel = 'portal-task-errors'
    slack_message('eze_uploads.slack', {'success': success, 'error': error}, channel=get_channel_name(slack_channel),
                  token=settings.SLACK_TOKEN)
    dbutils.add_task_record()


# Eze Upload Alert (4pm)
@shared_task
def post_alert_before_eze_upload():
    """ Task should run at 4pm Mon-Fri """

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    downsides_df = pd.DataFrame.from_records(FormulaeBasedDownsides.objects.all().filter(IsExcluded__exact='No').
                                             values())

    null_risk_limits = downsides_df[(downsides_df['RiskLimit'] == 0) | (pd.isna(downsides_df['RiskLimit']) |
                                                                        (downsides_df['RiskLimit'].astype(str) == ''
                                                                         ))]['TradeGroup'].unique()

    null_base_case_downsides = downsides_df[(downsides_df['base_case'] == 0) | (pd.isna(downsides_df['base_case']))
                                            | (downsides_df['base_case'] == '')]['TradeGroup'].unique()
    null_outlier_downsides = downsides_df[(downsides_df['outlier'] == 0) | (pd.isna(downsides_df['outlier'])
                                                                            | (downsides_df['outlier'] == ''))][
        'TradeGroup'].unique()

    null_risk_limits = ', '.join(null_risk_limits)
    null_base_case_downsides = ', '.join(null_base_case_downsides)
    null_outlier_downsides = ', '.join(null_outlier_downsides)

    risk_limits_alert = '_(Risk Automation)_ Following have NULL/0 Risk Limits *' + null_risk_limits + "*" \
        if null_risk_limits else "_(Risk Automation)_ All Risk Limits ready for Eze Upload (at 6pm)"
    base_case_alert = '_(Risk Automation)_ Following have NULL/0 Base Case *' + null_base_case_downsides + "*" \
        if null_base_case_downsides else "_(Risk Automation)_ All base case downsides ready for Eze Upload (at 6pm)"
    outlier_alert = '_(Risk Automation)_ Following have NULL/0 Outlier *' + null_outlier_downsides + "*" \
        if null_outlier_downsides else "_(Risk Automation)_ All outlier downsides ready for Upload (at 6pm)"

    slack_message('eze_uploads.slack', {'null_risk_limits': str(risk_limits_alert),
                                        'null_base_case': str(base_case_alert),
                                        'null_outlier': str(outlier_alert)},
                  channel=get_channel_name('portal_downsides'),
                  token=settings.SLACK_TOKEN
                  )
    dbutils.add_task_record()


# ESS Multi-Strat Drawdown Monitor
@shared_task
def ess_multistrat_drawdown_monitor():
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD +
                           "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()
    print('Now Processing: ESS Multi-Strat Drawdown Monitor')
    df, final_df = None, None
    try:
        api_host = bbgclient.bbgclient.get_next_available_host()
        ess_drawdown_query = "SELECT TradeGroup, Ticker, AlphaHedge, RiskLimit, CurrentMktVal_Pct, DealDownside, SecType, " \
                             "LongShort, (amount*factor) as QTY, CurrentMktVal, FXCurrentLocalToBase as FxFactor, aum, " \
                             "PutCall, Strike as StrikePrice, DealUpside FROM " \
                             "wic.daily_flat_file_db WHERE Flat_file_as_of = " \
                             "(SELECT MAX(Flat_file_as_of) FROM wic.daily_flat_file_db) " \
                             "AND AlphaHedge IN ('Alpha', 'Alpha Hedge') AND amount<>0 AND Fund LIKE 'AED' " \
                             "AND Sleeve = 'Equity Special Situations'"

        today = datetime.datetime.now().date().strftime('%Y-%m-%d')
        capital_query = "SELECT tradegroup, ytd_romc_bps, ytd_dollar, days_5_bps, days_5_dollar, days_1_bps, " \
                        "days_1_dollar FROM " + settings.CURRENT_DATABASE + \
                        ".funds_snapshot_tradegroupperformanceovercapital WHERE sleeve = 'EQUITY SPECIAL SITUATIONS'" \
                        "AND fund LIKE 'AED' AND date = '" + today + "'AND status = 'ACTIVE'"
        ess_drawdown_df = pd.read_sql_query(ess_drawdown_query, con=con)
        ess_tg_perf_df = pd.read_sql_query(capital_query, con=con)
        ess_tg_perf_df.rename(columns={'tradegroup': 'TradeGroup', 'ytd_romc_bps': 'ROMC_YTD_bps',
                                       'ytd_dollar': 'YTD_Dollar', 'days_5_bps': '5D_bps', 'days_5_dollar': '5D_Dollar',
                                       'days_1_bps': '1D_bps', 'days_1_dollar': '1D_Dollar'}, inplace=True)
        # Exclude tradgroups (manually)
        ess_drawdown_df = ess_drawdown_df[~(ess_drawdown_df['TradeGroup'] == 'AVYA R/R')]
        ess_drawdown_df[
            ['RiskLimit', 'CurrentMktVal_Pct', 'DealDownside', 'QTY', 'aum', 'StrikePrice', 'DealUpside']].astype(float)
        # if Risk Limit is 0 or NULL assume 30 basis point Risk limit
        ess_drawdown_df.loc[ess_drawdown_df.RiskLimit == 0, "RiskLimit"] = 0.30
        ess_drawdown_df['RiskLimit'] = ess_drawdown_df['RiskLimit'].apply(lambda x: -x if x > 0 else x)
        ess_drawdown_df.rename(columns={'CurrentMktVal_Pct': 'aed_aum_pct'}, inplace=True)

        def calculate_break_pl(row):
            if row['AlphaHedge'] == 'Alpha':
                if row['SecType'] == 'EQ' and row['LongShort'] == 'Short':
                    return (row['DealUpside'] * row['QTY']) - (row['CurrentMktVal'] / row['FxFactor'])

                return (row['DealDownside'] * row['QTY']) - (row['CurrentMktVal'] / row['FxFactor'])
            # Todo Add logic for Options where LongShort = Short

            if row['PutCall'] == 'CALL':
                if row['StrikePrice'] <= row['DealDownside']:
                    x = (row['DealDownside'] - row['StrikePrice']) * (row['QTY'])
                else:
                    x = 0
            elif row['PutCall'] == 'PUT':
                if row['StrikePrice'] >= row['DealDownside']:
                    x = (row['StrikePrice'] - row['DealDownside']) * (row['QTY'])
                else:
                    x = 0
            return -row['CurrentMktVal'] + x

        ess_drawdown_df['Break PL'] = ess_drawdown_df.apply(calculate_break_pl, axis=1)
        ess_drawdown_df['NAV Risk'] = 1e2 * ess_drawdown_df['Break PL'] / ess_drawdown_df['aum']

        nav_risk_df = ess_drawdown_df[['TradeGroup', 'NAV Risk']].groupby(['TradeGroup']).sum().reset_index()
        ess_drawdown_df_equity = ess_drawdown_df[ess_drawdown_df['SecType'] == 'EQ']

        ess_drawdown_df_equity = ess_drawdown_df_equity[['TradeGroup', 'Ticker', 'AlphaHedge', 'RiskLimit',
                                                         'aed_aum_pct', 'DealDownside', 'SecType', 'LongShort',
                                                         'QTY', 'CurrentMktVal', 'aum']]
        ess_df_nav_risk = pd.merge(ess_drawdown_df_equity, nav_risk_df, on='TradeGroup')
        ess_df_nav_risk['pct_of_limit'] = 1e2 * ess_df_nav_risk['NAV Risk'] / ess_df_nav_risk['RiskLimit']
        ess_df_nav_risk['pct_aum_at_max_risk'] = 1e2 * ess_df_nav_risk['aed_aum_pct'] / ess_df_nav_risk['pct_of_limit']

        ess_df_nav_risk['Ticker'] = ess_df_nav_risk['Ticker'].apply(
            lambda x: x + " EQUITY" if 'equity' not in x.lower() else x)
        vol_series = ['VOLATILITY_260D', 'VOLATILITY_180D', 'VOLATILITY_90D', 'VOLATILITY_60D', 'VOLATILITY_30D',
                      'VOLATILITY_10D']
        tradegroup_volatility_dictionary = bbgclient.bbgclient.get_secid2field(list(ess_df_nav_risk['Ticker'].unique()),
                                                                               'tickers', vol_series,
                                                                               req_type='refdata', api_host=api_host)

        def volatility_cascade_logic(ticker):
            assumed_vol = 0.30  # 30% volatility (Assumed if nothing found in Bloomberg)
            for vol in vol_series:
                current_vol = tradegroup_volatility_dictionary[ticker][vol][0]
                if current_vol is not None:
                    return float(current_vol)

            return assumed_vol

        ess_df_nav_risk['Ann Vol'] = ess_df_nav_risk['Ticker'].apply(volatility_cascade_logic)
        ess_df_nav_risk['Ann Vol'] = ess_df_nav_risk['Ann Vol'].astype(float)

        ess_df_nav_risk['33% of Vol'] = ess_df_nav_risk['Ann Vol'] * 0.33
        ess_df_nav_risk['50% of Vol'] = ess_df_nav_risk['Ann Vol'] * 0.50
        final_df = pd.merge(ess_df_nav_risk, ess_tg_perf_df, how='left', on='TradeGroup')
        # Year-to-Date
        final_df['YTD ROMC'] = final_df['ROMC_YTD_bps'] / 100
        final_df['YTD NAV Cont'] = 1e2 * final_df['YTD_Dollar'] / final_df['aum']
        final_df['% of NAV Loss Limit'] = final_df.apply(
            lambda x: 0 if x['YTD NAV Cont'] > 0 else 1e2 * x['YTD NAV Cont'] / x['RiskLimit'], axis=1)

        # 5 Day
        final_df['5D ROC'] = final_df['5D_bps'] / 100
        final_df['5D NAV Cont'] = 1e2 * final_df['5D_Dollar'] / final_df['aum']

        # 1 Day
        final_df['1D ROC'] = final_df['1D_bps'] / 100
        final_df['1D NAV Cont'] = 1e2 * final_df['1D_Dollar'] / final_df['aum']

        final_df = final_df[['TradeGroup', 'Ticker', 'AlphaHedge', 'RiskLimit', 'aed_aum_pct', 'NAV Risk',
                             'pct_of_limit', 'pct_aum_at_max_risk', 'Ann Vol', '33% of Vol', '50% of Vol',
                             'YTD ROMC', 'YTD NAV Cont', '% of NAV Loss Limit', '5D ROC', '5D NAV Cont',
                             '1D ROC', '1D NAV Cont']]

        final_df.columns = ['TradeGroup', 'Alpha', 'AlphaHedge', 'RiskLimit', '% AUM (AED)', 'NAV Risk (%)',
                            '% of Limit',
                            '% AUM @ Max Risk', 'Ann Vol (%)', '33% of Vol (%)', '50% of Vol (%)', 'YTD ROMC (%)',
                            'YTD NAV Cont (%)', '% of NAV Loss Limit', '5D ROC (%)', '5D NAV Cont (%)',
                            '1D ROC (%)', '1D NAV Cont (%)']

        def define_color(row, column):
            color = 'black'
            ytd_romc = abs(row['YTD ROMC (%)'])
            risk_limit = abs(row['RiskLimit'])
            vol_tt_pct = abs(row['33% of Vol (%)'])
            vol_fifty_pct = abs(row['50% of Vol (%)'])
            ytd_nav_cont = abs(row['YTD NAV Cont (%)'])

            if column == 'ROMC':
                if (ytd_romc > vol_tt_pct) and (ytd_romc < vol_fifty_pct):
                    color = 'orange'
                if ytd_romc >= vol_fifty_pct:
                    color = 'red'
            else:
                if (ytd_nav_cont > 0.50 * risk_limit) and (ytd_nav_cont < 0.90 * risk_limit):
                    color = 'orange'
                if ytd_nav_cont >= 0.90 * risk_limit:
                    color = 'red'

            # No colors for Positive PnL names...
            if row['YTD ROMC (%)'] > 0:
                color = 'black'
            if row['YTD NAV Cont (%)'] > 0:
                color = 'black'

            return color

        final_df['YTD ROMC Color'] = final_df.apply(lambda x: define_color(x, 'ROMC'), axis=1)
        final_df['NAV Cont Color'] = final_df.apply(lambda x: define_color(x, 'NAV'), axis=1)

        colors_df = final_df[['TradeGroup', 'YTD ROMC Color', 'NAV Cont Color']].copy()
        del final_df['YTD ROMC Color']
        del final_df['NAV Cont Color']

        # Round to 2 decimals
        cols_precision = ['% AUM (AED)', 'NAV Risk (%)', '% of Limit', '% AUM @ Max Risk', 'Ann Vol (%)',
                          '33% of Vol (%)',
                          '50% of Vol (%)', 'YTD ROMC (%)', 'YTD NAV Cont (%)', '% of NAV Loss Limit', '5D ROC (%)',
                          '5D NAV Cont (%)', '1D ROC (%)', '1D NAV Cont (%)']
        final_df[cols_precision] = final_df[cols_precision].round(decimals=2)

        def highlight_breaches(row):
            tradegroup = row['TradeGroup']
            romc_color = colors_df[colors_df['TradeGroup'] == tradegroup]['YTD ROMC Color'].iloc[0]
            nav_color = colors_df[colors_df['TradeGroup'] == tradegroup]['NAV Cont Color'].iloc[0]

            ret = ["color:black" for _ in row.index]
            # Color Risk Limit and TradeGroup
            ytd_romc_color = romc_color if not romc_color == 'black' else romc_color
            nav_cont_color = nav_color if not nav_color == 'black' else nav_color

            ret[row.index.get_loc(
                "YTD ROMC (%)")] = "color:white;background-color:" + ytd_romc_color if not ytd_romc_color == 'black' else "color:black"
            ret[row.index.get_loc(
                "YTD NAV Cont (%)")] = "color:white;background-color:" + nav_cont_color if not nav_cont_color == 'black' else "color:black"
            ret[row.index.get_loc(
                "TradeGroup")] = "color:white;background-color:" + ytd_romc_color if not ytd_romc_color == 'black' else "color:black"

            return ret

        del final_df['% AUM @ Max Risk']  # Temporarily hiding this column

        df = final_df.style.apply(highlight_breaches, axis=1).set_table_styles([
            {'selector': 'tr:hover td', 'props': [('background-color', 'beige')]},
            {'selector': 'th, td', 'props': [('border', '1px solid black'),
                                             ('padding', '4px'),
                                             ('text-align', 'center')]},
            {'selector': 'th', 'props': [('font-weight', 'bold')]},
            {'selector': '', 'props': [('border-collapse', 'collapse'),
                                       ('border', '1px solid black')]}
        ])

    except Exception as e:
        print(e)
        slack_message('generic.slack',
                      {'message': 'ESS Multi-Strat DrawDown Monitor -- > ERROR: ' + str(e)},
                      channel=get_channel_name('portal-task-errors'),
                      token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')
    finally:
        print('Closing Connection to Relational Database Service....')
        con.close()

    return df, final_df


@shared_task
def update_nav_impacts_positions_table():
    """
    Populates Fresh Positions from latest Flat File. Further merged with Formulae Linked downsides for dynamic
    downsides
    """
    for name, info in django_db_connections.databases.items():  # Close the DB connections
        django_db_connection.close()
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()
    try:
        rs = con.execute("SELECT MAX(flat_file_as_of) from wic.daily_flat_file_db;")
        for results in rs:
            max_date_in_db = results[0].strftime('%Y-%m-%d')
        print('Getting Fresh Positions as of: ' + max_date_in_db)
        position_query = "call wic.GET_POSITIONS_FOR_NAV_IMPACTS('" + max_date_in_db + "','" + max_date_in_db + "')"
        fresh_positions = pd.read_sql_query(position_query, con=con)
        if fresh_positions.empty:
            slack_message('generic.slack',
                          {'message': '*Error: NAV Impacts positions table* <@ssuizhu> <@akubal> - Dataframe empty.'},
                          channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN,
                          name='ESS_IDEA_DB_ERROR_INSPECTOR')
            return 'Error: NAV Impacts positions table - Dataframe empty.'
        fresh_positions.rename(columns={'TG': 'TradeGroup'}, inplace=True)
        fresh_positions.reset_index(inplace=True)
        fresh_positions.rename(columns={'index': 'id'}, inplace=True)
        con.execute('SET FOREIGN_KEY_CHECKS=0;TRUNCATE TABLE ' + settings.CURRENT_DATABASE + \
                    '.risk_reporting_arbnavimpacts')
        fresh_positions.to_sql('risk_reporting_arbnavimpacts', con=con, schema=settings.CURRENT_DATABASE,
                               if_exists='append', index=False)
        slack_message('generic.slack',
                      {'message': '*Updated NAV Impacts positions table* - ' + \
                                  datetime.datetime.now().strftime('%B %d, %Y %H:%M')},
                      channel=get_channel_name('portal-task-reports'), token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')
    except Exception as e:
        slack_message('generic.slack',
                      {'message': '*Error: NAV Impacts positions table* <@ssuizhu> <@akubal>- ' + str(e)},
                      channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')
    finally:
        con.close()


@shared_task
def update_formulae_linked_downsides():
    """
    Function should run each morning and update the downsides.
    Adds new Positions for Formulae based Downsides.
    """
    for name, info in django_db_connections.databases.items():  # Close the DB connections
        django_db_connection.close()
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()

    api_host = bbgclient.bbgclient.get_next_available_host()
    df = pd.read_sql_query('call wic.GET_POSITIONS_FOR_DOWNSIDE_FORMULAE()', con=con)
    df.index.names = ['id']
    df.rename(columns={'TG': 'TradeGroup'}, inplace=True)
    # Remaining Fields to be Added are the following: 1. LastUpdate 2.IsExcluded 3. DownsideType 4. ReferenceDataPoint
    # 5. ReferencePrice 6.Operation 7. CustomInput 8.Outlier 9. cix_ticker

    cols2merge = ['TradeGroup', 'Underlying']

    df['TradeGroup'] = df['TradeGroup'].apply(lambda x: x.strip().upper())
    df['Underlying'] = df['Underlying'].apply(lambda x: x.strip().upper())
    df['Underlying'] = df['Underlying'].apply(lambda x: x + ' EQUITY' if 'EQUITY' not in x else x)
    all_unique_tickers = list(df['Underlying'].unique())
    live_price_df = pd.DataFrame.from_dict(bbgclient.bbgclient.get_secid2field(
        all_unique_tickers, 'tickers', ['PX_LAST'], req_type='refdata', api_host=api_host), orient='index')
    live_price_df = live_price_df.reset_index()
    live_price_df['PX_LAST'] = live_price_df['PX_LAST'].apply(lambda x: x[0])
    live_price_df.columns = ['Underlying', 'PX_LAST']

    # Merge Live Price Df
    df = pd.merge(df, live_price_df, how='left', on=['Underlying'])

    # Delete the old LastPrice
    del df['LastPrice']

    df.rename(columns={'PX_LAST': 'LastPrice'}, inplace=True)

    # Exclude the Current Positions
    current_df = pd.read_sql_query('SELECT * FROM ' + settings.CURRENT_DATABASE + \
                                   '.risk_reporting_formulaebaseddownsides', con=con)
    current_df = current_df[(current_df['Underlying'].isin(df['Underlying']))]  # Removes Stale positions from current
    current_df['Underlying'] = current_df['Underlying'].apply(lambda x: x.strip().upper())

    # Perform an Outer Merge on current and new df on Underlying and Tradegroup....
    # After that delete the previous Risklimit and DealValue

    # Upper case Tradegroup
    current_df['TradeGroup'] = current_df['TradeGroup'].apply(lambda x: x.strip().upper())

    current_df = pd.merge(current_df, df, how='outer', on=cols2merge).reset_index().drop(columns=['id']).rename(
        columns={'index': 'id'})

    # Delete all deals that are in CurrentDF but not present in the New DF (these are the closed positions)
    current_df = current_df[current_df['TradeGroup'].isin(df['TradeGroup'])]

    current_df.rename(columns={'DealValue_y': 'DealValue', 'LastPrice_y': 'LastPrice', 'RiskLimit_x': 'RiskLimit',
                               'TargetAcquirer_y': 'TargetAcquirer', 'Analyst_y': 'Analyst',
                               'OriginationDate_y': 'OriginationDate'}, inplace=True)
    # Delete the Old values...
    del current_df['DealValue_x']
    del current_df['LastPrice_x']
    del current_df['RiskLimit_y']  # Keep the old Risk Limit
    del current_df['TargetAcquirer_x']
    del current_df['Analyst_x']
    del current_df['OriginationDate_x']

    def fill_missing_risk_limits_for_existing_tradegroups(row):
        if pd.isna(row['RiskLimit']) or row['RiskLimit'] == '':
            # Check if current tradegroup already exists in current_df and if exists use the same risk limit
            value = current_df[current_df['RiskLimit'].notnull() & (current_df['TradeGroup'] == row['TradeGroup'])]
            if not value.empty:
                value = value['RiskLimit'].max()
            else:
                value = 0
            return value

        return row['RiskLimit']

    # Now Retroactively Fill Risk limit
    current_df['RiskLimit'] = current_df.apply(fill_missing_risk_limits_for_existing_tradegroups, axis=1)
    current_df.drop_duplicates(['TradeGroup', 'Underlying', 'Analyst'], inplace=True)
    current_df['IsExcluded'] = current_df['IsExcluded'].apply(lambda x: 'No' if pd.isnull(x) else x)

    # Truncate the Current downsides table. Rollback if fails
    old_formulae_linked_downsides = pd.read_sql_query('SELECT * FROM ' + settings.CURRENT_DATABASE + \
                                                      '.risk_reporting_formulaebaseddownsides', con=con)

    try:
        con.execute('SET FOREIGN_KEY_CHECKS=0;TRUNCATE TABLE ' + settings.CURRENT_DATABASE + \
                    '.risk_reporting_formulaebaseddownsides')
        current_df.to_sql(name='risk_reporting_formulaebaseddownsides', con=con, if_exists='append', index=False,
                          schema=settings.CURRENT_DATABASE)
        slack_message('generic.slack',
                      {'message': '*Formuale Linked Downside:* Updated with new positions - ' + \
                                  datetime.datetime.now().strftime('%B %d, %Y %H:%M')},
                      channel=get_channel_name('portal-task-reports'), token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')
    except Exception as e:
        # Restore the Previous state of the database...
        old_formulae_linked_downsides.to_sql(name='risk_reporting_formulaebaseddownsides', con=con, if_exists='append',
                                             index=False, schema=settings.CURRENT_DATABASE)
        slack_message('generic.slack',
                      {
                          'message': '*Error: Formuale Linked Downsides* Restored to previous state <@ssuizhu> <@akubal>' + str(
                              e)},
                      channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')
    finally:
        con.close()


@shared_task
def update_formulae_linked_downsides_historical():
    """
    Function should run post market hour and update the downsides and store all data historically.
    Adds new Positions for Formulae based Downsides.
    """

    obj_status = HistoricalFormulaeBasedDownsides.objects.filter(
        Datestamp=datetime.datetime.today().strftime('%Y-%m-%d')).values()
    if len(list(obj_status)) > 0:
        return
    for name, info in django_db_connections.databases.items():  # Close the DB connections
        django_db_connection.close()
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()

    # Read and copy table to historical downsides.
    current_df = pd.read_sql_query('SELECT * FROM ' + settings.CURRENT_DATABASE + \
                                   '.risk_reporting_formulaebaseddownsides', con=con)

    current_df['Datestamp'] = datetime.datetime.today().strftime('%Y-%m-%d')
    ##set id to none to allow db to select id
    current_df['id'] = None

    try:
        current_df.to_sql(name='risk_reporting_historicalformulaebaseddownsides', con=con, if_exists='append',
                          index=False,
                          schema=settings.CURRENT_DATABASE)
        slack_message('generic.slack',
                      {'message': '*Historical Formulae Linked Downside:* Updated with new positions - ' + \
                                  datetime.datetime.now().strftime('%B %d, %Y %H:%M')},
                      channel=get_channel_name('portal-task-reports'), token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')
        dbutils.add_task_record(task_name='risk_reporting.tasks.update_formulae_linked_downsides_historical')
    except Exception as e:
        # Restore the Previous state of the database...
        # current_df.to_sql(name='risk_reporting_historicalformulaebaseddownsides', con=con, if_exists='append',
        #                                      index=False, schema=settings.CURRENT_DATABASE)
        error_concise = str(e)[0:200]
        slack_message('generic.slack',
                      {
                          'message': '*Error: Historical Formuale Linked Downsides* not saved <@ssuizhu> <@akubal>' + error_concise},
                      channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')
    finally:
        con.close()

    # save current active deal downside to historical deal downside
    update_deal_downsides()
    deal_downsides = DealDownside.objects.filter(deal__archived=False)
    todays_date = holiday_utils.get_todays_date()
    for deal_downside in deal_downsides:
        HistoricalDealDownside.objects.create(deal=deal_downside.deal,
                                              underlying=deal_downside.underlying,
                                              date=todays_date,
                                              downside=deal_downside.downside,
                                              content_type=deal_downside.content_type,
                                              object_id=deal_downside.object_id)




@shared_task
def send_px_last_drop_alert():
    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    try:
        for name, info in django_db_connections.databases.items():  # Close the DB connections
            django_db_connection.close()
        engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                               + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
        con = engine.connect()
        query = 'Select Underlying, cix_ticker from ' + settings.CURRENT_DATABASE + \
                '.risk_reporting_formulaebaseddownsides where IsExcluded="No";'
        formulae_df = pd.read_sql_query(query, con=con)
        history_df = pd.read_sql_query(
            'SELECT * from ' + settings.CURRENT_DATABASE + '.risk_reporting_cixtickerpxlasthistory', con=con)
        con.close()
        formulae_df.rename(columns={'Underlying': 'underlying'}, inplace=True)
        empty_cix_tickers = formulae_df[pd.isna(formulae_df['cix_ticker'])]['underlying'].unique().tolist()
        formulae_df = formulae_df[pd.notna(formulae_df['cix_ticker'])].copy()
        formulae_df = formulae_df[~formulae_df['cix_ticker'].str.contains('N/A')].copy()
        cix_indexes = formulae_df['cix_ticker'].unique().tolist()
        api_host = bbgclient.bbgclient.get_next_available_host()
        live_price_df = pd.DataFrame.from_dict(
            bbgclient.bbgclient.get_secid2field(cix_indexes, 'tickers', ['PX_LAST'], req_type='refdata',
                                                api_host=api_host), orient='index').reset_index()
        # replace null values in px_last with 0, report it later in the process
        live_price_df['PX_LAST'] = live_price_df['PX_LAST'].apply(lambda x: x[0] if x and x[0] else 0)
        live_price_df.columns = ['cix_ticker', 'PX_LAST']
        live_price_df.dropna(inplace=True)
        formulae_df = formulae_df.drop_duplicates(subset=['cix_ticker'], keep='first').reset_index(drop=True).copy()
        merged_df = pd.merge(live_price_df, formulae_df, how='left', on='cix_ticker')
        now = datetime.datetime.now()
        if (history_df.empty) or (now.hour == 10 and now.minute >= 00 and now.minute <= 10):
            merged_df['curr_diff'] = 0
            merged_df.rename(columns={'PX_LAST': 'curr_price'}, inplace=True)
            merged_df['last_threshold_price'] = merged_df['curr_price']
            merged_df['alert_sent'] = False
            engine = create_engine(
                "mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
            con = engine.connect()
            con.execute(
                'SET FOREIGN_KEY_CHECKS=0; TRUNCATE TABLE ' + settings.CURRENT_DATABASE + '.risk_reporting_cixtickerpxlasthistory')
            con.close()
            time.sleep(2)
            engine = create_engine(
                "mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
            con = engine.connect()
            merged_df.to_sql(name='risk_reporting_cixtickerpxlasthistory', con=con, if_exists='append', index=False,
                             schema=settings.CURRENT_DATABASE, chunksize=100)
            con.close()
            if empty_cix_tickers:
                slack_message('generic.slack',
                              {
                                  'message': '*Formula Linked Downsides*: Empty CIX Ticker found for the following tickers\n' + \
                                             ', '.join(cix for cix in empty_cix_tickers)},
                              channel=get_channel_name('portal_downsides'), token=settings.SLACK_TOKEN,
                              name='ESS_IDEA_DB_ERROR_INSPECTOR')
            return 'send_px_last_drop_alert task successfully done.'
        merged_df = pd.merge(merged_df, history_df, on=['cix_ticker', 'underlying'], how='left')
        merged_df['curr_price'] = merged_df['PX_LAST']
        del merged_df['PX_LAST']
        merged_df['curr_diff'] = 1e2 * ((merged_df['last_threshold_price'].astype(float) -
                                         merged_df['curr_price'].astype(float)) /
                                        merged_df['last_threshold_price'].astype(float)).abs()
        merged_df['last_threshold_price'] = merged_df.apply(
            lambda row: row['curr_price'] if row['curr_diff'] >= 10 else row['last_threshold_price'], axis=1)
        above_threshold_df = merged_df[merged_df['curr_diff'] >= 10].copy()
        cix_ticker_updated_list = []
        if not above_threshold_df.empty:
            cix_ticker_list = above_threshold_df['cix_ticker'].unique().tolist()
            for cix_ticker in cix_ticker_list:
                temp_df = merged_df[merged_df['cix_ticker'] == cix_ticker].iloc[0]
                if not temp_df['alert_sent']:
                    logger.error(merged_df.loc[merged_df['cix_ticker'] == cix_ticker])
                    merged_df.at[merged_df['cix_ticker'] == cix_ticker, 'alert_sent'] = True
                    cix_ticker_updated_list.append(cix_ticker)
            if cix_ticker_updated_list:
                slack_message('generic.slack',
                              {'message': '*ALERT*: 10% move detected in the following Peer Groups\n' + \
                                          '\n'.join(ticker for ticker in cix_ticker_updated_list)},
                              channel=get_channel_name('risk-portal'), token=settings.SLACK_TOKEN,
                              name='ESS_IDEA_DB_ERROR_INSPECTOR')

        del merged_df['id']

        engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                               + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
        con = engine.connect()
        con.execute(
            'SET FOREIGN_KEY_CHECKS=0; TRUNCATE TABLE ' + settings.CURRENT_DATABASE + '.risk_reporting_cixtickerpxlasthistory')
        con.close()
        time.sleep(2)
        engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                               + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
        con = engine.connect()
        merged_df.to_sql(name='risk_reporting_cixtickerpxlasthistory', con=con, if_exists='append', index=False,
                         schema=settings.CURRENT_DATABASE, chunksize=100)
        con.close()
        empty_value_tickers = []
        wrong_value_tickers = []  # values with 1 for curr_price
        for index, row in merged_df.iterrows():
            ticker = row['cix_ticker']
            curr_price = float(row['curr_price'])
            if curr_price == 0:
                empty_value_tickers.append(ticker)
            elif curr_price == 1.0:
                wrong_value_tickers.append(ticker)

        additional_messages = ("following cix tickers have null prices: " + ', '.join(empty_value_tickers) + "\n"
                               if empty_value_tickers else "")

        # message is sent to error channel if additional_messages is not empty
        if additional_messages:
            slack_message('generic.slack', {'message': 'send_px_last_drop_alert task completed.' +
                                                       "\n However, some tickers have incorrect values:\n" + additional_messages},
                          channel=get_channel_name('portal-task-errors'),
                          token=settings.SLACK_TOKEN, name='ESS_IDEA_DB_ERROR_INSPECTOR')


    except Exception as e:
        logger.error(traceback.format_exc())
        slack_message('generic.slack',
                      {'message': '*ERROR*: Error in send_px_last_drop_alert task. ' + str(traceback.format_exc())},
                      channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN,
                      name='ESS_IDEA_DB_ERROR_INSPECTOR')


@shared_task
def export_daily_excel_files():
    """
    Task for aggregating all daily files into a specific folder
    """

    for name, info in django.db.connections.databases.items():  # Close the DB connections
        django.db.connection.close()

    if os.name == 'nt':  # change save location based on system os
        state_street_dir = 'K:/ETF_Files/ARB/StateStreet/'
        target_folder = 'K:/ETF_Files/Daily Files/'
    else:
        state_street_dir = '/mnt/shares/KDrive/ETF_Files/ARB/StateStreet/'
        target_folder = '/mnt/shares/KDrive/ETF_Files/Daily Files/'

    last_trade_date_str = datetime.datetime.now().strftime('%Y-%m-%d')

    query_ma_deal_detailed_export = "SELECT * FROM " + settings.CURRENT_DATABASE + ".risk_ma_deals as RMA " \
                                                                                   "LEFT OUTER JOIN " + settings.CURRENT_DATABASE + \
                                    ".risk_madealsactioniddetails as AID" \
                                    " ON RMA.action_id = AID.action_id LEFT OUTER JOIN  " + \
                                    settings.CURRENT_DATABASE + ".risk_ma_deals_risk_factors as RF " \
                                                                "ON RF.deal_id = RMA.id "
    risk_attr_df = pd.read_sql_query(query_ma_deal_detailed_export, con=connection)
    risk_attr_df.to_csv(target_folder + 'ARBDetailedExport.csv', index=False)

    # query_wic_position = f"SELECT * FROM wic.daily_flat_file_db where flat_file_as_of = '{last_trade_date_str}'"
    # wic_position_df = pd.read_sql_query(query_wic_position, con=connection)
    # wic_position_df.to_csv(target_folder + 'WicPositions.csv', index=False)

    get_attachment_by_subject('Closing File Water Island Merger Arbitrage Index',
                              target_folder + 'Closing_Values_WIMARB.xls')
    get_attachment_by_subject('Opening File Water Island Merger Arbitrage Index',
                              target_folder + 'Opening_Values_WIMARB.xls')
    get_attachment_by_subject('Pro-Forma Closing File Water Island Merger Arbitrage Index',
                              target_folder + 'PRO-FORMA-WIMARB-CLOSING.xls')
    get_attachment_by_subject('Pro-Forma Opening File Water Island Merger Arbitrage Index',
                              target_folder + 'PRO-FORMA-WIMARB-OPENING.xls')
    get_attachment_by_subject('WIMARBH-CLOSING-' + last_trade_date_str + '.xls', target_folder + 'WIMARBH-CLOSING.xls')
    get_attachment_by_subject('WIMARBH-OPENING-', target_folder + 'WIMARBH-OPENING.xls')

    copy_latest_file('WIC_CashComponent\.', state_street_dir, target_folder + 'WIC_CashComponent.csv')
    copy_latest_file('WIC_NAV\.', state_street_dir, target_folder + 'WIC_NAV.CSV')
    copy_latest_file('WIC_PCF\.', state_street_dir, target_folder + 'WIC_PCF.CSV')
    copy_latest_file('WIC_PCF_INAV\.', state_street_dir, target_folder + 'WIC_PCF_INAV.CSV')
    copy_latest_file('WIC_Trading_Cash\.', state_street_dir, target_folder + 'WIC_Trading_Cash.CSV')
    copy_latest_file('WIC_UCF\.', state_street_dir, target_folder + 'WIC_UCF.CSV')
    copy_latest_file('WIC_UCF_Preburst\.', state_street_dir, target_folder + 'WIC_UCF_Preburst.CSV')
    copy_latest_file('WIC_WIX5PR_PCF\.', state_street_dir, target_folder + 'WIC_WIX5PR_PCF.CSV')
    copy_latest_file('WIC_WIX7PR_PCF\.', state_street_dir, target_folder + 'WIC_WIX7PR_PCF.CSV')
    copy_latest_file('WIC_WIX5_PCF\.', state_street_dir, target_folder + 'WIC_WIX5_PCF.CSV')
    copy_latest_file('WIC_WIX7_PCF\.', state_street_dir, target_folder + 'WIC_WIX7_PCF.CSV')
    dbutils.add_task_record()


@shared_task
def export_ice_indices_report():
    """
    Task for collecting ICE Indices report into a specific folder, split from export_daily_excel_files since the ICE
    file is received after 1AM
    """
    if os.name == 'nt':  # change save location based on system os
        target_folder = 'K:/ETF_Files/Daily Files/'
    else:
        target_folder = '/mnt/shares/KDrive/ETF_Files/Daily Files/'

    get_attachment_by_subject('ICE_ETP_Water_Island', target_folder + 'ICE_ETP_Water_Island.csv',
                              'ICE_ETP_Water_Island_files')
    dbutils.add_task_record()


def upload_trendline():
    """
    Depericated method for updating trendlines
    """
    cols_to_get = ['TradeGroup', 'Target', 'Target Downside (Base)', 'Acq Downside (Base)', 'Acq Downside (Outlier)',
                   'UFP', 'Jefferies', 'WallachBeth',
                   'UPF.1', 'Jefferies.1', 'WallachBeth.1']
    path = r'K:\Connor\Downside Files'  # use your path
    final_df = pd.DataFrame()
    set_of_cols = set()
    filename = 'K:/Connor/Downside Files/Downsides 2022-03-03.xlsx'
    df = pd.read_excel(filename, header=1)
    for each in df.columns.values:
        set_of_cols.add(each)

    for each in cols_to_get:
        if each not in df.columns.values:
            df[each] = np.NaN
    df = df[cols_to_get]
    df['Date'] = '2022-03-03'
    final_df = pd.concat([final_df, df])
    final_df = final_df.reset_index()[cols_to_get + ['Date']]
    from sqlalchemy import create_engine
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()
    rename_scheme = {'Target': 'target_ticker', 'TradeGroup': 'tradegroup',
                     'Target Downside (Base)': 'target_downside_base',
                     'UFP': 'ufp_base', 'Jefferies': 'jefferies_base', 'Kepler (EU)': 'kepler_base',
                     'Olive Tree': 'olive_tree_base', 'WallachBeth': 'wallachbeth_base', 'Date': 'date',
                     'Acq Downside (Base)': 'acquirer_downside_base',
                     'Acq Downside (Outlier)': 'acquirer_downside_outlier',
                     'UFP.1': 'ufp_acquirer', 'UPF.1': 'ufp_acquirer', 'Jefferies.1': 'jefferies_acquirer',
                     'Kepler (EU).1': 'kepler_acquirer',
                     'Olive Tree.1': 'olive_tree_acquirer', 'WallachBeth.1': 'wallachbeth_acquirer'}
    final_df.rename(columns=rename_scheme, inplace=True)
    import ipdb;
    ipdb.set_trace()

    final_df.to_sql(con=con, schema='prod_wic_db', name='risk_downside_trendlines', if_exists='append', index=False)
    con.close()


def parse_UFP_downsides():
    results_df = {}
    pdfFileObject = open('UFP.pdf', 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObject)

    on_downside_page = False
    for i in range(0, pdfReader.numPages):
        # creating a page object
        pageObj = pdfReader.getPage(i)

        # extracting text from page
        pageContent = pageObj.extractText()
        pageContent = pageContent.replace('-', '-\n')
        pageContent = pageContent.replace('\n', '')

        if pageContent.startswith('Downside Values'):
            on_downside_page = True
            # use "Dn1wk-DnChange" or "Dn1 Day-DnChange" for splitting page content
            try:
                pageContent = pageContent.split('Current Dn1 Week-DnChange')[2].split('UFP :')[
                    0]  # get just the downsides content
            except IndexError:
                pageContent = pageContent.split('Dn1 Day-DnChange')[2].split('UFP :')[0]
            pageContent = pageContent.rsplit('*', 1)[0].replace('*', '')  # remove special situations asterisk
            matching_groups = re.split('([A-Z| |\/]{2,}[$\£]\d+.\d{2}(?:[$\£]\d+.\d{2}|-)(?:\(*\d+\%\)*|-))',
                                       pageContent)
            matching_groups = list(filter(None, matching_groups))
            for matching_group in matching_groups:
                if '£' in matching_group:  # using only $
                    continue
                matching_group = matching_group.replace('-', '')
                # print(matching_group)
                ticker = re.split('[$\£]', matching_group)[0].strip()
                downside = re.search('(\d+.\d{2})', matching_group).group()
                results_df[ticker] = float(downside)
        else:
            if on_downside_page:  # skip after all pages related to the downside values are processed
                break
            else:
                continue
    return results_df


def parse_jefferies_downsides():
    jeff_df = pd.read_excel('Jeff.xlsx', skiprows=9, usecols='B:X', skipfooter=6)
    target_df = jeff_df[['Target', 'Downside']]
    target_df.columns = ['Ticker', 'Downside']
    acq_df = jeff_df[['Acquirer', 'Downside.1']]
    acq_df.columns = ['Ticker', 'Downside']
    merge_df = acq_df.append(target_df)
    merge_df = merge_df[merge_df['Downside'] > 0]  # filter out empty values
    merge_dict = {}
    for index, row in merge_df.iterrows():
        merge_dict[row['Ticker']] = float(row['Downside'])
    return merge_dict


def parse_cowen_downsides():
    file = open('Cowen.pdf', 'rb')
    pdf_reader = PyPDF2.PdfFileReader(file)
    downside_txt = pdf_reader.getPage(0).extractText()

    # get just table content
    downside_txt = downside_txt.split('Upside\nDownside\n')[1].split('Multiple Based\nQuantitative\n')[0]
    downside_cells = downside_txt.split('\n')
    downside_cells.pop()  # remove last empty cell
    downside_dict = {}

    # import ipdb;
    # ipdb.set_trace()
    while downside_cells:  # iterate through all the cells
        current_row = []  # place holder list for aggregating current row
        current_cell = ''
        # continue taking cells until reaching end of row
        while not (re.match(r'([+-]?([0-9]*[.])?[0-9]+ \/ [+-]?([0-9]*[.])?[0-9]+)',
                            current_cell) or current_cell == 'Positive Break'):
            current_cell = downside_cells.pop(0)
            current_row.append(current_cell)

        if current_cell == 'Positive Break':
            downside_cells.pop(0)
            current_row.append('-')  # padding place holder to keep format consistant

        ticker = current_row[1].replace('*', '').split(' ')[0]
        downside_dict[ticker] = float(current_row[-6])
        if not downside_cells:
            break
        ahead_cell = downside_cells[0].replace('*', '')
        if re.match(r'\w+ US', ahead_cell):
            ticker = ahead_cell.split(' ')[0]
            # if ticker == 'AVGO':  # handles exception where only two rows show up
            #     downside_dict[ticker] = float(downside_cells[2])
            #     downside_cells = downside_cells[4:]
            # else:
            downside_dict[ticker] = float(downside_cells[11])
            downside_cells = downside_cells[13:]

        # look ahead to see if it's same tradegroup

    return downside_dict


def parse_WallachBeth_downsides(selected_date):
    selected_date = selected_date.replace('-', '')
    # B:Target AA:RiskBreakPrice AC:Short AG:Acquirer AZ:Ratio
    wb_df = pd.read_excel('WB.xlsx', skiprows=8, usecols='B,AA,AC,AG,AZ', header=None, skipfooter=6)
    wb_df.columns = ['Target', 'Downside', 'short', 'Acquirer', 'ratio']
    acquirer_list = [x if x.endswith('EQUITY') else x + ' Equity' for x in wb_df['Acquirer']]
    acquirer_list.append('IBM Equity')
    # acquirer_list = acquirer_list[15:18]
    # acquirer_list = [x for x in wb_df['Acquirer']]
    slice_length = 5
    acq_slices = [acquirer_list[i:i + slice_length] for i in range(0, len(acquirer_list), slice_length)]
    acq_close_prices = {}
    for slice in acq_slices:
        bbg_data = bbgclient.bbgclient.get_secid2hp(slice,
                                                    'tickers',
                                                    start_date=selected_date,
                                                    end_date=selected_date,
                                                    api_host=bbgclient.bbgclient.get_next_available_host())
        for ticker, values in bbg_data.items():
            if values['Prices'] and values['Prices'][0]:
                acq_close_prices[ticker] = float(values['Prices'][0])

    def calc_acquirer_downside(row, acquire_close_dict):  # function used for calculating downside for wallachbeth
        acq_name = row['Acquirer'] if row['Acquirer'].endswith('EQUITY') else row['Acquirer'] + ' Equity'

        if acq_name not in acquire_close_dict:
            return None
        acq_close = acquire_close_dict[acq_name]
        try:
            row_ratio = float(row['ratio'])
        except ValueError:
            return None
        if np.isnan(row['short']) or np.isnan(row_ratio):
            return None
        return acq_close * (((row['short'] / row_ratio) / acq_close) + 1.0)

    wb_df['AcqDown'] = wb_df.apply(lambda x: calc_acquirer_downside(x, acq_close_prices), axis=1)
    target_df = wb_df[['Target', 'Downside']]
    target_df.columns = ['Ticker', 'Downside']
    acq_df = wb_df[['Acquirer', 'AcqDown']]
    acq_df.columns = ['Ticker', 'Downside']
    merge_df = acq_df.append(target_df)
    merge_df = merge_df[~np.isnan(merge_df['Downside'])]
    merge_dict = {}
    for index, row in merge_df.iterrows():
        merge_dict[row['Ticker']] = round(float(row['Downside']), 2)
    return merge_dict


def look_up_historical_trendline(ticker, field_name):
    # find the latest peer downside in Downside_Trendlines
    # if not found, return none
    latest_entry = Downside_Trendlines.objects.filter(tradegroup__contains=ticker).order_by('-date').first()
    if latest_entry:
        try:
            return getattr(latest_entry, field_name)
        except AttributeError:
            return None
        except Exception as e:
            logger.error(traceback.format_exc())
    else:
        return None


def populate_peer_downsides(selected_date=None):
    if not selected_date:
        selected_date = datetime.datetime.now().strftime('%Y-%m-%d')
        formula_df = pd.DataFrame.from_records(FormulaeBasedDownsides.objects.filter(IsExcluded='No').values())
    else:
        formula_df = pd.DataFrame.from_records(
            HistoricalFormulaeBasedDownsides.objects.filter(IsExcluded='No', Datestamp=selected_date).values())
    formula_df = formula_df[~formula_df['Underlying'].str.upper().str.contains('CVR EQUITY')]
    target_df = formula_df[formula_df['TargetAcquirer'] == 'Target']
    acq_df = formula_df[formula_df['TargetAcquirer'] == 'Acquirer']
    merge_df = pd.merge(target_df, acq_df, on='TradeGroup', how='left')
    merge_df = merge_df[['TradeGroup', 'base_case_x', 'base_case_y', 'outlier_y']]
    merge_df.rename(columns={'base_case_x': 'target_downside_base', 'base_case_y': 'acquirer_downside_base',
                             'outlier_y': 'acquirer_downside_outlier'}, inplace=True)
    last_date = Downside_Trendlines.objects.latest('date').date.strftime('%Y-%m-%d')
    last_downsides_df = pd.DataFrame.from_records(Downside_Trendlines.objects.filter(date=last_date).values())

    # MANUALLY CHANGE HERE TO POPULATE PEER DOWNSIDES
    ufp_data = parse_UFP_downsides()
    jeff_data = parse_jefferies_downsides()
    cowen_data = parse_cowen_downsides()
    # wb_data = parse_WallachBeth_downsides(selected_date)

    print('UFP:')
    print(dict(ufp_data.items()))
    print('Jefferies:')
    print(dict(jeff_data.items()))
    print("Cowen:")
    print(dict(cowen_data.items()))
    import ipdb
    ipdb.set_trace()

    # for index, row in merge_df.iterrows():
    #     trend_obj = Downside_Trendlines.objects.filter(tradegroup=row['TradeGroup'], date=selected_date).first()
    #     if not trend_obj:
    #         continue
    #     trend_obj.date = '2022-06-29'
    # if row['target_downside_base'] and not np.isnan(float(row['target_downside_base'])):
    #     trend_obj.target_downside_base = float(row['target_downside_base'])
    # if row['acquirer_downside_base'] and not np.isnan(float(row['acquirer_downside_base'])):
    #     trend_obj.acquirer_downside_base = float(row['acquirer_downside_base'])

    # trend_obj.save()
    try:
        entry_list = []
        for index, row in merge_df.iterrows():
            target_ticker, acquirer_ticker = row['TradeGroup'].split(' - ')
            trend_obj = Downside_Trendlines(date=selected_date,
                                            tradegroup=row['TradeGroup'],
                                            target_ticker=target_ticker,
                                            )
            if row['target_downside_base'] and not np.isnan(float(row['target_downside_base'])):
                trend_obj.target_downside_base = float(row['target_downside_base'])
            if row['acquirer_downside_base'] and not np.isnan(float(row['acquirer_downside_base'])):
                trend_obj.acquirer_downside_base = float(row['acquirer_downside_base'])

            if target_ticker in ufp_data:
                trend_obj.ufp_base = ufp_data[target_ticker]
            else:
                trend_obj.ufp_base = look_up_historical_trendline(target_ticker, 'ufp_base')

            if acquirer_ticker in ufp_data:
                trend_obj.ufp_acquirer = ufp_data[acquirer_ticker]
            else:
                trend_obj.ufp_acquirer = look_up_historical_trendline(acquirer_ticker, 'ufp_acquirer')

            if target_ticker in jeff_data:
                trend_obj.jefferies_base = jeff_data[target_ticker]
            else:
                trend_obj.jefferies_base = look_up_historical_trendline(target_ticker, 'jefferies_base')

            if acquirer_ticker in jeff_data:
                trend_obj.jefferies_acquirer = jeff_data[acquirer_ticker]
            else:
                trend_obj.jefferies_acquirer = look_up_historical_trendline(acquirer_ticker, 'jefferies_acquirer')

            if target_ticker in cowen_data:
                trend_obj.cowen_base = cowen_data[target_ticker]
            else:
                trend_obj.cowen_base = look_up_historical_trendline(target_ticker, 'cowen_base')

            if acquirer_ticker in cowen_data:
                trend_obj.cowen_acquirer = cowen_data[acquirer_ticker]
            else:
                trend_obj.cowen_acquirer = look_up_historical_trendline(acquirer_ticker, 'cowen_acquirer')

            # keeping previous notes
            last_row_df = last_downsides_df[last_downsides_df['tradegroup'] == row['TradeGroup']]
            if not last_row_df.empty:  # tradegroup in previous
                last_row_df = last_row_df.iloc[0]
                if last_row_df['notes'] and last_row_df['notes'] != '#ffffff|' and last_row_df[
                    'notes'] != '#c6efce|':  # if entry has notes last time
                    trend_obj.notes = last_row_df['notes']
            else:  # highlight new tradegroup
                trend_obj.notes = '#c6efce|'
            entry_list.append(trend_obj)
        import ipdb;
        ipdb.set_trace()
        Downside_Trendlines.objects.bulk_create(entry_list)
    except Exception as e:
        traceback.print_exc()


def merge_historical_downsides(merge_df, tradegroup):
    """
        function to join third-party downside values with historical downsides stored in HistoricalFormulaeBasedDownsides
    """
    # historical formatting
    historical_downsides_df = pd.DataFrame.from_records(
        HistoricalFormulaeBasedDownsides.objects.filter(TradeGroup=tradegroup).values())
    pivot_df = historical_downsides_df.pivot_table(index=['Datestamp', 'TradeGroup'],
                                                   columns='TargetAcquirer', aggfunc='first')
    pivot_df.columns = ['_'.join(col) for col in pivot_df.columns.values]
    pivot_df.reset_index(inplace=True)
    pivot_df['acquirer_ticker'] = pivot_df['TradeGroup'].str.split(' - ').str[1]

    # rename fields and to avoid key error of missing fields
    column_mapping = {
        'Datestamp': 'date',
        'TradeGroup': 'tradegroup',
        'base_case_Target': 'target_downside_base',
        'base_case_Acquirer': 'acquirer_downside_base',
        'TargetAcquirer': 'historical_targetacquirer',
        'outlier_Target': 'outlier',
        'day_one_downside_Target': 'day_one_downside',
        'backup_bid_Target': 'backup_bid'
    }
    # Filter the existing columns, if the column doesn't exist then don't include in new dataframe
    filtered_mapping = {old: new for old, new in column_mapping.items() if old in pivot_df.columns}
    # Rename the columns
    pivot_df.rename(columns=filtered_mapping, inplace=True)
    renamed_columns = list(filtered_mapping.values())
    # Select only the renamed columns
    pivot_df = pivot_df[renamed_columns]
    merge_df = pd.merge(merge_df, pivot_df, how='outer', on=['tradegroup', 'date'])

    # Merge the overlapping columns if there are any, and remove additioanl columns
    if 'acquirer_downside_base_x' in merge_df.columns:
        merge_df['acquirer_downside_base'] = merge_df['acquirer_downside_base_x'].combine_first(
            merge_df['acquirer_downside_base_y'])
        merge_df.drop(columns=['acquirer_downside_base_x', 'acquirer_downside_base_y'], inplace=True)
    if 'target_downside_base_x' in merge_df.columns:
        merge_df['target_downside_base'] = merge_df['target_downside_base_x'].combine_first(
            merge_df['target_downside_base_y'])
        merge_df.drop(columns=['target_downside_base_x', 'target_downside_base_y'], inplace=True)
    merge_df.sort_values(by='date', inplace=True)
    return merge_df


@shared_task
def copy_credit_team_export():
    """
        function to copy credit team export from email to local
    """
    file_name = 'Credit Team Export.csv'
    if os.name == 'nt':
        local_file_dir = 'K:/Credit/AAA Daily Performance/'
    else:
        local_file_dir = '/mnt/shares/KDrive/Credit/AAA Daily Performance/'
    local_file_path = local_file_dir + file_name
    try:
        # get the attachment from email
        get_attachment_by_subject('Credit Team Export', local_file_path)
        pd.read_csv(local_file_path)
        df = pd.read_csv(local_file_path, skiprows=1)
        df.to_excel(local_file_dir + 'Credit Team Export.xlsx', index=False)

        # remove the csv
        os.remove(local_file_path)
        dbutils.add_task_record()
    except Exception as e:
        dbutils.add_task_record(traceback.format_exc())
