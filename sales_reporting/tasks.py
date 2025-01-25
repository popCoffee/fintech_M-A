import io
import os
import pandas as pd
import django
from celery_progress.backend import ProgressRecorder
from django.conf import settings
import datetime
from threading import Thread

import holiday_utils
from email_utilities import send_email2
from celery import shared_task
from sqlalchemy import create_engine
from .render import *
import dbutils
import dfutils
import pydf
import xlsxwriter
import html2text
from django.conf import settings
import re
from slack_utils import get_channel_name
from django_slack import slack_message
import traceback
import logging

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WicPortal_Django.settings")
logger = logging.getLogger(__name__)
django.setup()


@shared_task
def email_weekly_sales_report(additional_recipients=None):

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)

    con = engine.connect()
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    bps_attributions = pd.read_sql_query('Select date, fund, sum(ytd_bps), sum(qtd_bps) from ' + \
                                         settings.CURRENT_DATABASE + '.funds_snapshot_tradegroupperformancefundnavbps' + \
                                         ' where date=(select max(date) from ' + settings.CURRENT_DATABASE + \
                                         '.funds_snapshot_tradegroupperformancefundnavbps) GROUP BY date, fund',
                                         con=con)
    bps_attributions.rename(columns={'date': 'Date', 'fund': 'Fund', 'sum(ytd_bps)': 'YTD (bps)',
                                     'sum(qtd_bps)': 'QTD (bps)'}, inplace=True)

    pivoted_bps_attributions = pd.pivot_table(bps_attributions, columns=['Fund'])
    pivoted_bps_attributions.columns.name = ' '

    pivoted_bps_attributions = pivoted_bps_attributions.round(decimals=2)

    def color_negative_red(val):
        try:
            if float(val) < 0:
                color = 'red'
            else:
                color = 'green'

            return 'color: %s' % color
        except Exception:
            return 'color: black'

    styles = [
        {'selector': 'tr:hover td', 'props': [('background-color', 'yellow')]},
        {'selector': 'th, td', 'props': [('border', '1px solid black'),
                                         ('padding', '12px'),
                                         ('text-align', 'center')]},
        {'selector': 'th', 'props': [('font-weight', 'bold')]},
        {'selector': '', 'props': [('border-collapse', 'collapse'),
                                   ('border', '1px solid black'),
                                   ('margin', '0 auto')]}
    ]

    pivoted_bps_attributions = pivoted_bps_attributions.style.applymap(color_negative_red,
                                                                       ).set_table_styles(styles)

    aed_bucket_weightings = pd.read_sql_query("SELECT `date`, fund, Bucket, SUM(alpha_exposure) FROM "
                                              "prod_wic_db.exposures_exposuressnapshot "
                                              "WHERE `date` = (SELECT MAX(`date`) FROM prod_wic_db.exposures_exposuressnapshot) "
                                              "AND fund LIKE 'AED' "
                                              "GROUP BY `date`, fund, bucket", con=con)

    aed_bucket_weightings = aed_bucket_weightings.round(decimals=2)
    del aed_bucket_weightings['date']
    del aed_bucket_weightings['fund']
    aed_bucket_weightings.columns = ['Bucket', 'Alpha Exposure']
    aed_bucket_weightings['Alpha Exposure'] = aed_bucket_weightings['Alpha Exposure'].apply(lambda x: str(x) + " %")
    pivoted_aed_bucket_weightings = pd.pivot_table(aed_bucket_weightings, columns=['Bucket'], aggfunc='first')
    pivoted_aed_bucket_weightings.columns.name = ' '
    pivoted_aed_bucket_weightings = pivoted_aed_bucket_weightings.style.set_table_styles(styles)

    # Buckets Contribution Section

    aed_buckets_performance = pd.read_sql_query("SELECT * FROM wic.buckets_snapshot where Fund like 'AED' ", con=con)

    aed_buckets_performance['EndDate'] = aed_buckets_performance['EndDate'].apply(
        lambda x: x if x is None else pd.to_datetime(x).strftime('%Y-%m-%d'))
    aed_buckets_performance['InceptionDate'] = aed_buckets_performance['InceptionDate'].apply(
        lambda x: x if x is None else pd.to_datetime(x).strftime('%Y-%m-%d'))

    metrics2include = [('P&L(bps)', 'YTD'),
                       ('P&L(bps)', 'QTD')
                       ]

    metric2display_name = {'P&L(bps)': ''}
    metric2unit = {'P&L(bps)': 'bps'}

    metrics_df = pd.DataFrame([dfutils.json2row(json) for json in aed_buckets_performance['Metrics in NAV JSON']])
    metrics_df.index = aed_buckets_performance.index

    for (metric, period) in metrics2include:
        unit = metric2unit[metric]
        disp_name = metric2display_name[metric]
        display_colname = disp_name + ' ' + period + '(' + unit + ')'
        aed_buckets_performance[display_colname] = metrics_df[metric + '|' + period]

    del aed_buckets_performance['Metrics in NAV JSON']
    del aed_buckets_performance['Metrics in NAV notes JSON']
    del aed_buckets_performance['Metrics in Bet JSON']
    del aed_buckets_performance['Metrics in Bet notes JSON']

    base_cols = ['Fund', 'Bucket', 'InceptionDate', 'EndDate']
    bps_cols = [' YTD(bps)', ' QTD(bps)']
    aed_buckets_bps_df = aed_buckets_performance[base_cols + bps_cols].sort_values(by=' YTD(bps)')
    aed_buckets_bps_df.rename(columns={' YTD(bps)': 'YTD_bps', ' QTD(bps)': 'QTD_bps'}, inplace=True)

    del aed_buckets_bps_df['Fund']
    pivoted_aed_buckets_bps_df = pd.pivot_table(aed_buckets_bps_df, columns=['Bucket'])

    pivoted_aed_buckets_bps_df = pivoted_aed_buckets_bps_df.style.applymap(color_negative_red,
                                                                           ).set_table_styles(styles)

    # P&L Monitors Section
    df = pd.read_sql_query("SELECT * FROM " + settings.CURRENT_DATABASE + ".realtime_pnl_impacts_pnlmonitors"
                                                                          " where last_updated = "
                                                                          "(select max(last_updated) from "
                           + settings.CURRENT_DATABASE + ".realtime_pnl_impacts_pnlmonitors)",
                           con=con)

    # Close Connection
    con.close()
    del df['last_updated']
    df.rename(columns={'fund': 'Fund', 'ytd_active_deal_losses': 'YTD Active Deal Losses',
                       'ytd_closed_deal_losses': 'YTD Closed Deal Losses', 'ann_loss_budget_perc': 'Loss Budget',
                       'investable_assets': 'AUM', 'gross_ytd_pnl': 'Gross YTD P&L',
                       'ann_gross_pnl_target_perc': 'Ann. Gross P&L Target %', 'time_passed': 'Time Passed',
                       'gross_ytd_return': 'Gross YTD Return',
                       'ann_gross_pnl_target_dollar': 'Ann. Gross P&L Target $',
                       'ytd_pnl_perc_target': 'YTD P&L % of Target', 'ann_loss_budget_dollar': 'Ann Loss Budget $',
                       'ytd_total_loss_perc_budget': 'YTD Total Loss % of Budget'}, inplace=True)

    pivoted = pd.pivot_table(df, columns=['Fund'], aggfunc=lambda x: x, fill_value='')

    pivoted = pivoted[['ARB', 'MACO', 'MALT', 'AED', 'CAM', 'LG', 'LEV', 'TACO', 'EVNT']]
    pivoted = pivoted.reindex(['AUM',
                               'Ann. Gross P&L Target %',
                               'Gross YTD Return',
                               'YTD P&L % of Target',
                               'Time Passed',
                               'Ann. Gross P&L Target $',
                               'Gross YTD P&L',
                               'Loss Budget',
                               'YTD Total Loss % of Budget',
                               'Time Passed',
                               'Ann Loss Budget $',
                               'YTD Closed Deal Losses',
                               'YTD Active Deal Losses',
                               ])
    df1 = pivoted.iloc[:7].copy()
    df2 = pivoted.iloc[7:].copy()
    df3 = pd.DataFrame([list(pivoted.columns.values)], columns=list(pivoted.columns.values))
    df1 = df1.append(df3)
    df1.index.values[5] = '* Ann. Gross P&L Target $'
    df1.index.values[7] = 'Loss Budgets'
    df1 = df1.append(df2)
    df1.index.values[8] = 'Ann Loss Budget %'
    df1.index.values[0] = 'Investable Assets'
    df1.index.values[4] = 'Time Passed%'
    df1.index.values[10] = 'Time Passed %'
    df1.index.values[11] = '* Ann Loss Budget $'

    df1 = df1.style.set_table_styles(styles)

    # Winners/Losers
    tg_snapshot_df = dbutils.Wic.get_tradegroups_snapshot()

    aed_qtd_winners, aed_qtd_losers, aed_ytd_winners, aed_ytd_losers, aed_ytd_active_winners, aed_ytd_active_losers = get_fund_winners_losers(
        tg_snapshot_df, 'AED')

    arb_qtd_winners, arb_qtd_losers, arb_ytd_winners, arb_ytd_losers, arb_ytd_active_winners, arb_ytd_active_losers = get_fund_winners_losers(
        tg_snapshot_df, 'ARB')

    taco_qtd_winners, taco_qtd_losers, taco_ytd_winners, taco_ytd_losers, taco_ytd_active_winners, taco_ytd_active_losers = get_fund_winners_losers(
        tg_snapshot_df, 'TACO')

    aed_qtd_winners = aed_qtd_winners.style.applymap(color_negative_red).set_table_styles(styles)
    aed_qtd_losers = aed_qtd_losers.style.applymap(color_negative_red).set_table_styles(styles)
    aed_ytd_winners = aed_ytd_winners.style.applymap(color_negative_red).set_table_styles(styles)
    aed_ytd_losers = aed_ytd_losers.style.applymap(color_negative_red).set_table_styles(styles)
    aed_ytd_active_winners = aed_ytd_active_winners.style.applymap(color_negative_red).set_table_styles(styles)
    aed_ytd_active_losers = aed_ytd_active_losers.style.applymap(color_negative_red).set_table_styles(styles)

    arb_qtd_winners = arb_qtd_winners.style.applymap(color_negative_red).set_table_styles(styles)
    arb_qtd_losers = arb_qtd_losers.style.applymap(color_negative_red).set_table_styles(styles)
    arb_ytd_winners = arb_ytd_winners.style.applymap(color_negative_red).set_table_styles(styles)
    arb_ytd_losers = arb_ytd_losers.style.applymap(color_negative_red).set_table_styles(styles)
    arb_ytd_active_winners = arb_ytd_active_winners.style.applymap(color_negative_red).set_table_styles(styles)
    arb_ytd_active_losers = arb_ytd_active_losers.style.applymap(color_negative_red).set_table_styles(styles)

    taco_qtd_winners = taco_qtd_winners.style.applymap(color_negative_red).set_table_styles(styles)
    taco_qtd_losers = taco_qtd_losers.style.applymap(color_negative_red).set_table_styles(styles)
    taco_ytd_winners = taco_ytd_winners.style.applymap(color_negative_red).set_table_styles(styles)
    taco_ytd_losers = taco_ytd_losers.style.applymap(color_negative_red).set_table_styles(styles)
    taco_ytd_active_winners = taco_ytd_active_winners.style.applymap(color_negative_red).set_table_styles(styles)
    taco_ytd_active_losers = taco_ytd_active_losers.style.applymap(color_negative_red).set_table_styles(styles)

    params = {'current_date': datetime.datetime.now().strftime('%Y-%m-%d'),
              'bps_attributions': pivoted_bps_attributions.render(),
              'aed_bucket_weightings': pivoted_aed_bucket_weightings.render(),
              'pnl_monitors': df1.render(),
              'buckets_contribution_bps': pivoted_aed_buckets_bps_df.render(),
              'aed_qtd_winners': aed_qtd_winners.hide_index().render(),
              'aed_qtd_losers': aed_qtd_losers.hide_index().render(),
              'aed_ytd_winners': aed_ytd_winners.hide_index().render(),
              'aed_ytd_losers': aed_ytd_losers.hide_index().render(),
              'aed_ytd_active_winners': aed_ytd_active_winners.hide_index().render(),
              'aed_ytd_active_losers': aed_ytd_active_losers.hide_index().render(),
              'arb_qtd_winners': arb_qtd_winners.hide_index().render(),
              'arb_qtd_losers': arb_qtd_losers.hide_index().render(),
              'arb_ytd_winners': arb_ytd_winners.hide_index().render(),
              'arb_ytd_losers': arb_ytd_losers.hide_index().render(),
              'arb_ytd_active_winners': arb_ytd_active_winners.hide_index().render(),
              'arb_ytd_active_losers': arb_ytd_active_losers.hide_index().render(),
              'taco_qtd_winners': taco_qtd_winners.hide_index().render(),
              'taco_qtd_losers': taco_qtd_losers.hide_index().render(),
              'taco_ytd_winners': taco_ytd_winners.hide_index().render(),
              'taco_ytd_losers': taco_ytd_losers.hide_index().render(),
              'taco_ytd_active_winners': taco_ytd_active_winners.hide_index().render(),
              'taco_ytd_active_losers': taco_ytd_active_losers.hide_index().render(),
              }

    file = Render.render_to_file('sales_weekly_template.html', params)
    recipients = ['risk@wicfunds.com', 'cbuntic@wicfunds.com', 'kfeeney@wicfunds.com', 'jnespoli@wicfunds.com',
                  'cfazioli@wicfunds.com', 'cwalker@wicfunds.com', 'peisenmann@wicfunds.com']
    if additional_recipients:
        recipients = additional_recipients

    thread = Thread(target=send_email2, args=(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD,
                                              recipients, "(Risk Automation) Weekly Sales Report - " +
                                              datetime.datetime.now().strftime('%Y-%m-%d'), 'dispatch@wicfunds.com',
                                              'Please find attached Weekly Sales Report!', file))
    thread.start()
    dbutils.add_task_record()
    return "Completed Task - (Weekly Sales Reporting)"


def get_fund_winners_losers(tg_snapshot_df, fund_code):
    f_df = tg_snapshot_df[tg_snapshot_df['Fund'] == fund_code].copy()
    metrics_df = pd.DataFrame([dfutils.json2row(json) for json in f_df['Metrics in NAV JSON']])
    metrics_df.index = f_df.index
    f_df['YTD(bps)'] = metrics_df['P&L(bps)|YTD']
    f_df['YTD($)'] = metrics_df['P&L($)|YTD']
    f_df['QTD(bps)'] = metrics_df['P&L(bps)|QTD']
    f_df['QTD($)'] = metrics_df['P&L($)|QTD']

    tg_ytd_df = f_df[~pd.isnull(f_df['YTD(bps)'])][
        ['TradeGroup', 'InceptionDate', 'EndDate', 'Status', 'YTD(bps)', 'YTD($)']].sort_values(by='YTD(bps)',
                                                                                                ascending=False).rename(
        columns={'YTD(bps)': 'bps', 'YTD($)': '$'})
    active_tg_ytd_df = tg_ytd_df[tg_ytd_df['Status'] == 'ACTIVE'].sort_values(by='bps', ascending=False)
    tg_qtd_df = f_df[~pd.isnull(f_df['QTD(bps)'])][
        ['TradeGroup', 'InceptionDate', 'EndDate', 'Status', 'QTD(bps)', 'QTD($)']].sort_values(by='QTD(bps)',
                                                                                                ascending=False).rename(
        columns={'QTD(bps)': 'bps', 'QTD($)': '$'})

    string_cols = ['bps', '$']
    tg_qtd_df[string_cols] = tg_qtd_df[string_cols].astype(str)
    active_tg_ytd_df[string_cols] = active_tg_ytd_df[string_cols].astype(str)
    tg_qtd_df[string_cols] = tg_qtd_df[string_cols].astype(str)

    qtd_winners = tg_qtd_df.head(5)
    qtd_losers = tg_qtd_df.tail(5)

    ytd_winners = tg_ytd_df.head(5)
    ytd_losers = tg_ytd_df.tail(5)

    ytd_active_winners = active_tg_ytd_df.head(5)
    ytd_active_losers = active_tg_ytd_df.tail(5)

    return qtd_winners, qtd_losers, ytd_winners, ytd_losers, ytd_active_winners, ytd_active_losers

def generate_sales_report(fund_name, top_count):
    ordered_cols = ['Rank', 'tradegroup', 'sleeve', 'bucket', 'target_name', 'acquirer_name', 'deal_description',
                    'deal_rationale',
                    'nature_of_bid', 'transaction_type', 'payment_type', 'deal_value', 'alpha_exposure',
                    'hedge_exposure',
                    'net_exposure', 'gross_exposure', 'capital', 'inception_date', 'announced_date',
                    'days_since_announce', 'expected_completion_quarter',
                    'target_to_acquirer_termination_fees',
                    'acquirer_to_target_termination_fess', 'activists_involved', 'approved_approvals',
                    'pending_approvals',
                    'itd_bps', 'ytd_bps', 'qtd_bps', 'mtd_bps', 'days_30_bps', 'days_5_bps', 'days_1_bps']
    title_names = ["Rank", "TradeGroup", "Sleeve", "Bucket", "Target", "Acquirer", "Description", "Rationale",
                   "Friendly?",
                   "Transaction Type", "Consideration Type", "Deal Value / Sh", "Alpha Exposure (%)",
                   "Hedge Exposure (%)",
                   "Net Exposure (%)", "Gross Exposure (%)", "Capital (%)", "Inception Date", "Announced Date",
                   'Days since announcement',
                   "Expected Completion Date", "Target Termination Fee ($ million)",
                   "Acquirer Termination Fee ($ million)",
                   "Activist Involved?", "Completed Approvals", "Pending Approvals", "ITD (bps)", "YTD (bps)",
                   "QTD (bps)", "MTD (bps)", "30D (bps)", "5D (bps)", "1D (bps)"]
    FUND = fund_name
    TOP_COUNT = top_count  # Edit here to change the number of tradegroups to include in the top
    WARNING_OFFSET = 2  # Number of rows the warning takes
    WARNING_WIDTH = 8
    HEADER_OFFSET = WARNING_OFFSET + 3  # Warning +  Title row(2) + ColumnName row (1)
    COL_COUNT = len(title_names)
    SUBTITLE_OFFSET = 5  # Change this to allow longer subtitle tab
    CASH_LEVEL_OFFSET = 2  # How many rows after main table to start cash level tabs
    MAX_NOTE_HEIGHT = 35  # Change this it allow larger notes display
    WARNING_MESSAGE = "Water Island Capital - INTERNAL USE ONLY"


    try:
        engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                               + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
        con = engine.connect()

        output_df = pd.DataFrame()  # DataFrame that contains the output data
        # selected_date = "2020-09-18"  # Set default date

        # get newest date
        date_query = "SELECT MAX(date) as m_d FROM exposures_exposuressnapshot"
        date_result = pd.read_sql_query(date_query, con=con)
        selected_date = date_result['m_d'][0].strftime('%Y-%m-%d')  # Using the latest date in the database
        selected_date_obj = datetime.datetime.strptime(selected_date, '%Y-%m-%d')


        # get TradeGroup, AlphaExposure, HedgeExposure, NetExposure, GrossExposure, Capital
        # from exposures_exposuressnapshot
        if FUND == "ARB":
            snapshot_query = "SELECT tradegroup, sleeve, bucket, alpha_exposure, hedge_exposure, net_exposure, gross_exposure, capital " \
                             " FROM exposures_exposuressnapshot WHERE date ='" + selected_date \
                             + "' AND fund = '" + FUND + "' AND sleeve = 'MERGER ARBITRAGE' AND tradegroup <> 'AEDNX' " \
                             + "ORDER BY alpha_exposure DESC LIMIT " + str(TOP_COUNT)
        else:
            snapshot_query = "SELECT tradegroup, sleeve, bucket, alpha_exposure, hedge_exposure, net_exposure, gross_exposure, capital " \
                             " FROM exposures_exposuressnapshot WHERE date ='" + selected_date \
                             + "' AND fund = '" + FUND + "' AND tradegroup <> 'AEDNX' " \
                             + "ORDER BY alpha_exposure DESC LIMIT " + str(TOP_COUNT)
        output_df = pd.read_sql_query(snapshot_query, con=con)

        # DataFrames used for collecting data from different tables
        total_fund_performance_df = pd.DataFrame()
        total_deal_df = pd.DataFrame()
        total_detail_df = pd.DataFrame()
        total_risk_df = pd.DataFrame()


        for index, row in output_df.iterrows():
            tradegroup = row['tradegroup']

            # get ITD, YTD, QTD, MTD, 30D, 50D. 1D from funds_snapshot_tradegroupperformancefundnavbps
            if FUND == "ARB":
                funds_performance_query = "SELECT tradegroup, itd_bps, ytd_bps, qtd_bps, mtd_bps, days_30_bps," \
                                          " days_5_bps, days_1_bps FROM funds_snapshot_tradegroupperformancefundnavbps " \
                                          " WHERE  fund = '" + FUND + "' AND status = 'ACTIVE' AND sleeve = 'MERGER ARBITRAGE'" \
                                                                      " AND tradegroup = '" + tradegroup + "' AND date = '" + selected_date + "'"
            else:
                funds_performance_query = "SELECT tradegroup, itd_bps, ytd_bps, qtd_bps, mtd_bps, days_30_bps," \
                                          " days_5_bps, days_1_bps FROM funds_snapshot_tradegroupperformancefundnavbps " \
                                          " WHERE  fund = '" + FUND + "' AND status = 'ACTIVE'" \
                                                                      " AND tradegroup = '" + tradegroup + "' AND date = '" + selected_date + "'"
            fund_performance_result = pd.read_sql_query(funds_performance_query, con=con)

            total_fund_performance_df = total_fund_performance_df.append(fund_performance_result, ignore_index=True)

            # get DealValue, InceptionDate from risk_ma_deals
            deal_value_query = "SELECT id, deal_name, deal_value, created, action_id FROM risk_ma_deals " \
                               " WHERE deal_name = '" + tradegroup + "' AND status = 'ACTIVE'"
            deal_value_result = pd.read_sql_query(deal_value_query, con=con)
            total_deal_df = total_deal_df.append(deal_value_result[['deal_name', 'created', 'deal_value']])

            if len(deal_value_result.index) > 0:  # in case if null value received
                deal_id = deal_value_result.get('id')[0]
                action_id = deal_value_result.get('action_id')[0]

                # get Target, Acuirer, DealDescription, Friendly,TransactionType, ConsiderationType, AnnouncedDate,
                # ExpectedCompletionDate, TargetTerminationFee, AcquirerTerminationFee,
                # CompletedApprovals, PendingApprovals from risk_madealsactioniddetails
                # get target_ticker for getting news and notes later
                deal_detail_query = "SELECT target_name, target_ticker, acquirer_name, deal_description," \
                                    " nature_of_bid, transaction_type, " \
                                    "payment_type, announced_date, expected_completion_date, " \
                                    "target_to_acquirer_termination_fees, acquirer_to_target_termination_fess, " \
                                    "approved_approvals, pending_approvals " \
                                    " FROM risk_madealsactioniddetails WHERE action_id = '" + str(action_id) + "'"

                deal_detail_result = pd.read_sql_query(deal_detail_query, con=con)
                deal_detail_result['tradegroup'] = tradegroup
                deal_detail_result['action_id'] = action_id

                total_detail_df = total_detail_df.append(deal_detail_result)

                # duplicate hostile_friendly found?
                # get Rationale, ActivistInvolved from risk_ma_deals_risk_factors
                # from risk_ma_deals_risk_factors
                risk_factor_query = "SELECT deal_rationale, activists_involved " \
                                    " FROM risk_ma_deals_risk_factors WHERE deal_id = '" + str(deal_id) + "'"
                risk_factor_result = pd.read_sql_query(risk_factor_query, con=con)
                risk_factor_result['tradegroup'] = tradegroup
                total_risk_df = total_risk_df.append(risk_factor_result)
            else:  # data missing due to two reason: not merger arb or insufficient data for merger arb
                if row['sleeve'] != "MERGER ARBITRAGE":



                    detail_result = pd.DataFrame(data={"tradegroup": [tradegroup], 'deal_description': [""]})
                    total_detail_df = total_detail_df.append(detail_result)

        output_df = output_df.merge(total_fund_performance_df, how='outer', on='tradegroup')

        # Rename column before merging
        total_deal_df = total_deal_df.rename(columns={'deal_name': 'tradegroup', 'created': 'inception_date'})
        if not total_deal_df.empty:
            output_df = output_df.merge(total_deal_df, how='outer', on='tradegroup')
        if not total_detail_df.empty:
            output_df = output_df.merge(total_detail_df, how='outer', on='tradegroup')
        if not total_risk_df.empty:
            output_df = output_df.merge(total_risk_df, how='outer', on='tradegroup')

        if 'deal_value' in output_df.columns:
            output_df['deal_value'] = "$" + output_df['deal_value'].astype(float).round(2).fillna("").astype(str)
        else:
            output_df['deal_value'] = ''
        if 'target_to_acquirer_termination_fees' in output_df.columns:
            output_df['target_to_acquirer_termination_fees'] = \
                output_df['target_to_acquirer_termination_fees'].astype(float).round(2).fillna("").astype(str)
        else:
            output_df['target_to_acquirer_termination_fees'] = ''
        if 'acquirer_to_target_termination_fess' in output_df.columns:
            output_df['acquirer_to_target_termination_fess'] = \
                output_df['acquirer_to_target_termination_fess'].astype(float).round(2).fillna("").astype(str)
        else:
            output_df['acquirer_to_target_termination_fess'] = ''

        # fill in empty tables
        for missing_col in ordered_cols+['target_ticker','expected_completion_date']:
            if missing_col not in output_df.columns:
                output_df[missing_col] = ''

        # In case if there are duplicates in the table

        output_df = output_df.round(1)  # round to 1 decimal
        output_df = output_df.drop_duplicates()

        # calculate total fund percentage
        total_asset_query = "SELECT aum FROM positions_and_pnl_tradegroupmaster" \
                            " WHERE fund = '" + FUND + f"' AND tradegroup = 'CASH' AND date = '{selected_date}'"
        total_asset_result = pd.read_sql_query(total_asset_query, con=con)
        total_asset_value = total_asset_result['aum'][0]

        total_alpha_query = "SELECT SUM(alpha_exposure) as s FROM exposures_exposuressnapshot" \
                            " WHERE fund = '" + FUND + f"' AND longshort = 'LONG' AND date = '{selected_date}'"
        total_alpha_result = pd.read_sql_query(total_alpha_query, con=con)
        total_alpha_sum = total_alpha_result['s'][0]

        if FUND == 'ARB':
            total_performance_query = "SELECT sum(itd_bps) as itd_sum, sum(ytd_bps) as ytd_sum ," \
                                      " sum(qtd_bps) as qtd_sum, sum(mtd_bps) as mtd_sum," \
                                      " sum(days_30_bps) as thirty_day_sum, sum(days_5_bps) as five_day_sum," \
                                      " sum(days_1_bps) as one_day_sum" \
                                      " FROM funds_snapshot_tradegroupperformancefundnavbps" \
                                      " WHERE fund = '" + FUND + "' AND status = 'ACTIVE' AND sleeve = 'MERGER ARBITRAGE'" \
                                                                 f" AND date = '{selected_date}'"
        else:
            total_performance_query = "SELECT sum(itd_bps) as itd_sum, sum(ytd_bps) as ytd_sum ," \
                                      " sum(qtd_bps) as qtd_sum, sum(mtd_bps) as mtd_sum," \
                                      " sum(days_30_bps) as thirty_day_sum, sum(days_5_bps) as five_day_sum," \
                                      " sum(days_1_bps) as one_day_sum" \
                                      " FROM funds_snapshot_tradegroupperformancefundnavbps" \
                                      " WHERE fund = '" + FUND + "' AND status = 'ACTIVE'" \
                                                                 f" AND date = '{selected_date}'"
        total_performance_result = pd.read_sql_query(total_performance_query, con=con)

        total_itd = float(total_performance_result['itd_sum'])
        total_ytd = float(total_performance_result['ytd_sum'])
        total_qtd = float(total_performance_result['qtd_sum'])
        total_mtd = float(total_performance_result['mtd_sum'])
        total_thirty_day = float(total_performance_result['thirty_day_sum'])
        total_five_day = float(total_performance_result['five_day_sum'])
        total_one_day = float(total_performance_result['one_day_sum'])
        cash_level = (100.0 - total_alpha_sum) * 0.01 * total_asset_value
        capital_sum = output_df['capital'].sum().round(2)

        # add percentage sign
        itd_sum = output_df['itd_bps'].sum()
        ytd_sum = output_df['ytd_bps'].sum()
        qtd_sum = output_df['qtd_bps'].sum()
        mtd_sum = output_df['mtd_bps'].sum()
        thirty_day_sum = output_df['days_30_bps'].sum()
        five_day_sum = output_df['days_5_bps'].sum()
        one_day_sum = output_df['days_1_bps'].sum()

        output_df['alpha_exposure'] = output_df['alpha_exposure'].fillna('').astype(str) + "%"
        output_df['hedge_exposure'] = output_df['hedge_exposure'].fillna('').astype(str) + "%"
        output_df['net_exposure'] = output_df['net_exposure'].fillna('').astype(str) + "%"
        output_df['gross_exposure'] = output_df['gross_exposure'].fillna('').astype(str) + "%"
        output_df['capital'] = output_df['capital'].fillna('').astype(str) + "%"
        output_df['itd_bps'] = output_df['itd_bps'].fillna('').astype(str)
        output_df['ytd_bps'] = output_df['ytd_bps'].fillna('').astype(str)
        output_df['qtd_bps'] = output_df['qtd_bps'].fillna('').astype(str)
        output_df['mtd_bps'] = output_df['mtd_bps'].fillna('').astype(str)
        output_df['days_30_bps'] = output_df['days_30_bps'].fillna('').astype(str)
        output_df['days_5_bps'] = output_df['days_5_bps'].fillna('').astype(str)
        output_df['days_1_bps'] = output_df['days_1_bps'].fillna('').astype(str)

        output_df = output_df.replace(['%', '$'], '')
        # import  ipdb;
        # ipdb.set_trace()
        # calculate days since announcement
        if 'announced_date' in output_df.columns and not (output_df['announced_date'] == "").all():
            output_df['days_since_announce'] = selected_date_obj.date() - output_df['announced_date']
            output_df['days_since_announce'] = output_df['days_since_announce'].astype('timedelta64[D]')

        # format date quater format: "YYYY QQ"
        if 'expected_completion_date' in output_df.columns and 'expected_completion_quarter' in output_df.columns and \
                not (output_df['expected_completion_date'] == "").all() and \
                not (output_df['expected_completion_quarter'] == "").all():
            output_df['expected_completion_quarter'] = pd.to_datetime(output_df['expected_completion_date'])
            output_df['expected_completion_quarter'] = pd.PeriodIndex(output_df['expected_completion_quarter'],freq='Q')
            output_df['expected_completion_quarter'] = (output_df['expected_completion_quarter'].astype(str).str[:-2] +
                        " " + output_df['expected_completion_quarter'].astype(str).str[-2:]).replace('N aT', "")

        # format inception date and announced date to realign to center
        if 'inception_date' in output_df.columns:
            output_df['inception_date'] = output_df['inception_date'].astype(str).replace('nan', '')

        if 'announced_date' in output_df.columns:
            output_df['announced_date'] = output_df['announced_date'].astype(str).replace('nan', '')
        else:
            'announced_date'
        if 'expected_completion_date' in output_df.columns:
            output_df['expected_completion_date'] = output_df['expected_completion_date'].astype(str).replace('nan', '')
        # #  END COLLECTING DATA

        # FOR LOCAL TESTING ONLY: SAVE/LOAD TEST DATA TO CSV
        # output_df.to_csv('test_data.csv', index=False)
        # output_df = pd.read_csv('test_data.csv')
        # END LOAD TEST DATA FROM CSV

        # select data for populating detailed page
        detail_report_df = output_df[[
            'target_ticker', 'tradegroup', 'target_name', 'acquirer_name', 'payment_type', 'deal_value',
            'announced_date', 'expected_completion_date']]

        # REORDER DATA
        # Reorder columns based on example excel
        output_df = output_df.reset_index()
        output_df['Rank'] = output_df.index
        output_df['Rank'] = output_df['Rank'] + 1  # start ranking from 1

        output_df = output_df[ordered_cols]  # reorder titles
        output_df.columns = title_names  # rename title to readable format

        # store file in buffer to avoid permission error when storing in drive
        file_buffer = io.BytesIO()
        writer = pd.ExcelWriter(file_buffer, engine='xlsxwriter',
                                options={'strings_to_numbers': True, 'in_memory': True})
        table_width = len(output_df.columns)

        # Write data to excel
        output_df.to_excel(writer, sheet_name='Top ' + str(TOP_COUNT),
                           startrow=HEADER_OFFSET, startcol=0, index=False, header=False)

        workbook = writer.book
        worksheet = writer.sheets['Top ' + str(TOP_COUNT)]

        html_format = workbook.add_format({'font_color': 'blue', 'underline': 1, 'bold': True})
        html_warning_format = workbook.add_format({'font_color': 'red', 'underline': 1, 'bold': True})
        html_format.set_align('center')
        html_warning_format.set_align('center')
        output_df['Announced Date'].replace('None', "", inplace=True)
        output_df['announced_year'] = pd.DatetimeIndex(output_df['Announced Date']).year
        output_df['announced_month'] = pd.DatetimeIndex(output_df['Announced Date']).month

        # add hyperlink for tradegroup
        for index, entry in output_df.iterrows():
            tradegroup = entry['TradeGroup']
            link = "http://192.168.0.16:8000/position_stats/get_tradegroup_story?TradeGroup=" \
                   f"{tradegroup.replace(' ', '+')}&Fund={FUND}"

            # warning for two months end
            if selected_date_obj.month - entry['announced_month'] < 3 \
                    and selected_date_obj.year == entry['announced_year']:
                worksheet.write_url(HEADER_OFFSET + index, 1, link, string=tradegroup, cell_format=html_warning_format)
            else:
                worksheet.write_url(HEADER_OFFSET + index, 1, link, string=tradegroup, cell_format=html_format)

        # Begin format declaration
        # format for top table
        blank_format = workbook.add_format()
        blank_format.set_pattern(1)
        blank_format.set_bg_color('#f8cbad')

        default_cell_format = workbook.add_format(
            {'top': 1, 'bottom': 1, 'left': 1, 'right': 1})  # add border to cells
        default_cell_format.set_align('center')
        default_cell_format.set_align('vcenter')

        rightallign_format = workbook.add_format()
        rightallign_format.set_align('right')
        rightallign_format.set_align('vcenter')

        title_format = workbook.add_format({'font_size': '18', 'align': 'center', 'bold': True})
        title_format.set_top(2)
        title_format.set_bottom(2)
        title_format.set_align('vcenter')

        subtitle_format = workbook.add_format({'align': 'center', 'bottom': 2, 'top': 2, 'left': 2, 'bold': True})
        subtitle_format.set_align('vcenter')

        header_format = workbook.add_format({'bold': True, 'text_wrap': True})
        header_format.set_align('center')
        header_format.set_align('vcenter')

        warning_format = workbook.add_format({'bold': True, 'align': 'center', 'font_size': 18})
        warning_format.set_fg_color("#FFFF00")
        warning_format.set_align('vcenter')
        # format for individual detailed pages
        cell_color = '#d9e1f2'

        notes_other_odd_format = workbook.add_format({'top': 1, 'bottom': 1, 'left': 1, 'right': 1})
        notes_other_odd_format.set_align('center')
        notes_other_odd_format.set_align('vcenter')
        notes_other_odd_format.set_fg_color(cell_color)

        notes_other_even_format = workbook.add_format({'top': 1, 'bottom': 1, 'left': 1, 'right': 1})
        notes_other_even_format.set_align('center')
        notes_other_even_format.set_align('vcenter')

        notes_odd_format = workbook.add_format({'top': 1, 'bottom': 1, 'left': 1, 'right': 1})
        notes_odd_format.set_align('left')
        notes_odd_format.set_align('top')
        notes_odd_format.set_fg_color(cell_color)
        notes_odd_format.set_text_wrap()

        notes_even_format = workbook.add_format({'top': 1, 'bottom': 1, 'left': 1, 'right': 1})
        notes_even_format.set_align('left')
        notes_even_format.set_align('top')
        notes_even_format.set_text_wrap()

        news_even_format = workbook.add_format(
            {'top': 1, 'bottom': 1, 'left': 1, 'right': 1, 'font_color': 'blue', 'bold': 1})
        news_even_format.set_align('center')
        news_even_format.set_align('vcenter')

        news_odd_format = workbook.add_format(
            {'top': 1, 'bottom': 1, 'left': 1, 'right': 1, 'font_color': 'blue', 'bold': 1, 'fg_color': cell_color})
        news_odd_format.set_align('center')
        news_odd_format.set_align('vcenter')

        top_line_format = workbook.add_format()
        top_line_format.set_top(2)

        right_line_format = workbook.add_format()
        right_line_format.set_right(2)

        left_line_format = workbook.add_format()
        left_line_format.set_left(2)

        bottom_line_format = workbook.add_format()
        bottom_line_format.set_bottom(2)

        side_line_format = workbook.add_format()
        side_line_format.set_left(2)
        side_line_format.set_right(2)

        top_bottom_line_format = workbook.add_format()
        top_bottom_line_format.set_top(2)
        top_bottom_line_format.set_bottom(2)

        bold_center_format = workbook.add_format({'top': 1, 'bottom': 1, 'left': 1, 'right': 1, 'bold': True})
        bold_center_format.set_align('center')
        bold_center_format.set_align('vcenter')

        bold_thick_format = workbook.add_format({'top': 2, 'bottom': 2, 'left': 1, 'right': 1, 'bold': True})
        bold_thick_format.set_align('center')
        bold_thick_format.set_align('vcenter')

        # End format declaration
        # add header (doing this manually to able to edit format of this row)
        for col_num, data in enumerate(title_names):
            worksheet.write(HEADER_OFFSET - 1, col_num, data)

        worksheet.write(HEADER_OFFSET + len(output_df) + CASH_LEVEL_OFFSET, 0, "Cash Level", default_cell_format)
        worksheet.write(HEADER_OFFSET + len(output_df) + CASH_LEVEL_OFFSET, 1, "${:,.2f}".format(cash_level),
                        default_cell_format)

        worksheet.write(HEADER_OFFSET + len(output_df) + CASH_LEVEL_OFFSET + 1, 0, "Cash Level %", default_cell_format)
        worksheet.write(HEADER_OFFSET + len(output_df) + CASH_LEVEL_OFFSET + 1, 1,
                        "{:,.2f}%".format(100.0 - total_alpha_sum),

                        default_cell_format)

        cash_top_line = xlsxwriter.utility.xl_range(HEADER_OFFSET + len(output_df) + CASH_LEVEL_OFFSET - 1, 0,
                                                    HEADER_OFFSET + len(output_df) + CASH_LEVEL_OFFSET - 1, 1)

        cash_bottom_line = xlsxwriter.utility.xl_range(HEADER_OFFSET + len(output_df) + CASH_LEVEL_OFFSET + 2, 0,
                                                       HEADER_OFFSET + len(output_df) + CASH_LEVEL_OFFSET + 2, 1)

        cash_right_line = xlsxwriter.utility.xl_range(HEADER_OFFSET + len(output_df) + CASH_LEVEL_OFFSET, 2,
                                                      HEADER_OFFSET + len(output_df) + CASH_LEVEL_OFFSET + 1, 2)

        # Add border lines for cash level
        worksheet.conditional_format(cash_top_line, {'type': 'no_errors', 'format': bottom_line_format})
        worksheet.conditional_format(cash_bottom_line, {'type': 'no_errors', 'format': top_line_format})
        worksheet.conditional_format(cash_right_line, {'type': 'no_errors', 'format': left_line_format})

        # Fill empty cells with a warning color
        worksheet.conditional_format(
            xlsxwriter.utility.xl_range(HEADER_OFFSET, 0, len(output_df) + HEADER_OFFSET - 1, COL_COUNT - 1),
            {'type': 'blanks', 'format': blank_format})

        # https://stackoverflow.com/questions/55928797/python-excelwriter-formatting-all-borders
        worksheet.conditional_format(
            xlsxwriter.utility.xl_range(HEADER_OFFSET, 0, len(output_df) + HEADER_OFFSET - 1, table_width - 1),
            {'type': 'no_errors', 'format': default_cell_format})

        fund_title = "Arbitrage"
        if FUND == "AED":
            fund_title = "Event driven"
        elif FUND == "TACO":
            fund_title = 'Credit'
        # Text format for Title and Subtitles
        worksheet.merge_range(
            xlsxwriter.utility.xl_range(WARNING_OFFSET, 0, WARNING_OFFSET + 1, COL_COUNT - SUBTITLE_OFFSET - 1),
            'Top ' + str(TOP_COUNT) + F' Positions in the {fund_title} Fund as of:'
            + selected_date.replace('-', '/'), title_format)
        worksheet.merge_range(
            xlsxwriter.utility.xl_range(WARNING_OFFSET, COL_COUNT - SUBTITLE_OFFSET, WARNING_OFFSET, COL_COUNT - 1),
            'Top ' + str(TOP_COUNT) + ' Total (%)', subtitle_format)
        worksheet.merge_range(
            xlsxwriter.utility.xl_range(WARNING_OFFSET + 1, COL_COUNT - SUBTITLE_OFFSET, WARNING_OFFSET + 1,
                                        COL_COUNT - 1),
            capital_sum, subtitle_format)

        round_to = 1  # number of decimals to round to
        # Hard coding to write the totals to the next row
        worksheet.write(HEADER_OFFSET + len(output_df), 26, str(round(itd_sum, round_to)))
        worksheet.write(HEADER_OFFSET + len(output_df), 27, str(round(ytd_sum, round_to)))
        worksheet.write(HEADER_OFFSET + len(output_df), 28, str(round(qtd_sum, round_to)))
        worksheet.write(HEADER_OFFSET + len(output_df), 29, str(round(mtd_sum, round_to)))
        worksheet.write(HEADER_OFFSET + len(output_df), 30, str(round(thirty_day_sum, round_to)))
        worksheet.write(HEADER_OFFSET + len(output_df), 31, str(round(five_day_sum, round_to)))
        worksheet.write(HEADER_OFFSET + len(output_df), 32, str(round(one_day_sum, round_to)))

        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 26, str(round(total_itd, round_to)))
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 27, str(round(total_ytd, round_to)))
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 28, str(round(total_qtd, round_to)))
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 29, str(round(total_mtd, round_to)))
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 30, str(round(total_thirty_day, round_to)))
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 31, str(round(total_five_day, round_to)))
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 32, str(round(total_one_day, round_to)))

        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 26, str(round(itd_sum / total_itd * 100, round_to)) + "%")
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 27, str(round(ytd_sum / total_ytd * 100, round_to)) + "%")
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 28, str(round(qtd_sum / total_qtd * 100, round_to)) + "%")
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 29, str(round(mtd_sum / total_mtd * 100, round_to)) + "%")
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 30,
                        str(round(thirty_day_sum / total_thirty_day * 100, round_to)) + "%")
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 31,
                        str(round(five_day_sum / total_five_day * 100, round_to)) + "%")
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 32,
                        str(round(one_day_sum / total_one_day * 100, round_to)) + "%")

        worksheet.write(HEADER_OFFSET + len(output_df), 25, f"Total (Top {TOP_COUNT})", rightallign_format)
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 25, "Total (Fund)", rightallign_format)
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 25, F"Top {TOP_COUNT} as a % of Fund", rightallign_format)

        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 26, str(round(total_itd, 2)))
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 27, str(round(total_ytd, 2)))
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 28, str(round(total_qtd, 2)))
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 29, str(round(total_mtd, 2)))
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 30, str(round(total_thirty_day, 2)))
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 31, str(round(total_five_day, 2)))
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 32, str(round(total_one_day, 2)))

        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 26, str(round(itd_sum / total_itd * 100, 2)) + "%")
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 27, str(round(ytd_sum / total_ytd * 100, 2)) + "%")
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 28, str(round(qtd_sum / total_qtd * 100, 2)) + "%")
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 29, str(round(mtd_sum / total_mtd * 100, 2)) + "%")
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 30,
                        str(round(thirty_day_sum / total_thirty_day * 100, 2)) + "%")
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 31,
                        str(round(five_day_sum / total_five_day * 100, 2)) + "%")
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 32, str(round(one_day_sum / total_one_day * 100, 2)) + "%")

        worksheet.write(HEADER_OFFSET + len(output_df), 25, "Total (Top 10)")
        worksheet.write(HEADER_OFFSET + len(output_df) + 1, 25, "Total (Fund)")
        worksheet.write(HEADER_OFFSET + len(output_df) + 2, 25, "Top 10 as a % of Fund")

        # Add warning header
        worksheet.merge_range(0, 0, WARNING_OFFSET - 1, WARNING_WIDTH - 1, WARNING_MESSAGE, warning_format)

        # Set width for each column
        worksheet.set_column('A:A', 8)
        worksheet.set_column('B:B', 20)
        worksheet.set_column('C:C', 20)
        worksheet.set_column('D:D', 16)
        worksheet.set_column('E:E', 26)
        worksheet.set_column('F:F', 22)
        worksheet.set_column('G:G', 50)
        worksheet.set_column('H:H', 16)
        worksheet.set_column('I:I', 12)
        worksheet.set_column('J:P', 16)
        worksheet.set_column('Q:Q', 10)
        worksheet.set_column('R:S', 15)
        worksheet.set_column('T:X', 20)
        worksheet.set_column('Y:Z', 20)

        # Center cells
        center_format = workbook.add_format()
        center_format.set_align('center')
        center_format.set_align('vcenter')
        worksheet.set_column('A:AG', None, center_format)

        # add thicc border
        header_line_format = workbook.add_format({'bold': True})
        header_format.set_align('center')
        header_format.set_align('vcenter')
        header_line_format.set_top(2)
        header_line_format.set_left(1)
        header_line_format.set_bottom(2)
        worksheet.conditional_format(
            xlsxwriter.utility.xl_range(len(output_df) + HEADER_OFFSET, 0, len(output_df) + HEADER_OFFSET,
                                        COL_COUNT - 1),
            {'type': 'no_errors', 'format': top_line_format})

        worksheet.conditional_format(xlsxwriter.utility.xl_range(WARNING_OFFSET, COL_COUNT,
                                                                 len(output_df) + HEADER_OFFSET - 1, COL_COUNT),
                                     {'type': 'no_errors', 'format': left_line_format})

        worksheet.conditional_format(
            xlsxwriter.utility.xl_range(HEADER_OFFSET - 1, 0, HEADER_OFFSET - 1, COL_COUNT - 1),
            {'type': 'no_errors', 'format': header_line_format})

        worksheet.set_tab_color("#00cc00")

        # END SUMMARY WORKSHEET, BEGING FILLING INDIVIDUAL WORKSHEETS

        # replace invalid cells with empty string
        detail_report_df = detail_report_df.fillna("")

        for i, tradegroup_data in detail_report_df.iterrows():
            target_ticker = tradegroup_data['target_ticker']
            tradegroup = tradegroup_data['tradegroup']
            rank = str(i + 1)
            sheet_name = '#{} {}'.format(rank, tradegroup)
            sheet_name = re.sub(r'[\[\]\:\*\?\/\\]', '-', sheet_name)  # remove special chars from name
            sheet_name = sheet_name[:31]  # worksheet names have a max of 31 char limit
            worksheet = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = worksheet
            if target_ticker:
                # Get notes from notes_notesmasterupdate_securty_pnl_daily
                notes_query = "SELECT date, title, author, article FROM notes_notesmaster WHERE tickers LIKE " \
                              "'%%{}%%' ORDER BY date DESC".format(target_ticker)
                notes_result = pd.read_sql_query(notes_query, con=con)

                # Get news from wic_news_newsmaster
                news_query = "SELECT date, title, source, url,tickers FROM wic_news_newsmaster WHERE tickers LIKE " \
                             "'%%{}%%' ORDER BY date DESC".format(target_ticker)
                news_result = pd.read_sql_query(news_query, con=con)

                # remove some news since the sql filter is not a strict match
                def filter_news(input_text):
                    if input_text.startswith(target_ticker) or ", " + target_ticker in input_text:
                        return True
                    else:
                        return False

                news_result = news_result[news_result['tickers'].apply(filter_news) == True]

                # set tab color
                entry = output_df.loc[output_df['TradeGroup'] == tradegroup]
                # entry.fillna("")
                if not entry['announced_year'].isnull().values.any():
                    entry_year = int(entry['announced_year'])
                    entry_month = int(entry['announced_month'])
                    if selected_date_obj.month - entry_month < 3 and selected_date_obj.year == entry_year:
                        worksheet.set_tab_color("#ff0000")

                # Output overview data
                worksheet.write(0, 0, '#{}'.format(rank), bold_center_format)
                worksheet.write(0, 1, 'Company', bold_center_format)
                worksheet.write(1, 0, 'Tradegroup', bold_center_format)
                worksheet.write(2, 0, 'Target', bold_center_format)
                worksheet.write(3, 0, 'Acquirer', bold_center_format)
                worksheet.write(1, 1, tradegroup)
                worksheet.write(2, 1, tradegroup_data['target_name'])
                worksheet.write(3, 1, tradegroup_data['acquirer_name'])

                # Add warning message to top of the summary page
                worksheet.merge_range(5, 0, 6, WARNING_WIDTH, WARNING_MESSAGE, warning_format)

                worksheet.merge_range('D1:D2', "Consideration", bold_center_format)
                worksheet.write(0, 4, 'Type', bold_center_format)
                worksheet.write(0, 5, 'Amount', bold_center_format)
                worksheet.write(1, 4, tradegroup_data['payment_type'])
                worksheet.write(1, 5, tradegroup_data['deal_value'])

                worksheet.write(0, 7, "Announcement Date", bold_center_format)
                worksheet.write(0, 8, 'Expected Completion Date', bold_center_format)
                worksheet.write(1, 7, tradegroup_data['announced_date'])
                worksheet.write(1, 8, tradegroup_data['expected_completion_date'])

                title_format = workbook.add_format({'font_size': '18', 'align': 'center', 'top': 2, 'bold': True})
                title_format.set_align('vcenter')

                worksheet.merge_range('A9:I9', "INTERNAL NOTES", bold_thick_format)
                worksheet.conditional_format('A1:B4', {'type': 'no_errors', 'format': default_cell_format})
                worksheet.conditional_format('D1:F2', {'type': 'no_errors', 'format': default_cell_format})
                worksheet.conditional_format('H1:I2', {'type': 'no_errors', 'format': default_cell_format})

                base_row = 11  # entry begining row
                row_height = 3  # height for each entry, default set as 3 rows
                total_row_height = 0

                # Add notes table title
                worksheet.write(9, 0, "Date", bold_thick_format)
                worksheet.write(9, 1, "Title", bold_thick_format)
                worksheet.write(9, 2, "Author", bold_thick_format)
                worksheet.merge_range('D10:I10', 'Note', bold_thick_format)

                #  output notes
                for j, notes_entry in notes_result.iterrows():
                    selected_format = None
                    selected_notes_format = None
                    if j % 2 == 1:  # alternate color based on row
                        selected_format = notes_other_odd_format
                        selected_notes_format = notes_odd_format
                    else:
                        selected_format = notes_other_even_format
                        selected_notes_format = notes_even_format

                    # parse notes from html to plain text
                    note = html2text.html2text(notes_entry['article'])

                    note_lines = note.count('\n')
                    row_height = min(note_lines, MAX_NOTE_HEIGHT)

                    # TODO: find a way to display images properly
                    # Remove image base64 from text
                    pattern = re.compile(r'!\[\]\(data:image\/png;base64,.*\)')
                    note = pattern.sub("[image]", note)

                    if notes_entry['date']:
                        worksheet.merge_range(
                            'A{}:A{}'.format(base_row + total_row_height,
                                             base_row + total_row_height + (row_height - 1)),
                            notes_entry['date'].strftime('%m/%d/%Y'), selected_format)
                    worksheet.merge_range(
                        'B{}:B{}'.format(base_row + total_row_height, base_row + total_row_height + (row_height - 1)),
                        notes_entry['title'], selected_format)
                    worksheet.merge_range(
                        'C{}:C{}'.format(base_row + total_row_height, base_row + total_row_height + (row_height - 1)),
                        notes_entry['author'], selected_format)

                    worksheet.merge_range(
                        'D{}:I{}'.format(base_row + total_row_height, base_row + total_row_height + (row_height - 1)),
                        note, selected_notes_format)

                    total_row_height += row_height

                # end printing notes, print news articles
                max_row = base_row + total_row_height
                news_base_row = max_row + 2

                # Add news table title
                worksheet.merge_range(max_row, 0, max_row, 8, "ARTICLES", bold_thick_format)
                worksheet.write(max_row + 1, 0, "Date", bold_thick_format)
                worksheet.merge_range(xlsxwriter.utility.xl_range(max_row + 1, 1, max_row + 1, 7), "Description",
                                      bold_thick_format)
                worksheet.write(max_row + 1, 8, "Link", bold_thick_format)

                # output news
                for j, news_entry in news_result.iterrows():
                    date_string = ""
                    if news_entry['date']:
                        date_string = news_entry['date'].strftime('%m/%d/%Y')
                    if j % 2 == 1:  # alternate color based on row
                        news_format = news_odd_format
                        notes_format = notes_other_odd_format
                    else:
                        news_format = news_even_format
                        notes_format = notes_other_even_format

                    worksheet.write(news_base_row + j, 0, date_string, notes_format)
                    worksheet.merge_range(xlsxwriter.utility.xl_range(news_base_row + j, 1, news_base_row + j, 7),
                                          news_entry['title'], notes_format)
                    if news_entry['url']:
                        worksheet.write_url(news_base_row + j, 8, news_entry['url'], string=news_entry['source'],
                                    cell_format=news_format)
                    else:
                        worksheet.write_url(news_base_row + j, 8, '', string=news_entry['source'],
                                            cell_format=news_format)


                # Set width for each column
                worksheet.set_column('A:A', 30)
                worksheet.set_column('B:B', 30)
                worksheet.set_column('C:C', 15)
                worksheet.set_column('D:G', 18)
                worksheet.set_column('H:I', 30)

                # Add borderlines
                worksheet.conditional_format('A5:B5', {'type': 'no_errors', 'format': top_line_format})
                worksheet.conditional_format('D3:F3', {'type': 'no_errors', 'format': top_line_format})
                worksheet.conditional_format('H3:I3', {'type': 'no_errors', 'format': top_line_format})
                worksheet.conditional_format('C1:C2', {'type': 'no_errors', 'format': side_line_format})
                worksheet.conditional_format('G1:G2', {'type': 'no_errors', 'format': side_line_format})
                worksheet.conditional_format('C3:C4', {'type': 'no_errors', 'format': left_line_format})
                worksheet.conditional_format('J1:J2', {'type': 'no_errors', 'format': left_line_format})

                eos_row = (max_row + 1) + len(news_result.index)  # the row number for end of sheet

                worksheet.conditional_format(xlsxwriter.utility.xl_range(8, 9, max_row - 2, 9),
                                             {'type': 'no_errors', 'format': left_line_format})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(max_row - 1, 0, max_row - 1, 8),
                                             {'type': 'no_errors', 'format': top_bottom_line_format})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(max_row, 9, eos_row, 9),
                                             {'type': 'no_errors', 'format': left_line_format})
                worksheet.conditional_format(xlsxwriter.utility.xl_range(eos_row + 1, 0, eos_row + 1, 8),
                                             {'type': 'no_errors', 'format': top_line_format})

                worksheet.set_column('A:I', None, center_format)
            else:  # if no detailed page found for the tradegroup
                worksheet.write(0, 0, 'Insufficient data for {}!'.format(tradegroup))

        # Close the Pandas Excel writer and setup file buffer to write.
        workbook.close()
        file_buffer.seek(0)
        return  file_buffer
    except Exception as e:
        error_msg = f'*FAILED: {fund_name} top {TOP_COUNT} report failed to generate.*\n Error: {e}\n{traceback.format_exc()}'
        error_msg = re.sub(r'[\"\']', "", error_msg)  # resolve django-slack converting quotes to html format
        if settings.DEBUG:
            print(error_msg)
        else:
            logger.error(error_msg)
            slack_message('generic.slack',
                          {'message': error_msg},
                          channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN)
        return None


@shared_task(bind=True)
def generate_top_ten(self, additional_recipients=None, ip=None):  # AEDNX ACFIX MF TOP TEN REPORT
    """
    Wrapper Method for generating top N deals for the fund ARB.
    @param additional_recipients: list of string of email addresses, all addresses must end with @wicfunds.com
    @param ip: ip of the source of request, used for logging
    """

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    top_count = 10
    fund_list = ['ARB', 'AED']  # 'TACO' still in development,
    file_buffers = []
    selected_date = datetime.datetime.now().date().strftime('%Y-%m-%d')
    # remove invalid emails if additional recipients are passed in
    if additional_recipients:
        additional_recipients = [x for x in additional_recipients if x.endswith('@wicfunds.com')]

    progress_recorder = ProgressRecorder(self)
    progress_recorder.set_progress(0, 100)

    for fund_name in fund_list:

        file_buffer = generate_sales_report(fund_name, top_count)
        file_name = f"Top {top_count} {fund_name} Deals (by Alpha Exp) - {selected_date}.xlsx"
        file_buffers.append((file_name,file_buffer))

    # Send out the file via email
    progress_recorder.set_progress(90, 100)

    recipients = ['risk@wicfunds.com', 'jnespoli@wicfunds.com', 'cfazioli@wicfunds.com',
                  'kfeeney@wicfunds.com', 'cwalker@wicfunds.com', 'Ecasadei@wicfunds.com',
                  'jhoerl@wicfunds.com', 'cbuntic@wicfunds.com', 'peisenmann@wicfunds.com']
    if additional_recipients:
        recipients = additional_recipients

    if settings.DEBUG:
        for fund_name, file_buffer in file_buffers:
            with open(f"{fund_name}.xlsx", 'wb') as out:  # Open temporary file as bytes
                out.write(file_buffer.read())
    else:  # only send email when it's not in debug to avoid spam
        send_email2(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD, recipients,
                    "Top " + str(top_count) + " reports as of " + selected_date, 'dispatch@wicfunds.com',
                    filestoAttach=file_buffers)
        slack_message('generic.slack',
                      {'message': f'*SUCCESS: Weekly top {top_count} reports as of {selected_date} created! *:tada:'},
                      channel=get_channel_name('portal-task-reports'), token=settings.SLACK_TOKEN)
        if ip:
            logger.info(f"Weekly top {top_count} as of {selected_date} was created by {ip} and sent to {recipients}")
        else:
            logger.info(f'Weekly top {top_count} as of {selected_date} processed and mailed out to {recipients}.')



