import datetime
import io
import json

import pandas as pd
import numpy as np
from celery import shared_task
from django.db import connection
from pandas.tseries.offsets import CustomBusinessDay
from sqlalchemy import create_engine
from pandas.tseries import offsets
from django.conf import settings

import holiday_utils
from email_utilities import send_email2
from exposures.models import DailyScenarios
from funds_snapshot.Spread import get_full_spread_df, Spread
from funds_snapshot.chart_utils import get_spread_metrics_charts
from funds_snapshot.models import TradeGroupPerformanceOverCapital
from holiday_utils import USNyseHolidayCalendar, get_last_trade_date
from pnl_attribution.tasks import get_pnl_attribution
from positions_and_pnl.models import TradeGroupMaster, SecurityPositionsAndPnl

def get_gross_spread(fund):
    """ generate gross spread data similar to cache_fund_spread_info"""
    ytd_date = (datetime.date.today() - offsets.YearBegin()).strftime('%Y-%m-%d')
    max_date = TradeGroupMaster.objects.latest('date').date
    date_1YR = max_date.replace(year=max_date.year - 1).strftime('%Y-%m-%d')
    max_date = max_date.strftime('%Y-%m-%d')
    # Run data on 1YR data and then take YTD data
    fund_spread_df_ytd = get_full_spread_df(start_date=date_1YR, end_date=max_date, fund_code=fund)
    fund_spread_df_ytd['Date'] = fund_spread_df_ytd['Date'].apply(pd.to_datetime)
    # skip holidays
    fund_spread_df_ytd = fund_spread_df_ytd[~fund_spread_df_ytd['Date'].isin(holiday_utils.holidays)]
    # Calculate Gross return instead of annualized return
    df = get_spread_metrics_charts(fund_spread_df_ytd, normalize=False, annualized=False)
    # take only YTD data
    df = df[df['Date'] >= ytd_date]
    df['Date'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    charts_json = df[['Date', 'CapitalWeightedReturn', 'ReturnWeightedDuration', 'Ann.Ret (CWR,CWD)',
                      'Ann.Ret (CWR,RWD)', 'Ann.Ret (Avg)']].to_json(orient='records')

    return json.dumps(charts_json)

def get_portfolio_insights():
    """ Currently only works for ARB and AED."""
    final_data_dictionary = {}

    funds_to_populate = ['AED', 'ARB', 'TACO']
    funds_str = '\'' + '\', \''.join(funds_to_populate) + '\''

    for fund in funds_to_populate:
        final_data_dictionary[fund] = {}
        final_data_dictionary[fund]['alpha_exposure_charts'] = {}
        final_data_dictionary[fund]['gainers'] = {}
        final_data_dictionary[fund]['losers'] = {}
        final_data_dictionary[fund]['largest_position_changes'] = {}

    today = datetime.datetime.now().date()
    calendar_rule = USNyseHolidayCalendar()
    US_BUSINESS_DAY = CustomBusinessDay(calendar=calendar_rule)
    last_trade_date = get_last_trade_date()
    as_of = last_trade_date.strftime('%Y-%m-%d')
    today_str = today.strftime('%Y-%m-%d')
    compared_with_date = (today - 5 * US_BUSINESS_DAY).strftime('%Y-%m-%d')  # Get last week trade day
    if (today - 5 * US_BUSINESS_DAY).year != today.year:
        exposure_query_date = (today - 5 * US_BUSINESS_DAY)
    else:
        exposure_query_date = datetime.datetime(today.year, 1, 1)

    # compared_with_date = (datetime.datetime.strptime(as_of, '%Y-%m-%d') - datetime.timedelta(days=7)).strftime(
    #     '%Y-%m-%d')
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD +
                           "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    exposure_query = f"select * from prod_wic_db.exposures_exposuressnapshot where fund in ({funds_str}) and " \
                     f"date>='{exposure_query_date.strftime('%Y-%m-%d')}'"  # remove asof for charts
    country_query = f"select distinct tradegroup, tradegroup_country from prod_wic_db.positions_and_pnl_tradegroupmaster " \
                    f"where fund in ({funds_str}) and date>={compared_with_date}"
    spread_info_query = f"select fund, spread_charts_json from prod_wic_db.funds_snapshot_fundsspreadinfo " \
                        f"where date = '{today_str}' and fund in ({funds_str})"
    gainers_losers_query = f"select tradegroup, fund, itd_bps, ytd_bps, mtd_bps, qtd_bps, days_30_bps " \
                           f"FROM  " \
                           f"prod_wic_db.funds_snapshot_tradegroupperformancefundnavbps where fund in ({funds_str}) " \
                           f" and date = '{as_of}' and status like 'ACTIVE'"
    nav_impacts_query = "select TradeGroup,RiskLimit, BASE_CASE_NAV_IMPACT_AED,OUTLIER_NAV_IMPACT_AED," \
                        "  BASE_CASE_NAV_IMPACT_ARB,OUTLIER_NAV_IMPACT_ARB" \
                        "    FROM" \
                        "         prod_wic_db.risk_reporting_dailynavimpacts"
    # Gather data into dataframes
    con = engine.connect()
    try:
        exposures_df = pd.read_sql_query(exposure_query, con=con)
        spreads_df = pd.read_sql_query(spread_info_query, con=con)
        country_df = pd.read_sql_query(country_query, con=con)
        gainers_losers_df = pd.read_sql_query(gainers_losers_query, con=con)
        nav_impacts_df = pd.read_sql_query(nav_impacts_query, con=con)
    except Exception as e:
        print(e)
    finally:
        con.close()

    exposures_df['date'] = exposures_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

    # Spread charts calculation
    for fund in ['AED', 'ARB']:
        final_data_dictionary[fund]['change_in_spreads'] = get_gross_spread(fund)

    # final_data_dictionary['AED']['change_in_spreads'] = spreads_df[spreads_df['fund'] == 'AED']

    # Fund level alpha exposure charts
    def get_exp_charts(df, fund, col, type_of_chart):
        # Adds either alpha exposure or spread info charts for each fund into the dictionary. Specififed by col
        _charts_df = df[df['fund'] == fund]
        final_data_dictionary[fund][type_of_chart] = _charts_df[['date', 'fund', col]].groupby(
            ['date', 'fund']).sum().reset_index()

    get_exp_charts(exposures_df, 'ARB', 'alpha_exposure', 'alpha_exposure_charts')
    get_exp_charts(exposures_df, 'AED', 'alpha_exposure', 'alpha_exposure_charts')
    get_exp_charts(exposures_df, 'TACO', 'alpha_exposure', 'alpha_exposure_charts')

    # Merge Dataframe with countries
    country_exposures = pd.merge(
        exposures_df[(exposures_df['date'] == as_of) & (exposures_df['catalyst_type'] != 'N/A')], country_df,
        how='left', on='tradegroup')
    country_exposures = country_exposures[
        ['fund', 'sleeve', 'tradegroup', 'alpha_exposure', 'tradegroup_country']].sort_values(by='alpha_exposure',
                                                                                              ascending=False)

    last_country_exposures = pd.merge(
        exposures_df[(exposures_df['date'] == compared_with_date) & (exposures_df['catalyst_type'] != 'N/A')],
        country_df, how='left',
        on='tradegroup')
    last_country_exposures = last_country_exposures[
        ['fund', 'sleeve', 'tradegroup', 'alpha_exposure', 'tradegroup_country']]

    last_country_exposures = last_country_exposures[
        ['fund', 'sleeve', 'tradegroup', 'alpha_exposure', 'tradegroup_country']]
    last_country_exposures = last_country_exposures.rename(columns={'alpha_exposure': 'last_alpha_exposure'})

    country_exposures = pd.merge(country_exposures, last_country_exposures, how='left',
                                 on=['tradegroup', 'fund', 'sleeve', 'tradegroup_country'])

    def add_top_10_exposures(exposures_df, fund, country):
        # Add ARB position size column in AED tab for Top10 holdings US and rest of world

        if country == None:
            # General get all
            final_data_dictionary[fund]['all_exposures'] = exposures_df[exposures_df['fund'] == fund].sort_values(
                by='alpha_exposure', ascending=False).head(10).to_json(orient='records')

        elif country == 'UNITED STATES':
            top_10_us_df = exposures_df[
                ((exposures_df['fund'] == fund) & (
                        exposures_df['tradegroup_country'].str.upper() == country))].sort_values(
                by='alpha_exposure', ascending=False).head(10)
            if fund == 'AED':
                arb_positions = exposures_df[
                    (exposures_df['fund'] == 'ARB') & (exposures_df['tradegroup_country'].str.upper() == country) &
                    exposures_df[
                        'tradegroup'].isin(top_10_us_df['tradegroup'])][
                    ['alpha_exposure', 'tradegroup']]
                arb_positions.rename(columns={"alpha_exposure": 'ARB Positions'}, inplace=True)
                top_10_us_df = top_10_us_df.merge(arb_positions, on='tradegroup', how='left')
            elif fund == 'ARB':
                aed_positions = exposures_df[
                    (exposures_df['fund'] == 'AED') & (exposures_df['tradegroup_country'].str.upper() == country) &
                    exposures_df['tradegroup'].isin(top_10_us_df['tradegroup'])][
                    ['alpha_exposure', 'tradegroup']]
                aed_positions.rename(columns={"alpha_exposure": 'AED Positions'}, inplace=True)
                top_10_us_df = top_10_us_df.merge(aed_positions, on='tradegroup', how='left')

            # Add NAV impact columns for table
            if fund != 'TACO':
                top_10_us_nav_df = get_nav_impact_by_tg(nav_impacts_df, top_10_us_df['tradegroup'], fund)
                top_10_us_df = top_10_us_df.merge(top_10_us_nav_df, how='left', left_on='tradegroup',
                                                  right_on='TradeGroup')
                top_10_us_df.drop(columns=['TradeGroup'], inplace=True)
            final_data_dictionary[fund]['usa_exposures'] = top_10_us_df.round(2).to_json(orient='records')

        else:
            # Outside US exposures
            top_10_rest_of_world_df = exposures_df[
                ((exposures_df['fund'] == fund) & (
                        exposures_df['tradegroup_country'].str.upper() != 'UNITED STATES'))].sort_values(
                by='alpha_exposure', ascending=False).head(10)
            if fund == 'AED':
                arb_positions = exposures_df[
                    (exposures_df['fund'] == 'ARB') & (
                            exposures_df['tradegroup_country'].str.upper() != 'UNITED STATES') &
                    exposures_df['tradegroup'].isin(top_10_rest_of_world_df['tradegroup'])][
                    ['alpha_exposure', 'tradegroup']]
                arb_positions.rename(columns={"alpha_exposure": 'ARB Positions'}, inplace=True)
                top_10_rest_of_world_df = top_10_rest_of_world_df.merge(arb_positions, on='tradegroup', how='left')
            elif fund == 'ARB':
                aed_positions = exposures_df[
                    (exposures_df['fund'] == 'AED') & (
                            exposures_df['tradegroup_country'].str.upper() != 'UNITED STATES') &
                    exposures_df['tradegroup'].isin(top_10_rest_of_world_df['tradegroup'])][
                    ['alpha_exposure', 'tradegroup']]
                aed_positions.rename(columns={"alpha_exposure": 'AED Positions'}, inplace=True)
                top_10_rest_of_world_df = top_10_rest_of_world_df.merge(aed_positions, on='tradegroup', how='left')

            # Add NAV impact columns for table
            if fund != 'TACO':
                top_10_rest_of_world_nav_df = get_nav_impact_by_tg(nav_impacts_df,
                                                                   top_10_rest_of_world_df['tradegroup'], fund)
                top_10_rest_of_world_df = top_10_rest_of_world_df.merge(top_10_rest_of_world_nav_df, how='left',
                                                                        left_on='tradegroup', right_on='TradeGroup')
                top_10_rest_of_world_df.drop(columns=['TradeGroup'], inplace=True)

            top_10_rest_of_world_df.dropna(subset=['alpha_exposure'], inplace=True)
            final_data_dictionary[fund]['restofworld_exposures'] = top_10_rest_of_world_df.round(2).to_json(
                orient='records')

            # top_10_rest_of_world_df['ARB position size'] = exposures_df[(exposures_df['fund'] == fund) & (exposures_df['tradegroup_country'] != 'UNITED STATES') & exposures_df['tradegroup'] == top_10_rest_of_world_df['tradegroup']]['alpha_expos']

    # Fill the data dictionary
    # add_top_10_exposures(country_exposures, 'ARB', None)
    # add_top_10_exposures(country_exposures, 'ARB', 'UNITED STATES')
    # add_top_10_exposures(country_exposures, 'ARB', 'REST')
    # AED

    for fund in funds_to_populate:
        add_top_10_exposures(country_exposures, fund, None)
        add_top_10_exposures(country_exposures, fund, 'REST')
        add_top_10_exposures(country_exposures, fund, 'UNITED STATES')

    # Biggest gainers and losers
    def get_top_5_gainers_losers(bps_df, fund):
        req_cols = {'days_30': '30D'}
        for col, modified_req_co in req_cols.items():
            bps_col = col + '_bps'
            final_data_dictionary[fund]['gainers'][bps_col] = bps_df[bps_df['fund'] == fund].nlargest(5, [bps_col])
            final_data_dictionary[fund]['losers'][bps_col] = bps_df[bps_df['fund'] == fund].nsmallest(5, [bps_col])

    for fund in funds_to_populate:
        get_top_5_gainers_losers(gainers_losers_df, fund)

    # Top 10 position changes
    def get_largest_10_changes_in_exposures(exp_df, fund):
        # Filter for fund
        fund_exp_df = exp_df[exp_df['fund'] == fund]
        exp_date_list = sorted(fund_exp_df['date'].unique())
        if len(exp_date_list) < 7:
            compare_date = exp_date_list[0]
        else:
            compare_date = exp_date_list[-7]
        max_date = fund_exp_df['date'].max()

        df1 = fund_exp_df[fund_exp_df['date'] == compare_date]
        df2 = fund_exp_df[fund_exp_df['date'] == max_date]

        cols = ['fund', 'tradegroup', 'sleeve', 'alpha_exposure']
        col_1_name = f'Last Week'
        col_2_name = f'Today'
        df1 = df1[cols].groupby(['fund', 'tradegroup', 'sleeve']).sum().reset_index().rename(
            columns={'alpha_exposure': col_1_name})
        df2 = df2[cols].groupby(['fund', 'tradegroup', 'sleeve']).sum().reset_index().rename(
            columns={'alpha_exposure': col_2_name})

        main = pd.merge(df1, df2, on=['fund', 'tradegroup', 'sleeve'])

        main['% change'] = main[col_2_name] - main[col_1_name]
        # Do both
        # Top 10 increases and top 10 decreases

        final_data_dictionary[fund]['largest_position_changes']['increases'] = main.sort_values('% change',
                                                                                                ascending=False).head(
            10).to_json(orient='records')
        final_data_dictionary[fund]['largest_position_changes']['decreases'] = main.sort_values('% change',
                                                                                                ascending=True).head(
            10).to_json(orient='records')

    for fund in funds_to_populate:
        get_largest_10_changes_in_exposures(exposures_df, fund)
        if fund != 'TACO':
            impacts_dict = get_top_10_nav_impacts(nav_impacts_df, fund)
            final_data_dictionary[fund]['top_10_nav_impacts'] = impacts_dict

    # Get Catalyst Attribution for 30 Days
    fund_list = funds_to_populate
    cuts_list = ['catalyst']
    period_list = ['YTD', 'QTD', 'MTD', '30D']
    end_date = datetime.datetime.strptime(as_of, '%Y-%m-%d')
    start_date = datetime.datetime(end_date.year, 1, 1)
    master_df = pd.DataFrame.from_records(TradeGroupMaster.objects.filter(date__gte=start_date, date__lte=end_date,
                                                                          fund__in=fund_list).values())
    del master_df['id']
    master_df.rename(columns={'date': 'Date', 'tradegroup_country': 'country', 'tradegroup_region': 'region',
                              'tradegroup_sector': 'sector', 'tradegroup_alphahedge': 'alphahedge',
                              'tradegroup_catalyst': 'catalyst'}, inplace=True)
    nav_df = pd.DataFrame.from_records(SecurityPositionsAndPnl.objects.filter(
        date_np__gte=start_date, date_np__lte=end_date, fund__in=fund_list).values(
        'date_np', 'fund', 'fund_nav').distinct())
    nav_df.rename(columns={'date_np': 'Date', 'fund_nav': 'nav'}, inplace=True)
    tasks_ids_dict = {}
    master_df['Date'] = master_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    nav_df['Date'] = nav_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    cash_list = ['SPOT', 'FORWARDS', 'CURRENCIES']
    master_df['catalyst'] = master_df['catalyst'].apply(lambda x: 'CASH' if x in cash_list else x)
    master_df['alphahedge'] = master_df['alphahedge'].apply(lambda x: 'CASH' if x in cash_list else x)

    for each_fund in fund_list:
        fund_master_df = master_df[master_df['fund'] == each_fund]
        fund_nav_df = nav_df[nav_df['fund'] == each_fund].drop_duplicates()
        data = get_pnl_attribution(fund_list, fund_master_df, fund_nav_df, each_fund, period_list,
                                   cuts_list, start_date, end_date, False, False, 'None')

        final_data_dictionary[each_fund]['catalyst_attribution'] = data['catalyst'][
            ['catalyst', 'Contribution to NAV(bps)', 'period']]

    # Add additional tables for TACO
    # Add scenario for TACO
    as_of_date = DailyScenarios.objects.latest('timestamp').timestamp
    scenario_obj = DailyScenarios.objects.get(timestamp=as_of_date)
    scenario_json = json.loads(scenario_obj.scenario_df)
    taco_scenario_dict = {}
    for row in scenario_json:
        fund = row['fund']
        if fund == 'TACO':
            taco_scenario_dict.setdefault(fund, [])
            taco_scenario_dict[fund].append(row)
    df_data = []
    for fund in taco_scenario_dict:
        df_data += taco_scenario_dict[fund]
    taco_scenario_df = pd.DataFrame(data=df_data)
    taco_scenario_df = taco_scenario_df.sort_values(['CR01(bps)', 'DV01(bps)'], ascending=(False, False)).head(10)

    taco_scenario_df = taco_scenario_df[['tradegroup', 'sleeve', 'CR01($)', 'CR01(bps)', 'DV01($)', 'DV01(bps)']]
    taco_scenario_df = taco_scenario_df.rename(columns={'tradegroup': 'TradeGroup', 'sleeve': 'Sleeve'})
    final_data_dictionary['TACO']['scenario'] = taco_scenario_df

    # add attribution_over_own_capital

    capital_query = 'SELECT TradeGroup, SUM(DeltaAdjGross) FROM wic.daily_flat_file_db WHERE Fund = "AED" ' \
                    f'AND Flat_file_as_of = "{as_of}" AND AlphaHedge = "Alpha" GROUP BY TradeGroup'

    pnl_query = f'SELECT tradegroup, SUM(pnl) AS pnl FROM prod_wic_db.positions_and_pnl_tradegroupmaster ' \
                f'WHERE fund = "AED" AND date >= "2021-01-01" AND status = "active" group by tradegroup'

    pnl_sub_query = 'SELECT tradegroup, tradegroup_country , capital_dollar ' \
                    'FROM prod_wic_db.positions_and_pnl_tradegroupmaster ' \
                    f'WHERE fund = "AED" AND status = "active" AND date = "{as_of}"'


    pnl_df = pd.read_sql_query(pnl_query, con=connection)
    pnl_sub_df = pd.read_sql_query(pnl_sub_query, con=connection)
    pnl_df = pnl_df.merge(pnl_sub_df, on='tradegroup')

    capital_df = pd.read_sql_query(capital_query, con=connection)

    result_df = pnl_df.merge(capital_df, left_on='tradegroup', right_on='TradeGroup')
    result_df.drop('TradeGroup', axis=1, inplace=True)

    def calculate_roc(row):
        """
        Helper method to calculate the return on capital
        For US:
        For Other countries:
        """
        denominator = 0
        if row['tradegroup_country'] == 'UNITED STATES':
            denominator = row['capital_dollar']
        else:
            denominator = row['SUM(DeltaAdjGross)']

        if denominator == 0:
            return 0
        return row['pnl'] * 10000.0 / denominator

    result_df['ROC'] = result_df.apply(calculate_roc, axis=1)
    aed_exposure = exposures_df[(exposures_df['fund'] == 'AED') & ~(
            exposures_df['alpha_exposure'].isnull() | (exposures_df['alpha_exposure'] == 0)) & (
                                        exposures_df['date'] == as_of)]

    result_df = result_df.merge(aed_exposure, on='tradegroup')
    result_df = result_df[['tradegroup', 'sleeve', 'alpha_exposure', 'ROC', 'tradegroup_country']]

    result_df = result_df.rename(
        columns={'tradegroup': 'TradeGroup', 'sleeve': 'Sleeve', 'alpha_exposure': 'Alpha Exposure',
                 'tradegroup_country': 'Country'})

    result_df['Country'] = result_df['Country'].replace('UNITED STATES', 'US').replace('UNITED KINGDOM', 'UK')
    top_10_roc = result_df.sort_values(by='ROC', ascending=False).head(10)
    bottom_10_roc = result_df.sort_values(by='ROC', ascending=True).head(10)
    final_data_dictionary['AED']['top_10_roc'] = top_10_roc
    final_data_dictionary['AED']['bottom_10_roc'] = bottom_10_roc

    return final_data_dictionary


def get_nav_impact_by_tg(impacts_df, tg_list, fund):
    other_fund = 'ARB' if fund == 'AED' else 'AED'
    col_prefixes = ['BASE_CASE_NAV_IMPACT_', 'OUTLIER_NAV_IMPACT_']
    req_cols = ['TradeGroup', 'RiskLimit', f'BASE_CASE_NAV_IMPACT_{other_fund}']

    for col_prefix in col_prefixes:
        req_col = col_prefix + fund
        req_cols.append(req_col)
        impacts_df[req_col] = impacts_df[req_col].fillna(0).replace("", 0).astype(float).round(3)

    impacts_df = impacts_df[req_cols].reset_index(drop=True)
    result_df = pd.DataFrame(columns=['TradeGroup', 'RiskLimit'])
    temp_req_cols = ['TradeGroup', 'RiskLimit', 'BASE_CASE_NAV_IMPACT_' + fund, 'OUTLIER_NAV_IMPACT_' + fund,
                     f'BASE_CASE_NAV_IMPACT_{other_fund}']

    temp_df = impacts_df[impacts_df['TradeGroup'].isin(tg_list)][temp_req_cols]
    result_df = pd.merge(result_df, temp_df, on=['TradeGroup', 'RiskLimit'], how='outer')
    result_df = result_df[req_cols].fillna(0)
    try:
        for index, row in result_df.iterrows():
            base = 'BASE_CASE_NAV_IMPACT_' + fund
            out = 'OUTLIER_NAV_IMPACT_' + fund
            if row[base] == 0:
                b_value = impacts_df[impacts_df['TradeGroup'] == row['TradeGroup']].iloc[0][base]
                result_df.loc[result_df['TradeGroup'] == row['TradeGroup'], base] = b_value

            if row[out] == 0:
                out_value = impacts_df[impacts_df['TradeGroup'] == row['TradeGroup']].iloc[0][out]
                result_df.loc[result_df['TradeGroup'] == row['TradeGroup'], out] = out_value
    except Exception as e:
        print(e)
    return result_df


def get_top_10_nav_impacts(impacts_df, fund):
    col_prefixes = ['BASE_CASE_NAV_IMPACT_', 'OUTLIER_NAV_IMPACT_']
    req_cols = ['TradeGroup', 'RiskLimit']
    for col_prefix in col_prefixes:
        req_col = col_prefix + fund
        req_cols.append(req_col)
        impacts_df[req_col] = impacts_df[req_col].fillna(0).replace("", 0).astype(float).round(3)

    impacts_df = impacts_df[req_cols].reset_index(drop=True)
    result_df = pd.DataFrame(columns=['TradeGroup', 'RiskLimit'])
    temp_req_cols = ['TradeGroup', 'RiskLimit', 'BASE_CASE_NAV_IMPACT_' + fund, 'OUTLIER_NAV_IMPACT_' + fund]
    temp_df = impacts_df.nsmallest(10, 'BASE_CASE_NAV_IMPACT_' + fund)[temp_req_cols]
    result_df = pd.merge(result_df, temp_df, on=['TradeGroup', 'RiskLimit'], how='outer')
    result_df = result_df[req_cols].fillna(0)
    try:
        for index, row in result_df.iterrows():
            base = 'BASE_CASE_NAV_IMPACT_' + fund
            out = 'OUTLIER_NAV_IMPACT_' + fund
            if row[base] == 0:
                b_value = impacts_df[impacts_df['TradeGroup'] == row['TradeGroup']].iloc[0][base]
                result_df.loc[result_df['TradeGroup'] == row['TradeGroup'], base] = b_value

            if row[out] == 0:
                out_value = impacts_df[impacts_df['TradeGroup'] == row['TradeGroup']].iloc[0][out]
                result_df.loc[result_df['TradeGroup'] == row['TradeGroup'], out] = out_value
    except Exception as e:
        print(e)
    return result_df.to_json(orient='records')


def write_talbe_on_sheet(workbook, sheet_name, start_pos, data: pd.DataFrame, title, color_cols=[]):
    """
    Helper method to write dataframe to excel with customizable options
    merge two columns for Tradgroup and Sleeve
    Parameters
    ----------
    workbook : workbook to write the table to
    sheet_name : str of sheet name to write on in the excel writer
    start_pos : tuple of ints indicating row and column for top left point to write the table
    data :dataframe that contains the info needed for the chart
    title: Tile of the table
    Returns
    -------
    row: int of bottom most row that is used by the table
    col: int of right most column that is used by the table

    """
    x, y = start_pos
    dx = dy = 0
    data.fillna('', inplace=True)
    data = data.round(2)
    table_width = len(data.columns)

    # adjust margin for special cases
    if "TradeGroup" in data.columns:
        table_width += 1
    if "Sleeve" in data.columns:
        table_width += 1

    sheet = workbook.get_worksheet_by_name(sheet_name)
    # define all the table formats
    title_format = workbook.add_format({'bold': True, 'top': 1, 'bottom': 1, 'left': 1, 'right': 1, 'align': 'center'})
    header_format = workbook.add_format({'top': 1, 'bottom': 1, 'left': 1, 'right': 1, 'font_size': 10})
    tg_format = workbook.add_format(
        {'font_color': 'blue', 'top': 1, 'bottom': 1, 'left': 1, 'right': 1, 'font_size': 9})

    data_format = workbook.add_format({'top': 1, 'bottom': 1, 'left': 1, 'right': 1, 'font_size': 9, 'align': 'left'})
    red_format = workbook.add_format(
        {'top': 1, 'bottom': 1, 'left': 1, 'right': 1, 'font_size': 9, 'font_color': '#FF0000', 'align': 'left',
         'bg_color': '#F2DCDB'})
    green_format = workbook.add_format(
        {'top': 1, 'bottom': 1, 'left': 1, 'right': 1, 'font_size': 9, 'font_color': '#38703b', 'align': 'left',
         'bg_color': '#EBF1DE'})

    # write title
    sheet.merge_range(y, x, y, x + table_width - 1, title, title_format)
    #     write columns
    y += 1

    # write column names
    for col in data.columns:
        if col == 'TradeGroup' or col == 'Sleeve':
            sheet.merge_range(y, x + dx, y, x + dx + 1, col, header_format)
            dx += 2
        else:
            sheet.write(y, x + dx, col, header_format)
            dx += 1
    y += 1
    dx = 0

    for index, row in data.iterrows():  # index
        for col in data.columns:
            entry = row[col]
            if not entry:
                entry = ''
            if col == 'TradeGroup':
                tradegroup = entry
                tg_url = "http://192.168.0.16:8000/position_stats/get_tradegroup_story?TradeGroup=" \
                         f"{tradegroup.replace(' ', '+')}&Fund={sheet_name}"
                # https://stackoverflow.com/questions/55223187/xlsxwirter-merge-range-with-url
                sheet.merge_range(y, x + dx, y, x + dx + 1, row[col], data_format)
                sheet.write(y, x + dx, tradegroup, tg_format)  # changed from write_url
                dx += 2
            elif col == 'Sleeve':
                # if row[col]
                sheet.merge_range(y, x + dx, y, x + dx + 1, row[col], data_format)
                dx += 2
            elif col in color_cols:
                if not entry:
                    sheet.write(y, x + dx, entry, data_format)
                elif float(entry) > 0:
                    sheet.write(y, x + dx, entry, green_format)
                elif float(entry) < 0:
                    sheet.write(y, x + dx, entry, red_format)
                else:
                    sheet.write(y, x + dx, entry, data_format)
                dx += 1
            else:
                sheet.write(y, x + dx, entry, data_format)
                dx += 1
        dx = 0
        y += 1

    return (y - 1, x + table_width - 1)
    # if data


@shared_task
def generate_portfolio_report():
    # collect data to generate the report
    source_df = get_portfolio_insights()

    file_buffer = io.BytesIO()
    excel_writer = pd.ExcelWriter(file_buffer, engine='xlsxwriter',
                                  options={'strings_to_numbers': True, 'in_memory': True})

    workbook = excel_writer.book

    page_title_format = workbook.add_format({'bold': True, 'align': 'center', 'font_size': 18})
    page_warning_format = workbook.add_format({'bold': True, 'align': 'center', 'font_size': 12, 'bg_color': '#FFFF00'})
    as_of_format = workbook.add_format({'align': 'right'})

    last_trade_date = get_last_trade_date()  # Get last trade day
    for fund in source_df.keys():

        if fund == 'TACO':
            workbook.add_worksheet('WICO')
            sheet = workbook.get_worksheet_by_name('WICO')
            sheet_name = 'WICO'
        else:
            workbook.add_worksheet(fund)
            sheet = workbook.get_worksheet_by_name(fund)
            sheet_name = fund

        # Add title
        sheet.merge_range(0, 0, 1, 14, sheet_name + ' Portfolio Report', page_title_format)
        # Add Warning
        sheet.merge_range(2, 0, 2, 14, ' Internal Use Only', page_warning_format)
        # Add as_of date
        sheet.merge_range(3, 13, 3, 14, 'As of ' + last_trade_date.strftime('%m/%d/%Y'), as_of_format)

        # TOP 10 ALL EXPOSURE
        top_10_all_exposure_df = pd.DataFrame(json.loads(source_df[fund]['all_exposures']))
        top_10_all_exposure_df = top_10_all_exposure_df.rename(
            columns={f'alpha_exposure': 'Alpha Expo', f'sleeve': 'Sleeve',
                     'tradegroup': 'TradeGroup', 'tradegroup_country': 'Country'})
        top_10_all_exposure_df['Country'] = top_10_all_exposure_df['Country'].replace('UNITED STATES', 'US').replace(
            'UNITED KINGDOM', 'UK')
        top_10_all_exposure_df = top_10_all_exposure_df[['TradeGroup', 'Sleeve', 'Alpha Expo', 'Country']]
        write_talbe_on_sheet(workbook, sheet_name, (0, 5), top_10_all_exposure_df, "Top 10 Holdings")

        sum_format = workbook.add_format({'font_size': 9, 'align': 'left'})

        other_fund = 'AED' if fund == 'ARB' else 'ARB'
        # TOP 10 US
        top_10_us_exposure_df = pd.DataFrame(json.loads(source_df[fund]['usa_exposures']))

        if fund == 'TACO':
            top_10_us_exposure_df = top_10_us_exposure_df.rename(
                columns={f'alpha_exposure': 'Alpha Expo', f'sleeve': 'Sleeve', 'tradegroup': 'TradeGroup',
                         'last_alpha_exposure': 'Last Week'})
            top_10_us_exposure_df = top_10_us_exposure_df[['TradeGroup', 'Sleeve', 'Alpha Expo', 'Last Week']]

        else:
            top_10_us_exposure_df = top_10_us_exposure_df.rename(
                columns={f'alpha_exposure': 'Alpha Expo', f'sleeve': 'Sleeve', 'tradegroup': 'TradeGroup',
                         'last_alpha_exposure': 'Last Week', f'BASE_CASE_NAV_IMPACT_{fund}': f'{fund} Base case',
                         f'OUTLIER_NAV_IMPACT_{fund}': f'{fund} Outlier', 'RiskLimit': 'Risk Limit',
                         f'BASE_CASE_NAV_IMPACT_{other_fund}': f'{other_fund} Base case'})

            top_10_us_exposure_df = top_10_us_exposure_df[
                ['TradeGroup', 'Sleeve', 'Alpha Expo', 'Last Week', f'{other_fund} Positions', f'{fund} Base case',
                 f'{other_fund} Base case', f'{fund} Outlier', 'Risk Limit']]

        end_row, end_col = write_talbe_on_sheet(workbook, sheet_name, (0, 19), top_10_us_exposure_df,
                                                "Top 10 Holdings US")
        # Add sum below
        sheet.write(end_row + 1, end_col - 7, "Total:", sum_format)
        sheet.write(end_row + 1, end_col - 6, str(round(top_10_us_exposure_df['Alpha Expo'].replace('', 0).sum(), 2)),
                    sum_format)
        sheet.write(end_row + 1, end_col - 5, str(round(top_10_us_exposure_df['Last Week'].replace('', 0).sum(), 2)),
                    sum_format)
        if fund != 'TACO':
            sheet.write(end_row + 1, end_col - 4, str(
                round(top_10_us_exposure_df[f'{other_fund} Positions'].replace('', 0).sum(), 2)), sum_format)
        # TOP 10 REST OF WORLD
        top_10_rest_exposure_df = pd.DataFrame(json.loads(source_df[fund]['restofworld_exposures']))
        top_10_rest_exposure_df['tradegroup_country'] = top_10_rest_exposure_df['tradegroup_country'].replace(
            'UNITED STATES', 'US').replace('UNITED KINGDOM', 'UK')

        if fund == 'TACO':
            top_10_rest_exposure_df = top_10_rest_exposure_df.rename(
                columns={f'alpha_exposure': 'Alpha Expo', f'sleeve': 'Sleeve',
                         'tradegroup': 'TradeGroup', 'last_alpha_exposure': 'Last Week',
                         'tradegroup_country': 'Country'})
            top_10_rest_exposure_df = top_10_rest_exposure_df[
                ['TradeGroup', 'Sleeve', 'Alpha Expo', 'Last Week', 'Country']]
        else:
            top_10_rest_exposure_df = top_10_rest_exposure_df.rename(
                columns={f'alpha_exposure': 'Alpha Expo', f'sleeve': 'Sleeve',
                         'tradegroup': 'TradeGroup', 'last_alpha_exposure': 'Last Week',
                         'tradegroup_country': 'Country',
                         f'BASE_CASE_NAV_IMPACT_{fund}': f'{fund} Base case',
                         f'OUTLIER_NAV_IMPACT_{fund}': f'{fund} Outlier',
                         'RiskLimit': 'Risk Limit', f'BASE_CASE_NAV_IMPACT_{other_fund}': f'{other_fund} Base case'})

            top_10_rest_exposure_df = top_10_rest_exposure_df[
                ['TradeGroup', 'Sleeve', 'Alpha Expo', 'Last Week', f'{other_fund} Positions', f'{fund} Base case',
                 f'{other_fund} Base case', f'{fund} Outlier', 'Risk Limit', 'Country']]

        end_row, end_col = write_talbe_on_sheet(workbook, sheet_name, (0, 33), top_10_rest_exposure_df,
                                                "Top 10 Holdings Rest of the world")
        # Add sum below
        sheet.write(end_row + 1, end_col - 8, "Total:", sum_format)
        sheet.write(end_row + 1, end_col - 7, str(round(top_10_rest_exposure_df['Alpha Expo'].replace('', 0).sum(), 2)),
                    sum_format)
        sheet.write(end_row + 1, end_col - 6, str(round(top_10_rest_exposure_df['Last Week'].replace('', 0).sum(), 2)),
                    sum_format)
        if fund != 'TACO':
            sheet.write(end_row + 1, end_col - 5, str(
                round(top_10_rest_exposure_df[f'{other_fund} Positions'].replace('', 0).sum(), 2)), sum_format)

        # NAV TABLE
        if fund != 'TACO':
            top_10_nav_data_df = pd.DataFrame(json.loads(source_df[fund]['top_10_nav_impacts']))
            top_10_nav_data_df = top_10_nav_data_df.rename(
                columns={f'BASE_CASE_NAV_IMPACT_{fund}': 'Base case', f'OUTLIER_NAV_IMPACT_{fund}': 'Outlier',
                         'RiskLimit': 'Risk Limit'})
            top_10_nav_data_df = top_10_nav_data_df[['TradeGroup', 'Base case', 'Outlier', 'Risk Limit']]
            write_talbe_on_sheet(workbook, sheet_name, (7, 5), top_10_nav_data_df, "Top 10 NAV Impact")

        # TOP ALPHA INCREASE
        top_10_alpha_increase_df = pd.DataFrame(json.loads(source_df[fund]['largest_position_changes']['increases']))
        top_10_alpha_increase_df = top_10_alpha_increase_df.rename(
            columns={'tradegroup': 'TradeGroup', 'sleeve': 'Sleeve', '% change': '% Change'})
        top_10_alpha_increase_df = top_10_alpha_increase_df[['TradeGroup', 'Sleeve', 'Last Week', 'Today', '% Change']]
        write_talbe_on_sheet(workbook, sheet_name, (0, 64), top_10_alpha_increase_df, "Top Alpha Exposure Increase")

        # TOP ALPHA DECREASE
        top_10_alpha_decrease_df = pd.DataFrame(json.loads(source_df[fund]['largest_position_changes']['decreases']))
        top_10_alpha_decrease_df = top_10_alpha_decrease_df.rename(
            columns={'tradegroup': 'TradeGroup', 'sleeve': 'Sleeve', '% change': '% Change'})
        top_10_alpha_decrease_df = top_10_alpha_decrease_df[['TradeGroup', 'Sleeve', 'Last Week', 'Today', '% Change']]
        write_talbe_on_sheet(workbook, sheet_name, (8, 64), top_10_alpha_decrease_df, "Top Alpha Exposure Decrease")

        # TOP GAINER 30 DAY
        top_10_gainer_df = source_df[fund]['gainers']['days_30_bps']

        top_10_gainer_df = top_10_gainer_df.rename(
            columns={'tradegroup': 'TradeGroup', 'itd_bps': 'ITD BPS', 'ytd_bps': 'YTD BPS', 'mtd_bps': 'MTD BPS',
                     'qtd_bps': 'QTD BPS', 'days_30_bps': '30 Day BPS'})
        top_10_gainer_df = top_10_gainer_df[['TradeGroup', 'ITD BPS', 'YTD BPS', 'MTD BPS', 'QTD BPS', '30 Day BPS']]
        write_talbe_on_sheet(workbook, sheet_name, (0, 78), top_10_gainer_df, "Top Gainers 30Days",
                             color_cols=['ITD BPS', 'YTD BPS', 'MTD BPS', 'QTD BPS', '30 Day BPS'])

        # TOP LOSER 30 DAY
        top_10_loser_df = source_df[fund]['losers']['days_30_bps']
        top_10_loser_df = top_10_loser_df.rename(
            columns={'tradegroup': 'TradeGroup', 'itd_bps': 'ITD BPS', 'ytd_bps': 'YTD BPS', 'mtd_bps': 'MTD BPS',
                     'qtd_bps': 'QTD BPS', 'days_30_bps': '30 Day BPS'})
        top_10_loser_df = top_10_loser_df[['TradeGroup', 'ITD BPS', 'YTD BPS', 'MTD BPS', 'QTD BPS', '30 Day BPS']]
        write_talbe_on_sheet(workbook, sheet_name, (8, 78), top_10_loser_df, "Top Losers 30Days",
                             color_cols=['ITD BPS', 'YTD BPS', 'MTD BPS', 'QTD BPS', '30 Day BPS'])

        # CATALYST ATTRIBUTION
        catalyst_df = source_df[fund]['catalyst_attribution']
        catalyst_df.rename_axis(None, axis=1, inplace=True)
        catalyst_df = catalyst_df.pivot(index='catalyst', columns='period', values='Contribution to NAV(bps)')
        catalyst_df.reset_index(inplace=True)
        catalyst_df = catalyst_df.rename({'catalyst': 'Catalyst'})
        end_row, end_col = write_talbe_on_sheet(workbook, sheet_name, (2, 87), catalyst_df, 'Catalyst Attribution')
        sheet.write("C80", "Total:", sum_format)
        sheet.write(end_row + 1, end_col - 2, str(round(catalyst_df['MTD'].replace('', 0).sum(), 2)), sum_format)
        sheet.write(end_row + 1, end_col - 1, str(round(catalyst_df['QTD'].replace('', 0).sum(), 2)), sum_format)
        sheet.write(end_row + 1, end_col - 0, str(round(catalyst_df['YTD'].replace('', 0).sum(), 2)), sum_format)

        # ROC FOR AED
        if fund == 'AED':
            top_10_roc_df = source_df[fund]['top_10_roc']
            bottom_10_roc_df = source_df[fund]['bottom_10_roc']
            write_talbe_on_sheet(workbook, sheet_name, (0, 94), top_10_roc_df, 'Top 10 Return on Capital')
            write_talbe_on_sheet(workbook, sheet_name, (8, 94), bottom_10_roc_df, 'Bottom 10 Return on Capital')

    # Inserting charts
    for fund in source_df.keys():
        worksheet = workbook.add_worksheet(fund + "_data")
        excel_writer.sheets[fund + "_data"] = worksheet

        # Adding Spread chart
        if fund != 'TACO':
            spread_df = pd.DataFrame(
                json.loads(json.loads(source_df[fund]['change_in_spreads'])))
            spread_df.to_excel(excel_writer, sheet_name=fund + '_data', startrow=0, startcol=0, index=False)
            spread_chart = workbook.add_chart({'type': 'line'})
            spread_chart.add_series({
                'categories': [fund + '_data', 1,0, len(spread_df.index), 0],
                'values': [fund + "_data", 1, 4, len(spread_df.index), 4],
            })
            spread_chart.set_title(
                {'name': f'{fund} Gross Spread (%, CWR,RWD)', 'name_font': {'name': 'Calibri', 'size': 12}})
            spread_chart.set_legend({'position': 'none'})

            max_spread_val = spread_df['Ann.Ret (CWR,RWD)'].max()
            min_spread_val = spread_df['Ann.Ret (CWR,RWD)'].min()
            spread_chart.set_y_axis({'min': round(min_spread_val) - 1, 'max': round(max_spread_val) + 1})

            # spread_chart.set_x_axis({'line': {'none': True}})

        # Adding Alpha exposure chart
        exposure_df = source_df[fund]['alpha_exposure_charts']
        exposure_df.to_excel(excel_writer, sheet_name=fund + '_data', startrow=0, startcol=6, index=False)
        exposure_chart = workbook.add_chart({'type': 'line'})
        exposure_chart.add_series({
            'categories': [fund + '_data', 1, 6, len(exposure_df), 6],
            'values': [fund + "_data", 1, 8, len(exposure_df), 8],
        })
        if fund == 'TACO':
            exposure_chart.set_title(
                {'name': f'WICO Alpha Exposure (%)', 'name_font': {'name': 'Calibri', 'size': 12}})
        else:
            exposure_chart.set_title(
                {'name': f'{fund} Alpha Exposure (%)', 'name_font': {'name': 'Calibri', 'size': 12}})
        max_exposure_val = exposure_df['alpha_exposure'].max()
        min_exposure_val = exposure_df['alpha_exposure'].min()
        exposure_chart.set_legend({'position': 'none'})
        exposure_chart.set_y_axis({'min': (min_exposure_val // 10 - 2) * 10, 'max': (max_exposure_val // 10 + 2) * 10})

        if fund == 'TACO':
            worksheet = workbook.get_worksheet_by_name('WICO')
            worksheet.insert_chart('A48', exposure_chart)

            # TOP 10 CR01 DV01 chart
            write_talbe_on_sheet(workbook, sheet_name, (0,96), source_df['TACO']['scenario'], 'Top 10 CR01 & DV01')


        else:
            worksheet = workbook.get_worksheet_by_name(fund)
            worksheet.insert_chart('A48', spread_chart)
            worksheet.insert_chart('I48', exposure_chart)

    excel_writer.save()
    file_name = f'Portfolio Report As of {last_trade_date.strftime("%m-%d-%Y")}.xlsx'

    if settings.DEBUG:
        with open(file_name, 'wb') as out_file:
            file_buffer.seek(0)
            out_file.write(file_buffer.getvalue())
            out_file.close()
    else:
        file_buffer.seek(0)
        recipients = ['nmiraj@wicfunds.com', 'ssuizhu@wicfunds.com', 'akubal@wicfunds.com']
        send_email2(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD, recipients,
                    "Portfolio report as of " + last_trade_date.strftime("%m-%d-%Y"), 'dispatch@wicfunds.com',
                    filestoAttach=[(file_name, file_buffer)])
