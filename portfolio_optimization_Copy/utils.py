from collections import OrderedDict
import datetime
import numpy as np
import pandas as pd
import warnings
import itertools
import bbgclient
from django.conf import settings
from sqlalchemy import create_engine
from django.db import connection
from loss_budgeting.models import LossBudgets
from portfolio_optimization.models import (PnlPotentialDate, PnlPotentialExclusions, PnlPotentialIncremental,
    PnlPotentialScenarios, PnlPotentialESSConstituents, ArbCreditPnLPotentialDrilldown, PnlPotentialOtherValues)
from realtime_pnl_impacts.models import PnlMonitors


def parse_fld(id2fld2val,fld, id):
    try:
        return float(id2fld2val[id][fld][0])
    except:
        return None


def get_pnl_potential():
    aed_df = pd.DataFrame.from_records(ArbCreditPnLPotentialDrilldown.objects.all().values())
    implied_prob_deduct = PnlPotentialOtherValues.objects.filter(field_name='implied_prob').first()
    implied_prob_deduct = implied_prob_deduct.field_value if implied_prob_deduct else 5
    aed_df['custom_implied_probability'] = aed_df['implied_probability'] - implied_prob_deduct
    aed_df.at[aed_df['implied_probability'] == 0, 'custom_implied_probability'] = 0
    aed_df['max_pnl_potential'] = aed_df['current_mkt_val_pct'] * aed_df['aed_nav'] * aed_df['gross_ror'] * 0.0001
    # pnl potential dates
    pnl_potential_dates = pd.DataFrame.from_records(PnlPotentialDate.objects.all().values())
    pnl_potential_dates.rename(columns={'start_date': 'Start Date', 'end_date': 'End Date', 'date_name': 'Name',
                                        'sleeve': 'Sleeve'}, inplace=True)

    pnl_potential_exclusions = pd.DataFrame.from_records(PnlPotentialExclusions.objects.all().values())
    pnl_potential_exclusions.rename(columns={'deal_name': 'TradeGroup', 'sleeve': 'Sleeve'}, inplace=True)

    pnl_potential_scenarios_data = []

    # Add Current Portfolio to P&L Potential Scenarios from Dates and Exclusions table
    if not pnl_potential_dates.empty:
        for each_row in pnl_potential_dates.iterrows():
            pnl_potential_scenarios_data.append(['Current Portfolio', each_row[1]['Name'], each_row[1]['Sleeve'], 100])

    if not pnl_potential_exclusions.empty:
        for each_row in pnl_potential_exclusions.iterrows():
            pnl_potential_scenarios_data.append(['Current Portfolio', each_row[1]['TradeGroup'], each_row[1]['Sleeve'], 100])

    # ADD INCREMENTAL DEALS FOR MAIN SCENARIO
    incremental_qs = pd.DataFrame.from_records(PnlPotentialIncremental.objects.all().values())
    for each_row in incremental_qs.iterrows():
        pnl_potential_scenarios_data.append(['Current Portfolio', each_row[1]['incremental_name'], each_row[1]['sleeve'], 100])

    pnl_potential_scenarios = pd.DataFrame(columns=['Scenario Name', 'Name', 'Sleeve', 'Pct'],
                                           data=pnl_potential_scenarios_data)

    scenarios_df = pd.DataFrame.from_records(PnlPotentialScenarios.objects.filter(
        sleeve__in=['Merger Arbitrage', 'Credit Opportunities']).values())
    if not scenarios_df.empty:
        scenarios_df.drop(columns=['id'], inplace=True)
        scenarios_df.rename(columns={'date_deal_name': 'Name', 'sleeve': 'Sleeve', 'scenario_value': 'Pct',
                                     'scenario_name': 'Scenario Name'}, inplace=True)
        pnl_potential_scenarios = pnl_potential_scenarios.append(scenarios_df, sort=False)

    aed_ess_df = get_ess_df()
    # Hard-code main scenarion based on empty dataframe (Empty JSON response instead)
    # Then loop over the Unique Scenarios in the DF and construct your JSON Response...
    scenario_response = OrderedDict()

    if aed_df.empty:
        df_dict = {'aed_df': aed_df, 'aed_credit_df': pd.DataFrame(), 'aed_ess_df': aed_ess_df,
                   'scenario_response_df': pd.DataFrame()}
        return df_dict, scenario_response, {'aed_nav': 0, 'implied_prob_deduct': implied_prob_deduct}
    aed_nav = aed_df['aed_nav'].unique()[0]
    aed_credit_df = aed_df[aed_df['sleeve'] == 'Credit Opportunities']
    aed_credit_df['closing_date'] = aed_credit_df['closing_date'].apply(str)
    aed_credit_df = aed_credit_df[['tradegroup', 'sleeve', 'closing_date', 'current_mkt_val_pct',
                                   'customized_mkt_val_pct', 'is_customized', 'gross_ror', 'aed_nav', 'pnl_potential',
                                   'id']]
    rename_cols = {'tradegroup': 'Tradegroup', 'current_mkt_val_pct': 'CurrentMktVal %', 'closing_date': 'Closing Date',
                   'gross_ror': 'Gross RoR', 'customized_mkt_val_pct': 'Customized MktVal%',
                   'pnl_potential': 'Pnl Potential', 'aed_nav': 'AED NAV', 'is_customized': 'Is MktVal updated',
                   'implied_probability': 'Implied Probability',
                   'custom_implied_probability': 'Custom Implied Probability'}

    aed_credit_df.rename(columns=rename_cols, inplace=True)
    for index, scenario in enumerate(pnl_potential_scenarios['Scenario Name'].unique()):
        pnl_potential_field = 'max_pnl_potential' if scenario.lower() == 'current portfolio' else 'pnl_potential'
        response = OrderedDict()
        scenario_slice = pnl_potential_scenarios[pnl_potential_scenarios['Scenario Name'] == scenario]
        # Iterate over Unique Sleeves...
        for sleeve in scenario_slice['Sleeve'].unique():
            scenario_data = []
            scenario_total_pnl = 0
            sleeve_scenario_df = scenario_slice[scenario_slice['Sleeve'] == sleeve]
            # Process all Unique Names with their Percentages
            known_deals_total = 0
            for name, pct in zip(sleeve_scenario_df['Name'], sleeve_scenario_df['Pct']):
                # Process Incremental Deals as $ value .....
                # Search for this name in the Dates Dataframe and get the Closing Date
                try:
                    start_closing_date = pnl_potential_dates[pnl_potential_dates['Name'] == name].iloc[0]['Start Date']
                    end_closing_date = pnl_potential_dates[pnl_potential_dates['Name'] == name].iloc[0]['End Date']
                    aed_df_sliced = aed_df[((aed_df['closing_date'] >= start_closing_date) &
                                            (aed_df['closing_date'] <= end_closing_date))]
                    aed_df_sliced = aed_df_sliced[aed_df_sliced['sleeve'] == sleeve]
                    unique_tgs = []
                    if not pnl_potential_exclusions.empty:
                        unique_tgs = pnl_potential_exclusions.TradeGroup.unique()
                    df = aed_df_sliced[aed_df_sliced['tradegroup'].isin(unique_tgs)]
                    exclusion_sum_100_pct = 0
                    if not df.empty:
                        exclusion_sum_100_pct = df[pnl_potential_field].sum()
                    total_pnl = aed_df_sliced[pnl_potential_field].sum()
                    if index > 0:
                        total_pnl = (total_pnl - exclusion_sum_100_pct) * float(pct) * 0.01
                    else:
                        total_pnl -= exclusion_sum_100_pct
                    known_deals_total += total_pnl
                except (IndexError, KeyError):
                    # Index Error implies this is Exclusion Deals
                    # Get the Exclusion $ numbers
                    exclusion_df = aed_df[(aed_df['tradegroup'] == name) & (aed_df['sleeve'] == sleeve)]
                    total_pnl = 0
                    if not exclusion_df.empty:
                        total_pnl = float(exclusion_df[pnl_potential_field].iloc[0]) * float(pct)/100
                        known_deals_total += total_pnl
                    # Handle Incremental Deals scenario
                    if name.lower() == 'incremental deals':
                        scenario_data.append(('KNOWN DEALS', known_deals_total))
                        filtered_df = incremental_qs[(incremental_qs['sleeve'] == sleeve) &
                                                     (incremental_qs['incremental_name'] == name)]
                        filtered_df = filtered_df.groupby(['sleeve', 'incremental_name']).sum()
                        if not filtered_df.empty:
                            total_pnl = filtered_df.iloc[0]['incremental_value'] *  float(pct) * 0.01
                        else:
                            total_pnl = float(pct)

                scenario_total_pnl += total_pnl
                scenario_data.append((name, total_pnl))

            # Attribute Total P&L across the Scenario
            known_deals_index = [i for i, v in enumerate(scenario_data) if v[0] == 'KNOWN DEALS']
            if known_deals_index:
                scenario_data[known_deals_index[0]] = ('KNOWN DEALS', known_deals_total)
            if 'incremental deals' not in sleeve_scenario_df['Name'].str.lower().tolist():
                scenario_data.append(('KNOWN DEALS', known_deals_total))
            scenario_data.append(('Total P&L', scenario_total_pnl))
            response[sleeve] = OrderedDict(scenario_data)

        scenario_response[scenario] = response
    scenario_response_df = pd.DataFrame(columns=['Index', 'sleeve'])
    for index, scenario in enumerate(scenario_response.keys()):
        counter = 0
        current_scenario_data = scenario_response[scenario]
        scenario_response_df[scenario] = ''
        for sleeve in current_scenario_data.keys():
            inner_dictionary = current_scenario_data[sleeve]
            for each_index in inner_dictionary.keys():
                if index > 0:
                    counter = get_row_id(sleeve, each_index, scenario_response_df)
                scenario_response_df.loc[counter, 'Index'] = each_index
                scenario_response_df.loc[counter, scenario] = inner_dictionary[each_index]
                scenario_response_df.loc[counter, 'sleeve'] = sleeve
                counter += 1

    aed_df = aed_df[aed_df['sleeve'] == 'Merger Arbitrage']
    currmktval_sum = sum(aed_df['current_mkt_val_pct'])
    custommktval_sum = sum(aed_df['customized_mkt_val_pct'])
    curr_mkt_curr_prob = round(sum((aed_df['current_mkt_val_pct'] / currmktval_sum) * aed_df['implied_probability']), 2)
    curr_mkt_custom_prob = round(sum((aed_df['current_mkt_val_pct'] / currmktval_sum) * aed_df['custom_implied_probability']), 2)
    custom_mkt_curr_prob = round(sum((aed_df['customized_mkt_val_pct'] / custommktval_sum) * aed_df['implied_probability']), 2)
    custom_mkt_custom_prob = round(sum((aed_df['customized_mkt_val_pct'] / custommktval_sum) * aed_df['custom_implied_probability']), 2)
    curr_mkt_gross_ror = sum((aed_df['current_mkt_val_pct'] / currmktval_sum) * aed_df['gross_ror'])
    custom_mkt_gross_ror = sum((aed_df['customized_mkt_val_pct'] / custommktval_sum) * aed_df['gross_ror'])
    curr_weighted_dtc = sum((aed_df['current_mkt_val_pct'] / currmktval_sum) * aed_df['days_to_close'])
    custom_weighted_dtc = sum((aed_df['customized_mkt_val_pct'] / custommktval_sum) * aed_df['days_to_close'])
    curr_mkt_curr_prob_ann = round((curr_mkt_gross_ror / curr_weighted_dtc) * 365, 2)
    custom_mkt_curr_prob_ann = round((custom_mkt_gross_ror / custom_weighted_dtc) * 365, 2)
    aed_df['custom_last_price'] = ((aed_df['deal_value'] - aed_df['deal_downside']) * 0.01 *
                                   aed_df['custom_implied_probability']) + aed_df['deal_downside']
    aed_df['custom_gross_ror'] = 100 * (aed_df['deal_value'] / aed_df['custom_last_price'] - 1).fillna(0)
    curr_mkt_custom_gross_ror = sum((aed_df['current_mkt_val_pct'] / currmktval_sum) * aed_df['custom_gross_ror'])
    custom_mkt_custom_gross_ror = sum((aed_df['customized_mkt_val_pct'] / custommktval_sum) * aed_df['custom_gross_ror'])
    curr_mkt_custom_prob_custom_ror = round((curr_mkt_custom_gross_ror / curr_weighted_dtc) * 365, 2)
    custom_mkt_custom_prob_custom_ann = round((custom_mkt_custom_gross_ror / custom_weighted_dtc) * 365, 2)
    dict_values = {'curr_mkt_curr_prob': curr_mkt_curr_prob, 'curr_mkt_custom_prob': curr_mkt_custom_prob,
                   'custom_mkt_curr_prob': custom_mkt_curr_prob, 'custom_mkt_custom_prob': custom_mkt_custom_prob,
                   'implied_prob_deduct': implied_prob_deduct, 'aed_nav': aed_nav,
                   'curr_mkt_curr_prob_ann': curr_mkt_curr_prob_ann,
                   'custom_mkt_curr_prob_ann': custom_mkt_curr_prob_ann,
                   'curr_mkt_custom_prob_custom_ror': curr_mkt_custom_prob_custom_ror,
                   'custom_mkt_custom_prob_custom_ann': custom_mkt_custom_prob_custom_ann}
    current_pnl_loss = round(sum(((aed_df['custom_last_price'] / aed_df['target_last_price']) - 1) * aed_df['current_mkt_val_pct'] * 0.01 * aed_nav), 2)
    custom_pnl_loss = round(sum(((aed_df['custom_last_price'] / aed_df['target_last_price']) - 1) * aed_df['customized_mkt_val_pct'] * 0.01 * aed_nav), 2)
    temp_data_length = len(scenario_response_df.columns.values) - 3
    temp_data = ['Pnl Loss - ' + str(implied_prob_deduct) + '% Lower Imp Prob', 'Merger Arbitrage', current_pnl_loss] +\
                temp_data_length * [custom_pnl_loss]
    temp_df = pd.DataFrame(columns=scenario_response_df.columns.values, data=[temp_data])
    scenario_response_df = scenario_response_df.append(temp_df)
    aed_df = aed_df[['tradegroup', 'sleeve', 'closing_date', 'current_mkt_val_pct', 'customized_mkt_val_pct',
                     'is_customized', 'implied_probability', 'custom_implied_probability', 'gross_ror', 'aed_nav',
                     'pnl_potential', 'id']]
    aed_df['closing_date'] = aed_df['closing_date'].apply(str)
    aed_df.rename(columns=rename_cols, inplace=True)
    df_dict = {'aed_df': aed_df, 'aed_credit_df': aed_credit_df, 'aed_ess_df': aed_ess_df,
               'scenario_response_df': scenario_response_df}
    return df_dict, scenario_response, dict_values


def get_row_id(sleeve, index, df):
    df = df[(df['sleeve'] == sleeve) & (df['Index'] == index)]
    if not df.empty:
        return df.iloc[0].name


def get_realized_gross_pnl():
    gross_ytd_pnl = PnlMonitors.objects.filter(fund='AED').latest('last_updated').gross_ytd_pnl
    if gross_ytd_pnl:
        gross_ytd_pnl = float(gross_ytd_pnl.replace(',', ''))
        return gross_ytd_pnl
    return 0


def get_ess_df(new_scenario=False, scenario_name='Current Portfolio'):
    """
    Returns a dataframe containing only AED rows if new_scenario is True
    If new_scenario is False, then also checks for customized data and 'Current Portfolio' data
    """
    aed_df = pd.read_sql_query("Select AUM as `aed_nav`, tradegroup, sleeve, sum(CurrentMktVal_Pct) as current_mkt_val_pct " \
                               "from wic.daily_flat_file_db where Flat_file_as_of = (select max(Flat_file_as_of) " \
                               "from wic.daily_flat_file_db) and fund like 'AED' and amount<>0 and AlphaHedge " \
                               "like 'Alpha' group by AUM, tradegroup, sleeve", con=connection)
    aed_nav = aed_df['aed_nav'].unique()[0]
    aed_ess_df = aed_df[aed_df['sleeve'] == 'Equity Special Situations']
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD +
                           "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()
    ess_ideas_df = pd.read_sql_query(get_ess_idea_database_query(), con)
    con.close()
    ess_idea_db = ess_ideas_df[['tradegroup', 'alpha_ticker', 'pt_up', 'pt_wic', 'pt_down', 'model_up', 'model_wic',
                                'model_down']]
    aed_ess_df = pd.merge(aed_ess_df, ess_idea_db, how='left', on=['tradegroup'])

    # Merge with current constituents to gather Up Probabilities
    if new_scenario:
        ess_customized_df = pd.DataFrame()
    else:
        ess_customized_df = pd.DataFrame.from_records(PnlPotentialESSConstituents.objects.all().values())
    # Take a slice of customized df to reflect updated prices
    # Get Extra (Potententially new tradegroups which are not in the ESS Constituents yet. Handle separately

    def get_customized_mkt_val(row):
        return row['customized_mkt_val_pct'] if row['is_customized'] else row['current_mkt_val_pct']

    if not ess_customized_df.empty:
        scenarios = ess_customized_df['scenario_name'].unique()
        if 'Current Portfolio' not in scenarios:
            current_portfolio_df = aed_ess_df.copy()
            current_portfolio_df['downside_value'] = aed_ess_df['model_down']
            current_portfolio_df['upside_value'] = aed_ess_df['model_up']
            current_portfolio_df['up_probability'] = 100
            current_portfolio_df['down_probability'] = 0
            current_portfolio_df['upside_field'] = 'model_up'
            current_portfolio_df['downside_field'] = 'model_down'
            current_portfolio_df['scenario_name'] = 'Current Portfolio'
            current_portfolio_df['scenario_type'] = 'ESS'
            current_portfolio_df['is_customized'] = False
            ess_customized_df = ess_customized_df.append(current_portfolio_df)
        new_aed_tradegroups = aed_ess_df[~(aed_ess_df['tradegroup'].isin(ess_customized_df['tradegroup'].unique()))]
        ess_customized_df = ess_customized_df[['tradegroup', 'up_probability', 'down_probability', 'upside_field',
                                               'downside_field', 'scenario_name', 'scenario_type', 'upside_value',
                                               'downside_value', 'pnl_potential_100', 'pnl_potential_50',
                                               'pnl_potential_0', 'customized_mkt_val_pct', 'is_customized']]
        aed_ess_df = pd.merge(aed_ess_df, ess_customized_df, how='left', on=['tradegroup'])
        aed_ess_df = aed_ess_df.dropna(subset=['upside_field', 'downside_field'])
        if not new_aed_tradegroups.empty:
        # Add new Tradegroups from Flat file (if any)
            scenario_types = ess_customized_df['scenario_type'].unique()

            for s_name, s_type in list(itertools.product(scenarios, scenario_types)):
                new_aed_tradegroups['downside_value'] = new_aed_tradegroups['model_down']
                new_aed_tradegroups['upside_value'] = new_aed_tradegroups['model_up']
                new_aed_tradegroups['up_probability'] = 100
                new_aed_tradegroups['down_probability'] = 0
                new_aed_tradegroups['upside_field'] = 'model_up'
                new_aed_tradegroups['downside_field'] = 'model_down'
                new_aed_tradegroups['scenario_name'] = s_name
                new_aed_tradegroups['scenario_type'] = s_type
                new_aed_tradegroups['is_customized'] = False

                aed_ess_df = pd.concat([aed_ess_df, new_aed_tradegroups])
    else:
        aed_ess_df['downside_value'] = aed_ess_df['model_down']
        aed_ess_df['upside_value'] = aed_ess_df['model_up']
        aed_ess_df['up_probability'] = 100
        aed_ess_df['down_probability'] = 0
        aed_ess_df['upside_field'] = 'model_up'
        aed_ess_df['downside_field'] = 'model_down'
        aed_ess_df['scenario_name'] = scenario_name
        aed_ess_df['scenario_type'] = 'ESS'
        aed_ess_df['is_customized'] = False

    aed_ess_df['is_customized'].fillna(False, inplace=True)
    aed_ess_df['customized_mkt_val_pct'] = aed_ess_df.apply(get_customized_mkt_val, axis=1)

    unique_alpha_tickers = aed_ess_df['alpha_ticker'].dropna().unique()
    api_host = bbgclient.bbgclient.get_next_available_host()
    live_price_df = pd.DataFrame.from_dict(bbgclient.bbgclient.get_secid2field(
        unique_alpha_tickers, 'tickers', ['PX_LAST'], req_type='refdata', api_host=api_host
        ), orient='index').reset_index()
    if live_price_df.empty:
        live_price_df = pd.DataFrame(columns=['alpha_ticker', 'PX_LAST'])
    live_price_df['PX_LAST'] = live_price_df['PX_LAST'].apply(lambda x: x[0])
    live_price_df.columns = ['alpha_ticker', 'PX_LAST']
    aed_ess_df = pd.merge(aed_ess_df, live_price_df, how='left', on='alpha_ticker')
    float_cols = ['model_up', 'model_wic', 'model_down', 'pt_up', 'pt_wic', 'pt_down', 'current_mkt_val_pct', 'PX_LAST',
                  'upside_value', 'downside_value', 'up_probability', 'down_probability', 'customized_mkt_val_pct']
    aed_ess_df[float_cols] = aed_ess_df[float_cols].astype(float)

    aed_ess_df['current_mkt_val_pct'] = aed_ess_df['current_mkt_val_pct'].abs()
    aed_ess_df['customized_mkt_val_pct'] = aed_ess_df['customized_mkt_val_pct'].abs()
    # Recalculate Upside Values & Downside Values based on the formuale

    for index, row in aed_ess_df.iterrows():
        aed_ess_df.at[index,'upside_value'] = row[row['upside_field']]
        aed_ess_df.at[index,'downside_value'] = row[row['downside_field']]

    aed_ess_df['upside_value'] = ((aed_ess_df['upside_value'] - aed_ess_df['model_down']) *
                                  aed_ess_df['up_probability'] * 0.01) + aed_ess_df['model_down']

    aed_ess_df['downside_value'] = ((aed_ess_df['upside_value'] - aed_ess_df['model_down']) *
                                    aed_ess_df['down_probability'] * 0.01) + aed_ess_df['model_down']

    aed_ess_df['pnl_potential_100'] = ((aed_ess_df['upside_value'] / aed_ess_df['PX_LAST']) -1) * \
                                             aed_ess_df['customized_mkt_val_pct']/100 * aed_nav

    aed_ess_df['pnl_potential_50'] = (((((aed_ess_df['model_up'] - aed_ess_df['model_down']) * 0.5) + \
        aed_ess_df['model_down'])/aed_ess_df['PX_LAST']) -1) * aed_ess_df['customized_mkt_val_pct']/100 * aed_nav

    aed_ess_df['pnl_potential_0'] = ((aed_ess_df['downside_value'] / aed_ess_df['PX_LAST']) -1) * \
        aed_ess_df['customized_mkt_val_pct']/100 * aed_nav
    aed_ess_df.rename(columns={'pnl_potential_100': 'Pnl Potential 100%', 'pnl_potential_50': 'Pnl Potential 50%',
                               'pnl_potential_0': 'Pnl Potential 0%'}, inplace=True)

    return aed_ess_df


def format_data():
    df_dict, scenario_response, dict_values = get_pnl_potential()
    aed_df = df_dict.get('aed_df')
    aed_credit_df = df_dict.get('aed_credit_df')
    aed_ess_df = df_dict.get('aed_ess_df')
    scenario_response_df = df_dict.get('scenario_response_df')
    aed_nav = dict_values.get('aed_nav')
    # Current Portfolio Processing
    ess_required_return = PnlPotentialOtherValues.objects.filter(field_name='required_return').first()
    ess_required_return = ess_required_return.field_value if ess_required_return else 8
    ess_required_return_1 = ess_required_return - 1
    ess_required_return_2 = ess_required_return - 2
    main_shortfall = 'Shortfall ('+str(ess_required_return)+'% )'

    cuts = ['Expected Gross PnL', 'Realized Gross PnL', 'Total Gross PnL', 'Gross Return', 'Net Return',
            'Required Return for '+str(ess_required_return)+'%', main_shortfall,
            'Required Return for '+str(ess_required_return_1)+'%', 'Shortfall ('+str(ess_required_return_1)+'%)',
            'Required Return for '+str(ess_required_return_2)+'%', 'Shortfall ('+str(ess_required_return_2)+'%)',
            'ESS 100% PnL Capture', 'Gross Return (100%)', 'ESS 50% PnL Capture',
            'Gross Return (50%)', 'ESS 0% PnL Capture', 'Gross Return (0%)']

    dict_values.update({'ess_required_return': ess_required_return})
    if not scenario_response:
        df_dict = {'scenario_response_df': pd.DataFrame(), 'scenario_processing_df': pd.DataFrame(),
                   'ess_achievement_returns_df': pd.DataFrame(), 'aed_ess_df': aed_ess_df, 'aed_df': aed_df,
                   'aed_credit_df': aed_credit_df}
        return df_dict, dict_values
    scenario_processing_df = pd.DataFrame(columns=['Index'], data=cuts)
    for scenario in scenario_response:

        ess_customized_df = aed_ess_df[aed_ess_df['scenario_name'] == scenario]
        counter = 0
        scenario_processing_df[scenario] = ''

        inner_dict = scenario_response[scenario]
        sleeves = inner_dict.keys()
        expected_gross_pnl = 0
        for slv in sleeves:
            expected_gross_pnl += inner_dict[slv]['Total P&L']

        scenario_processing_df.loc[counter, scenario] = expected_gross_pnl
        realized_gross_pnl = get_realized_gross_pnl()
        counter += 1
        scenario_processing_df.loc[counter, scenario] = realized_gross_pnl
        counter += 1

        total_gross_pnl = expected_gross_pnl + realized_gross_pnl
        scenario_processing_df.loc[counter, scenario] = total_gross_pnl
        counter += 1

        expected_gross_return = 1e2*total_gross_pnl/aed_nav
        scenario_processing_df.loc[counter, scenario] = expected_gross_return
        counter += 1

        expected_net_return = expected_gross_return - 1.3
        scenario_processing_df.loc[counter, scenario] = expected_net_return
        counter += 1

        required_return_main_pct = ess_required_return * aed_nav * 0.01
        scenario_processing_df.loc[counter, scenario] = required_return_main_pct
        counter += 1

        shortfall_main = total_gross_pnl - required_return_main_pct
        # If Shortfall is +ve then update it to 0
        if float(shortfall_main) > 0:
            shortfall_main = 0
        scenario_processing_df.loc[counter, scenario] = shortfall_main
        counter += 1

        # Process for Required Returns -1 & -2 from defined returns
        required_return_1_pct = ess_required_return_1 * aed_nav * 0.01
        scenario_processing_df.loc[counter, scenario] = required_return_1_pct
        counter += 1

        shortfall_1 = total_gross_pnl - required_return_1_pct
        if float(shortfall_1) > 0:
            shortfall_1 = 0
        scenario_processing_df.loc[counter, scenario] = shortfall_1
        counter += 1

        # x-2 Return required
        required_return_2_pct = ess_required_return_2 * aed_nav * 0.01
        scenario_processing_df.loc[counter, scenario] = required_return_2_pct
        counter += 1

        shortfall_2 = total_gross_pnl - required_return_2_pct
        # If Shortfall is +ve then update it to 0
        if float(shortfall_2) > 0:
            shortfall_2 = 0
        scenario_processing_df.loc[counter, scenario] = shortfall_2
        counter += 1

        ess_100_pnl = ess_customized_df[ess_customized_df['scenario_type'] == 'ESS']['Pnl Potential 100%'].sum()
        scenario_processing_df.loc[counter, scenario] = ess_100_pnl
        counter += 1
        ess_100_pnl_gross_return = 1e2*(ess_100_pnl + total_gross_pnl)/aed_nav
        scenario_processing_df.loc[counter, scenario] = ess_100_pnl_gross_return
        counter += 1

        ess_50_pnl = aed_ess_df['Pnl Potential 50%'].sum()
        scenario_processing_df.loc[counter, scenario] = ess_50_pnl
        counter += 1

        ess_50_pnl_gross_return = 1e2*(ess_50_pnl + total_gross_pnl)/aed_nav
        scenario_processing_df.loc[counter, scenario] = ess_50_pnl_gross_return
        counter += 1

        ess_0_pnl = ess_customized_df[ess_customized_df['scenario_type'] == 'ESS']['Pnl Potential 0%'].sum()
        scenario_processing_df.loc[counter, scenario] = ess_0_pnl
        counter += 1

        ess_0_pnl_gross_return = 1e2 * (ess_0_pnl + total_gross_pnl)/aed_nav
        scenario_processing_df.loc[counter, scenario] = ess_0_pnl_gross_return

    if not scenario_response_df.empty and not scenario_processing_df.empty:
        scenario_processing_df['sleeve'] = ' + '.join(i for i in scenario_response_df.sleeve.unique().tolist())
    capital_scenarios = [0.05, 0.10, 0.15, 0.20]
    ess_achievement_returns_df = pd.DataFrame(columns=['Index'])

    for scenario in scenario_response:
        counter = 0
        for capital in capital_scenarios:
            shortfall = 0
            shortfall_df = scenario_processing_df[scenario_processing_df['Index'] == main_shortfall]
            if not shortfall_df.empty:
                shortfall = abs(shortfall_df.iloc[0][scenario])
            ess_achievement_returns_df.loc[counter, 'Index'] = str(capital * 100) + '%'
            ess_achievement_returns_df.loc[counter, scenario] = 1e2 * float(shortfall) / (float(capital) * aed_nav)
            counter += 1
    ess_achievement_returns_df['sleeve'] = 'ESS Capital'
    aed_ess_df.rename(columns={'tradegroup': 'Tradegroup', 'current_mkt_val_pct': 'Current MktVal %',
                               'alpha_ticker': 'Alpha Ticker', 'pt_up': 'PT UP', 'pt_wic': 'PT WIC',
                               'pt_down': 'PT Down', 'model_up': 'Model Up', 'model_wic': 'Model WIC',
                               'model_down': 'Model Down', 'PX_LAST': 'PX LAST'}, inplace=True)
    df_dict = {'scenario_response_df': scenario_response_df, 'scenario_processing_df': scenario_processing_df,
               'ess_achievement_returns_df': ess_achievement_returns_df, 'aed_ess_df': aed_ess_df, 'aed_df': aed_df,
               'aed_credit_df': aed_credit_df}
    return df_dict, dict_values


def get_ess_idea_database_query():
    """
    Returns a general query for ess idea database. Filter for archived and backlogged needs to be applied.
    """
    query = "SELECT  A.tradegroup, A.alpha_ticker, A.pt_up, A.pt_wic, A.pt_down,  "\
            "IF(model_up=0, A.pt_up, " \
            "model_up) as model_up, IF(model_down=0, A.pt_down, model_down) as model_down, IF(model_wic=0, A.pt_wic, " \
            "model_wic) as model_wic FROM prod_wic_db.risk_ess_idea AS A  INNER JOIN " \
            "(SELECT deal_key, MAX(version_number) AS max_version FROM  prod_wic_db.risk_ess_idea GROUP BY deal_key )" \
            " AS B " \
            "ON A.deal_key = B.deal_key AND A.version_number = B.max_version {is_archived_filter} AND " \
            "A.status {status_filter} LEFT JOIN " \
            "(SELECT DISTINCT X.deal_key,X.pt_up as model_up, X.pt_down AS model_down, X.pt_wic " \
            "AS model_wic FROM prod_wic_db.risk_ess_idea_upside_downside_change_records  AS X " \
            "INNER JOIN (SELECT deal_key, MAX(date_updated) AS MaxDate " \
            "FROM prod_wic_db.risk_ess_idea_upside_downside_change_records " \
            "GROUP BY deal_key) AS Y ON " \
            "X.deal_key = Y.deal_key WHERE X.date_updated = Y.MaxDate) AS ADJ ON " \
            "ADJ.deal_key = A.deal_key".format(is_archived_filter="AND A.is_archived = 0 ",
                                               status_filter="!= 'Backlogged' ")
    return query


def get_aed_sleeves_performance():
    """ Returns the YTD, QTD and MTD performance for Sleeves in AED.
    :return dollar_df, bps_df (Performances captures in $ and basis points)
    """
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)

    con = engine.connect()
    sleeve_dollar_perf = pd.DataFrame()
    sleeve_bps_perf = pd.DataFrame()
    try:
        as_of = '(select max(`Date`)'
        dollar_df = pd.read_sql_query('Select * from ' + settings.CURRENT_DATABASE +
                                      '.funds_snapshot_tradegroupperformancefundnavdollar where date=(select ' +
                                      'max(date) from ' + settings.CURRENT_DATABASE +
                                      '.funds_snapshot_tradegroupperformancefundnavdollar) and fund = "AED";', con=con)
        bps_df = pd.read_sql_query('Select * from ' + settings.CURRENT_DATABASE +
                                   '.funds_snapshot_tradegroupperformancefundnavbps where date=(select ' +
                                   'max(date) from ' + settings.CURRENT_DATABASE +
                                   '.funds_snapshot_tradegroupperformancefundnavbps) and fund = "AED";', con=con)
        
        dollar_df = dollar_df[['date', 'fund', 'sleeve', 'ytd_dollar', 'qtd_dollar', 'mtd_dollar']].copy()
        bps_df = bps_df[['date', 'fund', 'sleeve', 'ytd_bps', 'qtd_bps', 'mtd_bps']].copy()

        sleeve_dollar_perf = dollar_df.groupby(['date', 'fund', 'sleeve']).agg(
            [('Gross +ve', lambda x: x[x > 0].sum()), ('Gross -ve', lambda x: x[x < 0].sum()),
             ('Net', lambda x: x.sum())]).reset_index()

        sleeve_bps_perf = bps_df.groupby(['date', 'fund', 'sleeve']).agg(
            [('Gross +ve', lambda x: x[x > 0].sum()), ('Gross -ve', lambda x: x[x < 0].sum()),
             ('Net', lambda x: x.sum())]).reset_index()

        sleeve_bps_perf.drop(columns=['date', 'fund'], inplace=True)
        sleeve_dollar_perf.drop(columns=['date', 'fund'], inplace=True)
        req_sleeves = ['CREDIT OPPORTUNITIES', 'EQUITY SPECIAL SITUATIONS', 'MERGER ARBITRAGE']
        sleeve_dict = {'CREDIT': 'CREDIT OPPORTUNITIES', 'ESS': 'EQUITY SPECIAL SITUATIONS', 'M&A': 'MERGER ARBITRAGE'}
        sleeve_bps_perf = sleeve_bps_perf[sleeve_bps_perf['sleeve'].isin(req_sleeves)]
        sleeve_dollar_perf = sleeve_dollar_perf[sleeve_dollar_perf['sleeve'].isin(req_sleeves)]
        try:
            max_date = LossBudgets.objects.latest('updated_at').updated_at
        except LossBudgets.DoesNotExist:
            max_date = None
        loss_budget_df = pd.DataFrame()
        if max_date:
            loss_budget_df = pd.DataFrame.from_records(LossBudgets.objects.filter(updated_at=max_date).values('sleeve', 'loss_value'))
            loss_budget_df['sleeve'] = loss_budget_df['sleeve'].apply(lambda x: sleeve_dict[x])
            loss_budget_df.sort_values(by=['sleeve'], inplace=True)
            aum_value = con.execute('select DISTINCT AUM from wic.daily_flat_file_db where flat_file_as_of = ' + \
                                    '(Select max(flat_file_as_of) from wic.daily_flat_file_db) and fund = "AED";')
            aum = 0
            for row in aum_value:
                aum = row['AUM']
                break
            loss_budget_df['ytd_dollar Risk Budget'] = aum * loss_budget_df['loss_value'] * 0.01
            loss_budget_df['qtd_dollar Risk Budget'] = (aum * loss_budget_df['loss_value'] * 0.01) / 4
            loss_budget_df['mtd_dollar Risk Budget'] = (aum * loss_budget_df['loss_value'] * 0.01) / 12
            loss_budget_df.rename(columns={'sleeve': 'sleeve'}, inplace=True)
            loss_budget_df.drop(columns=['loss_value'], inplace=True)
            for index, row in loss_budget_df.iterrows():
                sleeve = row['sleeve']
                field = 'Gross -ve'
                if sleeve and sleeve.upper() == 'EQUITY SPECIAL SITUATIONS':
                    field = 'Net'
                time_period = ['ytd_dollar', 'qtd_dollar', 'mtd_dollar']
                for period in time_period:
                    loss_budget_df.at[loss_budget_df['sleeve'] == sleeve, period + ' Gross -ve'] = sleeve_dollar_perf[sleeve_dollar_perf['sleeve'] == sleeve].iloc[0][period][field]
                    loss_budget_df.at[loss_budget_df['sleeve'] == sleeve, period + ' Budget'] = loss_budget_df[period + ' Gross -ve'] + loss_budget_df[period + ' Risk Budget']
            loss_budget_df = loss_budget_df.reindex(['sleeve'] + sorted(loss_budget_df.columns[1:], reverse=True), axis=1)

    except Exception as e:
        print(e)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    finally:
        con.close()

    sleeve_dollar_perf.columns = [(x[0]+' '+x[1]).strip() for x in sleeve_dollar_perf.columns.values]
    sleeve_bps_perf.columns = [(x[0]+' '+x[1]).strip() for x in sleeve_bps_perf.columns.values]
    sleeve_dict = {'CREDIT OPPORTUNITIES': 'CREDIT', 'EQUITY SPECIAL SITUATIONS': 'ESS', 'MERGER ARBITRAGE': 'M&A'}
    sleeve_dollar_perf['sleeve'] = sleeve_dollar_perf['sleeve'].apply(lambda x: sleeve_dict[x])
    sleeve_dollar_perf.rename(columns={'sleeve': 'Sleeve'}, inplace=True)
    sleeve_bps_perf['sleeve'] = sleeve_bps_perf['sleeve'].apply(lambda x: sleeve_dict[x])
    sleeve_bps_perf.rename(columns={'sleeve': 'Sleeve'}, inplace=True)
    loss_budget_df['sleeve'] = loss_budget_df['sleeve'].apply(lambda x: sleeve_dict[x])
    loss_budget_df.rename(columns={'sleeve': 'Sleeve'}, inplace=True)
    return sleeve_dollar_perf, sleeve_bps_perf, loss_budget_df
