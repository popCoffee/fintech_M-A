import operator
from functools import reduce

from django.db import connection
from django.db.models import Q
from django.http import JsonResponse
import pandas as pd

from risk.forms import MaDealsRiskFactorsForm
from risk.models import MA_Deals, MA_Deals_Risk_Factors
from risk_reporting.models import DailyNAVImpacts


FUND_NAMES = ['AED', 'ARB', 'LG', 'MALT', 'TACO', 'PRELUDE', 'EVNT']
REGULATORY_TAB_DICT = {'Acquirer SH Vote Reqd': 'acq_sh_vote_requirement', 'SEC': 'sec_requirement',
                       'UK CMA': 'uk_cma_requirement', 'HSR': 'hsr_requirement', 'MOFCOM': 'mofcom_requirement',
                       'CIFIUS': 'cifius_requirement', 'EC': 'ec_requirement', 'ACCC': 'accc_requirement',
                       'CANADIAN': 'investment_canada_requirement', 'CADE': 'cade_requirement',
                       'FCC': 'fcc_requirement', 'PUC': 'puc_requirement'}
MAIN_TABS_DICT = {'Regulatory Risk': REGULATORY_TAB_DICT,
                  'Country Risk': ['other_country_regulatory_risk_one', 'other_country_regulatory_risk_two'],
                  'Divestitures Reqd': 'divestitures_required',
                  'Target SH Vote Reqd': 'target_sh_vote_required_percentage', 'Valuation': 'fair_valuation',
                  'Acquirer Becomes Target': 'acquirer_becomes_target', 'PE Deals': 'strategic_pe',
                  'Commodities Risk': 'commodity_risk', 'Cyclical Industry': 'cyclical_industry',
                  'Inversion Risk': 'is_inversion_deal_or_tax_avoidance'}


def get_summary_for_risk_factors():
    response = {'msg': 'Failed', 'data': {}}
    data = {}
    try:
        risk_form = MaDealsRiskFactorsForm()
        for fund in FUND_NAMES:
            fund_tab_data = {}
            for main_tab in MAIN_TABS_DICT:
                if main_tab == 'Regulatory Risk':
                    main_tab_data = {}
                    for regulatory_tab in MAIN_TABS_DICT[main_tab]:
                        field_names_list = [REGULATORY_TAB_DICT[regulatory_tab]]
                        tab_data = get_tabs_data(field_names_list, risk_form)
                        main_tab_data[regulatory_tab] = tab_data
                    fund_tab_data['Regulatory Risk'] = main_tab_data
                else:
                    main_tab_data = {}
                    field_names_list = MAIN_TABS_DICT[main_tab]
                    field_names_list = field_names_list if isinstance(field_names_list, (list,)) else [field_names_list]
                    tab_data = get_tabs_data(field_names_list, risk_form)
                    fund_tab_data[main_tab] = tab_data
            data[fund] = fund_tab_data
        response = {'msg': 'Success', 'data': data}
    except Exception as e:
        response = {'msg': 'Failed', 'data': {}}
    return response


def get_unique_deal_names(dataframe, field_names, value):
    result = []
    if not dataframe.empty and field_names and value:
        for field_name in field_names:
            result += dataframe[dataframe[field_name] == value].deal_name.unique().tolist()
        return list(set(result))
    return result


def get_select_choices_list(fields_list, form):
    result = []
    if form and fields_list:
        for field_name in fields_list:
            form_field = form.fields.get(field_name)
            if form_field:
                choices = form_field._choices
                if choices:
                    for choice in choices:
                        result.append(choice[0])
        result = list(set(result))
    return result


def get_current_mkt_val(value, required_field):
    try:
        if not value.empty:
            index = value.index
            if not index.empty:
                index = index[0]
                return value.at[index, required_field]
        else:
            return 0
    except Exception as e:
        return 0.0


def get_tabs_data(field_names_list, risk_form):
    tab_data = {}
    option_list = get_select_choices_list(field_names_list, risk_form)
    if option_list:
        option_list = sorted(option_list, reverse=True)
        for option in option_list:
            if option:
                tab_data[option] = []
    return tab_data


def get_rows_data(request):
    data = request.POST
    option_data = []
    if data:
        filters = data.get('filters')
        if filters:
            filters = filters.split(",")
            fund = filters[0]
            impacts_df = pd.DataFrame.from_records(DailyNAVImpacts.objects.all().values())
            wic_flat_file_df = pd.read_sql_query('SELECT * FROM wic.daily_flat_file_db where flat_file_as_of = ' \
                                                 '(select max(flat_file_as_of) from wic.daily_flat_file_db) and ' \
                                                 'Sleeve = "Merger Arbitrage" and Fund="' + fund + '";', con=connection)
            first_filter = filters[1] if len(filters) > 1 else ''
            second_filter = filters[2] if len(filters) > 2 else ''
            third_filter = filters[3] if len(filters) > 3 else ''
            if first_filter and first_filter.lower() == 'regulatory risk':
                regulatory_field = REGULATORY_TAB_DICT.get(second_filter)
            else:
                regulatory_field = MAIN_TABS_DICT.get(first_filter)
                regulatory_field = [regulatory_field] if not isinstance(regulatory_field, (list)) else regulatory_field
            if third_filter:
                actual_clearance = regulatory_field.replace('requirement', 'actual_clearance')
                custom_filters = {regulatory_field: third_filter}
                regulatory_field = [regulatory_field] if not isinstance(regulatory_field, (list)) else regulatory_field
                risk_factors_df = pd.DataFrame.from_records(MA_Deals_Risk_Factors.objects.filter(**custom_filters).values())
            else:
                custom_filters = {k:second_filter for k in regulatory_field}
                risk_factors_df = pd.DataFrame.from_records(MA_Deals_Risk_Factors.objects.filter((reduce(operator.or_,
                                (Q(**d) for d in [dict([i]) for i in custom_filters.items()])))).values())
            ma_deals_df = pd.DataFrame.from_records(MA_Deals.objects.filter(archived=False).values('id', 'deal_name', 'status'))
            if not ma_deals_df.empty and not risk_factors_df.empty:
                merge_df = pd.merge(risk_factors_df, ma_deals_df, left_on=['deal_id'], right_on=['id'])
                merge_df.rename(columns={'id_x': 'id', 'id_y': 'deal_id'}, inplace=True)
                filtered_df = merge_df[merge_df[actual_clearance].isna()] if third_filter else merge_df
                unique_deal_names = get_unique_deal_names(filtered_df, regulatory_field, third_filter or second_filter)
                option_data = []
                for deal_name in unique_deal_names:
                    deal_name = deal_name.upper()
                    deal_data = {}
                    deal_df = wic_flat_file_df[wic_flat_file_df['TradeGroup'].str.upper() == deal_name]
                    if not deal_df.empty:
                        deal_df = deal_df.groupby(['AlphaHedge']).agg('sum').reset_index()
                        alpha_mkt_val = get_current_mkt_val(deal_df[deal_df['AlphaHedge'].str.upper() == 'ALPHA'], 'CurrentMktVal')
                        alphahedge_mkt_val = get_current_mkt_val(deal_df[deal_df['AlphaHedge'].str.upper() == 'ALPHA HEDGE'], 'CurrentMktVal')
                        hedge_mkt_val = get_current_mkt_val(deal_df[deal_df['AlphaHedge'].str.upper() == 'HEDGE'], 'CurrentMktVal')
                        alpha_alphahedge_val = alpha_mkt_val + alphahedge_mkt_val
                        deal_mkt_val = hedge_mkt_val + alpha_alphahedge_val
                        deal_status = 'UNKNOWN'
                        deal_status_index = merge_df[merge_df['deal_name'].str.upper() == deal_name].index
                        if not deal_status_index.empty:
                            deal_status_index = deal_status_index[0]
                            deal_status = merge_df.loc[deal_status_index, 'status']
                        impacts_df_index = impacts_df[impacts_df['TradeGroup'].str.upper() == deal_name].index
                        if not impacts_df_index.empty:
                            impacts_df_index = impacts_df_index[0]
                            try:
                                outlier_nav_impact = float(impacts_df.at[impacts_df_index, 'OUTLIER_NAV_IMPACT_' + fund])
                            except ValueError:
                                outlier_nav_impact = 0.0
                        else:
                            outlier_nav_impact = 0.0
                        deal_data = {'deal_name': deal_name, 'alpha_mkt_val': alpha_mkt_val,
                                     'alphahedge_mkt_val': alphahedge_mkt_val, 'hedge_mkt_val': hedge_mkt_val,
                                     'alpha_alphahedge_val': alpha_alphahedge_val, 'deal_mkt_val': deal_mkt_val,
                                     'outlier_nav_impact': outlier_nav_impact, 'deal_status': deal_status}
                        option_data.append(deal_data)
    return JsonResponse({'data': option_data})
