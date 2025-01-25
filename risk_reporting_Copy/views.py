import datetime
import json
import traceback

from django.views import View

import holiday_utils
from risk.tasks import get_peerdownside_by_tradegroup

try:
    from io import BytesIO as IO  # for modern python
except ImportError:
    from io import StringIO as IO  # for legacy python

import pandas as pd
import numpy as np
from django.conf import settings
from django.db import close_old_connections, connection
from django.db.models import Max
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django_slack import slack_message
from django.views.generic import ListView, TemplateView
from django.views.generic.edit import FormView
import dfutils
import bbgclient
import requests
from realtime_pnl_impacts import utils
from risk.models import MA_Deals, MA_Deals_Risk_Factors, Downside_Trendlines, MA_Deals_Approvals, MaDealsActionIdDetails
from risk_reporting.forms import FormulaDownsideForm
from risk_reporting.models import (ArbNAVImpacts, CreditDealsUpsideDownside, DailyNAVImpacts, FormulaeBasedDownsides,
                                   PositionLevelNAVImpacts, HistoricalFormulaeBasedDownsides, FormulaeBasedDownsides)
from risk_reporting import risk_factors_summary
from slack_utils import get_channel_name, get_ip_addr
from sqlalchemy import create_engine
from .ess_utils import get_ess_nav_impacts
from datetime import timedelta, date
from exposures.models import ExposuresSnapshot


def calculate_pl_base_case(row):
    """ Calculates the PL Base Case based on Security Type """

    x = 0
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
    return (-row['CurrMktVal'] * row['FxFactor']) + x


def calculate_base_case_nav_impact(row):
    """ Function to calculate Baase case NAV Impact. PL_BASE_CASE should be calculated first """
    return (row['PL_BASE_CASE'] / row['NAV']) * 100


def calculate_outlier_pl(row):
    """ Calculates Outlier PL or Outlier BASE_CASE """
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

    return (-row['CurrMktVal'] * row['FxFactor']) + x


def calculate_outlier_nav_impact(row):
    """ Calculates the Outlier NAV Impact based on OUTLIER_PL"""
    return (row['OUTLIER_PL'] / row['NAV']) * 100


def get_deal_info_dataframe():
    # DealInfo.csv @ the Deal Level
    deal_level = pd.DataFrame.from_records(FormulaeBasedDownsides.objects.filter(IsExcluded__contains='No',
                                                                                 RiskLimit__isnull=False)
                                           .values('TradeGroup', 'RiskLimit').distinct())

    deal_level.rename(columns={'RiskLimit': 'Risk Limit', 'TradeGroup': 'Deal'}, inplace=True)
    # Add extra columns
    deal_level['Analyst'] = ''
    deal_level['BBG Event ID'] = ''
    deal_level['Catalyst Rating'] = ''
    deal_level['Closing Date'] = ''
    deal_level['Deal Cash Terms'] = ''
    deal_level['Deal Downside'] = ''
    deal_level['Deal Stock Terms'] = ''
    deal_level['Deal Upside'] = ''
    deal_level['Expected Acquirer Dividends'] = ''
    deal_level['Expected Target Dividends'] = ''
    deal_level['Number of Acquirer Dividends'] = ''
    deal_level['Number of Target Dividends'] = ''
    deal_level['Strategy Type'] = ''
    deal_level['Tradegroup Bucket'] = ''
    deal_level['AED Weight'] = ''
    deal_level['ARB Weight'] = ''
    deal_level['CAM Weight'] = ''
    deal_level['LEV Weight'] = ''
    deal_level['LG Weight'] = ''
    deal_level['TACO Weight'] = ''
    deal_level['TAQ Weight'] = ''
    deal_level['WED Weight'] = ''
    deal_level['WIC Weight'] = ''
    deal_level['MACO Weight'] = ''
    deal_level['MALT Weight'] = ''
    deal_level['Catalyst Type'] = ''
    deal_level['Pre Weight'] = ''
    deal_level['EVNT Weight'] = ''

    deal_level = deal_level[['Deal', 'Analyst', 'BBG Event ID', 'Catalyst Rating', 'Closing Date', 'Deal Cash Terms',
                             'Deal Downside', 'Deal Stock Terms', 'Deal Upside', 'Expected Acquirer Dividends',
                             'Expected Target Dividends', 'Number of Acquirer Dividends', 'Number of Target Dividends',
                             'Risk Limit', 'Strategy Type', 'Tradegroup Bucket', 'AED Weight', 'ARB Weight',
                             'CAM Weight',
                             'LEV Weight', 'LG Weight', 'TACO Weight', 'TAQ Weight', 'WED Weight', 'WIC Weight',
                             'MACO Weight', 'MALT Weight', 'Catalyst Type', 'Pre Weight', 'EVNT Weight']]

    # Add % sign to Risk Limit
    deal_level['Risk Limit'] = deal_level['Risk Limit'].apply(lambda x: str(x) + "%")
    return deal_level


def get_security_info_dataframe():
    # Get the Security Level files in the required format.
    position_level = pd.DataFrame.from_records(
        FormulaeBasedDownsides.objects.filter(IsExcluded__contains='No', base_case__isnull=False,
                                              outlier__isnull=False).
        values('TradeGroup', 'Underlying', 'outlier', 'base_case'))
    # Rename columns
    position_level.rename(columns={'TradeGroup': 'Deal', 'Underlying': 'Security', 'outlier': 'Outliers',
                                   'base_case': 'PM Base Case'}, inplace=True)

    position_level['Security'] = position_level['Security'].apply(lambda x: ' '.join(x.split(' ')[0:2]))
    # Add the other required columns
    position_level['Alternate Ticker'] = ''
    position_level['Rebate Rate'] = ''
    position_level['Price'] = ''
    position_level['Adj_CR_01'] = ''
    position_level['CR_01'] = ''
    position_level['DV01'] = ''
    position_level['Beta'] = ''

    # Rearrange columns
    position_level = position_level[['Deal', 'Security', 'Alternate Ticker', 'Outliers', 'PM Base Case', 'Rebate Rate',
                                     'Price', 'Adj_CR_01', 'CR_01', 'DV01', 'Beta']]
    # This should be named SecurityInfo.csv
    return position_level


def deal_info_download(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=DealInfo.csv'

    deal_level = get_deal_info_dataframe()
    deal_level.to_csv(path_or_buf=response, index=False)

    return response


def security_info_download(request):
    position_level = get_security_info_dataframe()
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=SecurityInfo.csv'
    position_level.to_csv(path_or_buf=response, index=False)
    return response


def formula_based_downsides_download(request):
    all_downsides = pd.DataFrame.from_records(FormulaeBasedDownsides.objects.all().values())
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=FormulaBasedDownsides.csv'
    all_downsides.to_csv(path_or_buf=response, index=False)
    return response


def download_nav_data(request):
    formula_df = pd.DataFrame.from_records(FormulaeBasedDownsides.objects.filter(IsExcluded='No').values())
    formula_df = formula_df[~formula_df['Underlying'].str.upper().str.contains('CVR EQUITY')]
    daily_nav_df = pd.DataFrame.from_records(DailyNAVImpacts.objects.all().values())
    query = 'Select tradegroup, ytd_dollar from ' + settings.CURRENT_DATABASE + \
            '.funds_snapshot_tradegroupperformancefundnavdollar where fund = "ARB" and date = (Select max(date) from ' + \
            settings.CURRENT_DATABASE + '.funds_snapshot_tradegroupperformancefundnavdollar);'
    ytd_df = pd.read_sql_query(query, con=connection)
    ytd_df.rename(columns={'tradegroup': 'TradeGroup', 'ytd_dollar': 'YTD_Dollar'}, inplace=True)
    target_df = formula_df[formula_df['TargetAcquirer'] == 'Target']
    acq_df = formula_df[formula_df['TargetAcquirer'] == 'Acquirer']
    merge_df = pd.merge(target_df, acq_df, on='TradeGroup', how='left')
    merge_df = merge_df[['TradeGroup', 'base_case_x', 'base_case_y', 'outlier_y']]
    merge_df = pd.merge(merge_df, daily_nav_df[['TradeGroup', 'BASE_CASE_NAV_IMPACT_ARB', 'OUTLIER_NAV_IMPACT_ARB']],
                        on='TradeGroup', how='left')
    merge_df = pd.merge(merge_df, ytd_df, on='TradeGroup', how='left')
    merge_df = merge_df.replace(['None', None], ['', ''])

    def convert_to_float(value):
        try:
            if value:
                value = str(value).replace(',', '')
                value = float(value)
                return value
        except ValueError:
            return value
        return value

    merge_df['base_case_x'] = merge_df['base_case_x'].apply(convert_to_float)
    merge_df['base_case_y'] = merge_df['base_case_y'].apply(convert_to_float)
    merge_df['outlier_y'] = merge_df['outlier_y'].apply(convert_to_float)
    merge_df['BASE_CASE_NAV_IMPACT_ARB'] = merge_df['BASE_CASE_NAV_IMPACT_ARB'].apply(convert_to_float)
    merge_df['OUTLIER_NAV_IMPACT_ARB'] = merge_df['OUTLIER_NAV_IMPACT_ARB'].apply(convert_to_float)
    merge_df['YTD_Dollar'] = merge_df['YTD_Dollar'].apply(convert_to_float)
    merge_df['YTD_Dollar'] = pd.to_numeric(merge_df['YTD_Dollar'], errors='coerce')

    merge_df.rename(columns={'base_case_x': 'Target Downside (Base)', 'base_case_y': 'Acq Downside (Base)',
                             'outlier_y': 'Acq Downside (Outlier)', 'YTD_Dollar': 'YTD PnL'}, inplace=True)

    merge_df = merge_df.sort_values(by='YTD PnL')
    excel_file = IO()
    xlwriter = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    merge_df.to_excel(xlwriter, 'NAV Data')
    xlwriter.save()
    xlwriter.close()
    excel_file.seek(0)
    response = HttpResponse(excel_file.read(),
                            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=arb_tradegroup_breakdown.xlsx'
    return response


def merger_arb_risk_attributes(request):

    def get_last_update_downside(row):
        try:
            last_update = forumale_linked_downsides[forumale_linked_downsides['TradeGroup'] == row['TradeGroup']][
                'LastUpdate'].max()
        except:
            last_update = None
        return last_update

    """ View to Populate the Risk attributes for the Arbitrage Fund """
    close_old_connections()
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" +
                           settings.WICFUNDS_DATABASE_PASSWORD + "@" + settings.WICFUNDS_DATABASE_HOST + "/" +
                           settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()
    try:
        ytd_performances = pd.read_sql_query('Select tradegroup AS TradeGroup, fund AS FundCode, ytd_bps AS PnL_BPS '
                                             'from funds_snapshot_tradegroupperformancefundnavbps where date='
                                             '(Select max(date) from funds_snapshot_tradegroupperformancefundnavbps) '
                                             'GROUP BY TradeGroup, FundCode', con=con)

        ytd_performances['PnL_BPS'] = ytd_performances['PnL_BPS'].fillna(0).astype(float) * 0.01
    except:
        ytd_performances = pd.DataFrame()
    finally:
        con.close()

    negative_pnl_accounted = True
    if len(ytd_performances) == 0:
        negative_pnl_accounted = False

    nav_impacts_positions_df = pd.DataFrame.from_records(PositionLevelNAVImpacts.objects.all().values())
    last_calculated_on = nav_impacts_positions_df.CALCULATED_ON.max()

    if not request.is_ajax():
        return render(request, 'risk_attributes.html', context={'negative_pnl_accounted': negative_pnl_accounted,
                                                            'last_calculated_on': last_calculated_on})

    forumale_linked_downsides = pd.read_sql_query(
        'SELECT TradeGroup, Underlying,  base_case,  outlier, LastUpdate, LastPrice'
        ' FROM ' + settings.CURRENT_DATABASE + '.risk_reporting_formulaebaseddownsides',
        con=connection)

    impacts_df = pd.DataFrame.from_records(DailyNAVImpacts.objects.all().values())
    impacts_df['LastUpdate'] = impacts_df.apply(get_last_update_downside, axis=1)

    if not ytd_performances.empty:
        ytd_performances = pd.pivot_table(ytd_performances, index=['TradeGroup'], columns=['FundCode'],
                                          aggfunc=np.sum,
                                          fill_value='')

        ytd_performances.columns = ["_".join((i, j)) for i, j in ytd_performances.columns]
        ytd_performances.reset_index(inplace=True)
        # removed 'PnL_BPS_MACO' from floats in 2024
        floats = ['TradeGroup', 'PnL_BPS_ARB', 'PnL_BPS_MALT', 'PnL_BPS_AED',
                   'PnL_BPS_LG', 'PnL_BPS_PRELUDE', 'PnL_BPS_EVNT']
        ytd_performances = ytd_performances[floats].fillna(0)
        ytd_performances = ytd_performances.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

        return_data = {'data': impacts_df.to_json(orient='records'),
                       'positions': nav_impacts_positions_df.to_json(orient='records'),
                       'ytd_pnl': ytd_performances.to_json(orient='records')}
        return HttpResponse(json.dumps(return_data), content_type='application/json')

    else:
        return HttpResponse(json.dumps({'error': 'ytd_performances empty'}), content_type='application/json')



def arb_sector_concentration_trend(request):
    try:
        response = {'error': True}
        impacts_df = pd.DataFrame.from_records(DailyNAVImpacts.objects.all().values())
        required_funds = ['ARB', 'AED', 'LEV']
        col_prefixes = ['BASE_CASE_NAV_IMPACT_', 'OUTLIER_NAV_IMPACT_']
        req_cols = ['TradeGroup', 'RiskLimit']
        for col_prefix in col_prefixes:
            for fund in required_funds:
                req_col = col_prefix + fund
                req_cols.append(req_col)
                impacts_df[req_col] = impacts_df[req_col].fillna(0).replace("", 0).astype(float).round(3)
        impacts_df = impacts_df[req_cols].reset_index(drop=True)
        result_df = pd.DataFrame(columns=['TradeGroup', 'RiskLimit'])

        for fund in required_funds:
            temp_req_cols = ['TradeGroup', 'RiskLimit', 'BASE_CASE_NAV_IMPACT_' + fund, 'OUTLIER_NAV_IMPACT_' + fund]
            temp_df = impacts_df.nsmallest(10, 'OUTLIER_NAV_IMPACT_' + fund)[temp_req_cols]
            result_df = pd.merge(result_df, temp_df, on=['TradeGroup', 'RiskLimit'], how='outer')
        result_df = result_df[req_cols].fillna(0)

        # Todo Implement efficiently
        try:
            for index, row in result_df.iterrows():
                for f in required_funds:
                    base = 'BASE_CASE_NAV_IMPACT_' + f
                    out = 'OUTLIER_NAV_IMPACT_' + f
                    if row[base] == 0:
                        b_value = impacts_df[impacts_df['TradeGroup'] == row['TradeGroup']].iloc[0][base]
                        result_df.loc[result_df['TradeGroup'] == row['TradeGroup'], base] = b_value

                    if row[out] == 0:
                        out_value = impacts_df[impacts_df['TradeGroup'] == row['TradeGroup']].iloc[0][out]
                        result_df.loc[result_df['TradeGroup'] == row['TradeGroup'], out] = out_value
        except Exception as e:
            print(e)
            print(row)
        response = {'result_df': result_df.to_json(orient='records'), 'error': False}
    except Exception:
        response = {'error': True}
    return JsonResponse(response)


# The following should run in a scheduled job. Over here just get values from DB and render to the Front end...


def formulae_downsides_new_deal_add(request):
    """ Add new deal to formulae based downsides page """
    response = 'Failed'
    if request.method == 'POST':
        # Get the Data
        tradegroup = request.POST['tradegroup']
        underlying_security = request.POST['underlying_security']
        analyst = request.POST['analyst']
        origination_date = request.POST['origination_date']
        deal_value = request.POST['deal_value']
        position_in_acquirer = request.POST['position_in_acquirer']
        acquirer_security = request.POST['acquirer_security']
        risk_limit = request.POST['risk_limit']

        # Get the max ID
        try:
            max_id = int(FormulaeBasedDownsides.objects.all().aggregate(Max('id'))['id__max'])
            insert_id = max_id + 1
            obj = FormulaeBasedDownsides()
            obj.id = insert_id
            obj.TradeGroup = tradegroup
            obj.Underlying = underlying_security
            obj.TargetAcquirer = 'Target'
            obj.Analyst = analyst
            obj.RiskLimit = risk_limit
            obj.OriginationDate = origination_date
            obj.DealValue = deal_value
            obj.save()

            # If Position in Acquirer is Yes then create another row
            if position_in_acquirer == 'Yes':
                obj2 = FormulaeBasedDownsides()
                obj2.id = insert_id + 1
                obj2.TradeGroup = tradegroup
                obj2.Underlying = acquirer_security
                obj2.TargetAcquirer = 'Acquirer'
                obj2.Analyst = analyst
                obj2.RiskLimit = risk_limit
                obj2.OriginationDate = origination_date
                obj2.DealValue = deal_value
                obj2.save()

            response = 'Success'

        except Exception as e:
            response = 'Failed'
            print(e)
    return HttpResponse(response)


def merger_arb_nav_impacts(request):
    """ Render the NAV Imacts on Merger Arb """
    # Get the Dataframe from models
    nav_impacts_positions_df = pd.DataFrame.from_records(ArbNAVImpacts.objects.all().values())
    nav_impacts_positions_df['CurrMktVal'] = nav_impacts_positions_df['QTY'] * nav_impacts_positions_df['LastPrice']
    float_cols = ['DealTermsCash', 'DealTermsStock', 'DealValue', 'NetMktVal', 'FxFactor', 'Capital',
                  'BaseCaseNavImpact', 'RiskLimit',
                  'OutlierNavImpact', 'QTY', 'NAV', 'PM_BASE_CASE', 'Outlier', 'StrikePrice', 'LastPrice']
    nav_impacts_positions_df[float_cols] = nav_impacts_positions_df[float_cols].astype(float)
    # Calculate the Impacts
    nav_impacts_positions_df['PL_BASE_CASE'] = nav_impacts_positions_df.apply(calculate_pl_base_case, axis=1)
    nav_impacts_positions_df['BASE_CASE_NAV_IMPACT'] = nav_impacts_positions_df.apply(calculate_base_case_nav_impact,
                                                                                      axis=1)
    # Calculate Outlier Impacts
    nav_impacts_positions_df['OUTLIER_PL'] = nav_impacts_positions_df.apply(calculate_outlier_pl, axis=1)
    nav_impacts_positions_df['OUTLIER_NAV_IMPACT'] = nav_impacts_positions_df.apply(calculate_outlier_nav_impact,
                                                                                    axis=1)
    nav_impacts_positions_df.rename(columns={'TG': 'TradeGroup'}, inplace=True)  # Rename to TradeGroup
    # Sum Impacts of Individual Securities for Impacts @ TradeGroup level...
    nav_impacts_positions_df = nav_impacts_positions_df.round({'BASE_CASE_NAV_IMPACT': 2, 'OUTLIER_NAV_IMPACT': 2})
    nav_impacts_sum_df = nav_impacts_positions_df.groupby(['TradeGroup', 'FundCode', 'PM_BASE_CASE', 'RiskLimit']).agg(
        {'BASE_CASE_NAV_IMPACT': 'sum', 'OUTLIER_NAV_IMPACT': 'sum'})

    nav_impacts_sum_df = pd.pivot_table(nav_impacts_sum_df, index=['TradeGroup', 'RiskLimit'], columns='FundCode',
                                        fill_value='N/A')

    nav_impacts_sum_df.columns = ["_".join((i, j)) for i, j in nav_impacts_sum_df.columns]
    nav_impacts_sum_df.reset_index(inplace=True)

    nav_impacts_sum_df.to_sql(con=settings.SQLALCHEMY_CONNECTION, if_exists='append', index=False,
                              name='risk_reporting_dailynavimpacts',
                              schema=settings.CURRENT_DATABASE)
    return render(request, 'merger_arb_nav_impacts.html', context={'impacts':
                                                                       nav_impacts_sum_df.to_json(orient='index')})


class FormulaDownsideView(FormView):
    """ This View should return the positions from FormulaDownside Models with ability to update
    the calulation fields for each deal at security level """
    # Gather data from Model and send to front end..Listen for any updates
    template_name = 'downside_fomulae.html'
    form_class = FormulaDownsideForm
    fields = '__all__'
    success_url = '#'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['marb_positions'] = FormulaeBasedDownsides.objects.all()
        return context

    def form_valid(self, form):
        form_data = form.cleaned_data
        file_df = form_data.get('file_df')
        formula_df = pd.DataFrame.from_records(FormulaeBasedDownsides.objects.all().values())
        merge_df = pd.merge(formula_df, file_df, how='left', on=['TradeGroup', 'Underlying'])
        if len(formula_df) != len(merge_df) or merge_df.empty:
            form.add_error('file', 'Duplicate rows while merging. Downside values not updated.')
            return super(FormulaDownsideView, self).form_invalid(form)
        merge_df['Base Case'] = merge_df['Base Case'].fillna(merge_df['base_case'])
        merge_df['Outlier'] = merge_df['Outlier'].fillna(merge_df['outlier'])
        merge_df.drop(columns=['base_case', 'outlier'], inplace=True)
        merge_df.rename(columns={'Base Case': 'base_case', 'Outlier': 'outlier'}, inplace=True)
        engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                               + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
        con = engine.connect()
        try:
            con.execute('SET FOREIGN_KEY_CHECKS=0;TRUNCATE TABLE ' + settings.CURRENT_DATABASE +
                        '.risk_reporting_formulaebaseddownsides')
            merge_df.to_sql(name='risk_reporting_formulaebaseddownsides', con=con, if_exists='append', index=False,
                            schema=settings.CURRENT_DATABASE)
        except Exception as e:
            FormulaeBasedDownsides.objects.all().delete()
            formula_df.to_sql(name='risk_reporting_formulaebaseddownsides', con=con, if_exists='append', index=False,
                              schema=settings.CURRENT_DATABASE)
        finally:
            con.close()
        return super(FormulaDownsideView, self).form_valid(form)


def update_risk_limit(request):
    """
    View for updating risk limit
    """
    if request.method == 'POST':
        risk_limit = request.POST.get('risk_limit')
        row_id = request.POST.get('id')
        try:
            obj = FormulaeBasedDownsides.objects.get(id=row_id)
            old_risk_limit = obj.RiskLimit
            deal_name = obj.TradeGroup
            matching_tradegroups = FormulaeBasedDownsides.objects.filter(TradeGroup__exact=deal_name)
            for deals in matching_tradegroups:
                deals.RiskLimit = risk_limit
                deals.save()
            response = 'Success'

            ip_addr = get_ip_addr(request)
            slack_message('portal_risk_limit_update.slack',
                          {'updated_deal': str(obj.TradeGroup),
                           'risk_limit': str(old_risk_limit) + " -> " + str(risk_limit),
                           'IP': str(ip_addr)},
                          channel=get_channel_name('portal_downsides'),
                          token=settings.SLACK_TOKEN,
                          name='PORTAL DOWNSIDE UPDATE AGENT')
        except FormulaeBasedDownsides.DoesNotExist:
            response = 'Failed'
    return HttpResponse(response)


def update_downside_formulae(request):
    """ View to Update the downside formulae for each position """
    # Only process POST requests
    response = 'Failed'
    if request.method == 'POST':
        if request.POST.get('update_risk_limit') == 'true':
            risk_limit = request.POST.get('risk_limit')
            row_id = request.POST.get('id')
            try:
                # Retroactively update Risk Limit for all matching TradeGroups
                obj = FormulaeBasedDownsides.objects.get(id=row_id)
                original_risk_limit = obj.RiskLimit
                if str(original_risk_limit) != str(risk_limit):
                    return JsonResponse({'msg': 'Risk Limit Different', 'original_risk_limit': original_risk_limit})
                else:
                    response = 'Risk Limit Same'
            except FormulaeBasedDownsides.DoesNotExist:
                response = 'Failed'
        else:
            # Gather the data
            try:
                row_id = request.POST['id']
                is_excluded = request.POST['is_excluded']
                base_case_downside_type = request.POST['base_case_downside_type'] or None
                base_case_reference_data_point = request.POST['base_case_reference_data_point'] or None
                base_case_reference_price = request.POST['base_case_reference_price'] or None
                base_case_operation = request.POST['base_case_operation'] or None
                base_case_custom_input = request.POST['base_case_custom_input'] or None
                base_case = request.POST['base_case'] or None
                base_case_notes = request.POST['base_case_notes'] or None
                day_1_downside = request.POST['day_1_basecase'] or None
                cix_ticker = request.POST['cix_ticker'] or None
                outlier_downside_type = request.POST['outlier_downside_type'] or None
                outlier_reference_data_point = request.POST['outlier_reference_data_point'] or None
                outlier_reference_price = request.POST['outlier_reference_price'] or None
                outlier_operation = request.POST['outlier_operation'] or None
                outlier_custom_input = request.POST['outlier_custom_input'] or None
                outlier = request.POST['outlier'] or None
                outlier_notes = request.POST['outlier_notes'] or None
                backup_bid = request.POST['backup_bid'] or None

                if not outlier:
                    # Outlier should match base case by Default
                    outlier_downside_type = base_case_downside_type
                    outlier_reference_data_point = base_case_reference_data_point
                    outlier_reference_price = base_case_reference_price
                    outlier_operation = base_case_operation
                    outlier_custom_input = base_case_custom_input
                    outlier = base_case

                obj = FormulaeBasedDownsides.objects.get(id=row_id)
                old_base_case_downside = obj.base_case
                old_outlier = obj.outlier
                obj.IsExcluded = is_excluded
                obj.BaseCaseDownsideType = base_case_downside_type
                obj.BaseCaseReferenceDataPoint = base_case_reference_data_point
                obj.cix_ticker = cix_ticker
                obj.BaseCaseReferencePrice = base_case_reference_price
                obj.BaseCaseOperation = base_case_operation
                obj.BaseCaseCustomInput = base_case_custom_input
                obj.base_case = base_case
                obj.base_case_notes = base_case_notes
                obj.OutlierDownsideType = outlier_downside_type
                obj.OutlierReferenceDataPoint = outlier_reference_data_point
                obj.OutlierReferencePrice = outlier_reference_price
                obj.day_one_downside = day_1_downside
                obj.backup_bid = backup_bid

                if outlier_operation in ['+', '-', "*", '/']:  # guarantees mathmatic operators
                    obj.OutlierOperation = outlier_operation
                else:
                    obj.OutlierOperation = None

                try:  # guarantees numeric input
                    obj.OutlierCustomInput = str(float(outlier_custom_input))
                except ValueError:
                    obj.OutlierCustomInput = None
                except TypeError:
                    obj.OutlierCustomInput = None

                obj.outlier = outlier
                obj.outlier_notes = outlier_notes
                obj.LastUpdate = datetime.datetime.now()
                obj.save()

                # Update the target_downside / acquirer_upside in MA_Deals
                target_acquirer = obj.TargetAcquirer
                if outlier and outlier != 'None' and target_acquirer and target_acquirer.lower() == 'target':
                    underlying = obj.Underlying
                    underlying = underlying.lower().replace("equity", "").strip() if underlying else underlying
                    ma_deal = MA_Deals.objects.filter(deal_name=obj.TradeGroup,
                                                      target_ticker__icontains=underlying).first()
                    if ma_deal:
                        ma_deal.target_downside = outlier
                        ma_deal.last_downside_update = datetime.datetime.now().date()
                        ma_deal.save()
                elif outlier and outlier != 'None' and target_acquirer and target_acquirer.lower() == 'acquirer':
                    ma_deal = MA_Deals.objects.filter(deal_name=obj.TradeGroup).first()
                    if ma_deal:
                        ma_deal.acquirer_upside = outlier
                        ma_deal.last_downside_update = datetime.datetime.now().date()
                        ma_deal.save()
                response = 'Success'
                ip_addr = get_ip_addr(request)
                slack_message('portal_downsides.slack',
                              {'updated_deal': str(obj.TradeGroup), 'underlying_security': obj.Underlying,
                               'base_case': str(old_base_case_downside) + " -> " + str(obj.base_case),
                               'outlier': str(old_outlier) + " -> " + str(obj.outlier), 'IP': str(ip_addr)},
                              channel=get_channel_name('portal_downsides'), token=settings.SLACK_TOKEN,
                              name='PORTAL DOWNSIDE UPDATE AGENT')
            except Exception as e:
                import traceback;
                traceback.print_exc()
                print(e)
                response = 'Failed'

    return HttpResponse(response)


class CreditDealsUpsideDownsideView(ListView):
    """
    View for Credit Deals Upside Downside Page
    """
    template_name = 'credit_deals_upside_downside.html'
    model = CreditDealsUpsideDownside
    queryset = CreditDealsUpsideDownside.objects.all().order_by('downside', 'upside', '-last_updated',
                                                                '-origination_date')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        try:
            context['as_of'] = CreditDealsUpsideDownside.objects.all().latest('last_refreshed').last_refreshed
        except CreditDealsUpsideDownside.DoesNotExist:
            context['as_of'] = 'Unknown'
        return context


def update_credit_deals_upside_downside(request):
    """ View to Update the Upside/Downside for Credit Deals """
    # Only process POST requests
    response = 'Failed'
    if request.method == 'POST':
        if request.POST.get('update_risk_limit') == 'true':
            risk_limit = request.POST.get('risk_limit')
            row_id = request.POST.get('id')
            try:
                obj = CreditDealsUpsideDownside.objects.get(id=row_id)
                original_risk_limit = obj.risk_limit
                if str(original_risk_limit) != str(risk_limit):
                    return JsonResponse({'msg': 'Risk Limit Different', 'original_risk_limit': original_risk_limit})
                else:
                    response = 'Risk Limit Same'
            except CreditDealsUpsideDownside.DoesNotExist:
                response = 'Failed'
        else:
            try:
                row_id = request.POST['id']
                spread_index = request.POST['spread_index']
                is_excluded = request.POST['is_excluded']
                downside_type = request.POST['downside_type']
                downside = request.POST['downside'] or None
                downside_notes = request.POST['downside_notes']
                upside_type = request.POST['upside_type']
                upside = request.POST['upside'] or None
                upside_notes = request.POST['upside_notes']

                obj = CreditDealsUpsideDownside.objects.get(id=row_id)
                old_downside = obj.downside
                old_upside = obj.upside
                obj.spread_index = spread_index
                obj.is_excluded = is_excluded
                obj.downside_type = downside_type
                obj.downside = downside
                obj.downside_notes = downside_notes
                obj.upside_type = upside_type
                obj.upside = upside
                obj.upside_notes = upside_notes
                obj.last_updated = datetime.datetime.now()
                obj.save()
                response = 'Success'

                ip_addr = get_ip_addr(request)
                slack_message('credit_deal_upside_downsides.slack',
                              {'updated_deal': str(obj.tradegroup),
                               'ticker': obj.ticker,
                               'downside': str(old_downside) + " -> " + str(obj.downside),
                               'upside': str(old_upside) + " -> " + str(obj.upside),
                               'IP': str(ip_addr)},
                              channel=get_channel_name('portal_downsides'),
                              token=settings.SLACK_TOKEN,
                              name='PORTAL DOWNSIDE UPDATE AGENT')
            except Exception as e:
                print(e)
                response = 'Failed'

    return HttpResponse(response)


def update_credit_deal_risk_limit(request):
    """
    View for updating risk limit for Credit Deal Upside/Downside
    """
    if request.method == 'POST':
        risk_limit = request.POST.get('risk_limit')
        row_id = request.POST.get('id')
        try:
            obj = CreditDealsUpsideDownside.objects.get(id=row_id)
            old_risk_limit = obj.risk_limit
            obj.risk_limit = risk_limit
            obj.save()
            response = 'Success'
            ip_addr = get_ip_addr(request)
            slack_message('portal_risk_limit_update.slack',
                          {'updated_deal': str(obj.tradegroup),
                           'risk_limit': str(old_risk_limit) + " -> " + str(risk_limit),
                           'IP': str(ip_addr)},
                          channel=get_channel_name('portal_downsides'),
                          token=settings.SLACK_TOKEN,
                          name='PORTAL DOWNSIDE UPDATE AGENT')
        except FormulaeBasedDownsides.DoesNotExist:
            response = 'Failed'
    return HttpResponse(response)


def get_details_from_arb(request):
    response = {'msg': 'Failed'}
    if request.method == 'POST':
        try:
            ticker = request.POST.get('ticker')
            if 'equity' not in ticker.lower():
                ticker = ticker.upper() + ' EQUITY'
            object_list = FormulaeBasedDownsides.objects.filter(Underlying=ticker)
            if object_list:
                obj = object_list.first()
                deal_value = obj.DealValue
                outlier = obj.outlier
                return JsonResponse({'msg': 'Success', 'deal_value': deal_value, 'outlier': outlier})
            return JsonResponse({'msg': 'Not Found'})
        except Exception as e:
            return JsonResponse(response)
    return JsonResponse(response)


def fetch_from_bloomberg_by_spread_index(request):
    """
    Fetch data from Bloomberg API by using Spread Index
    """
    response = {'msg': 'Failed'}
    if request.method == 'POST':
        spread_index = request.POST.get('spread_index')
        if spread_index:
            try:
                api_host = bbgclient.bbgclient.get_next_available_host()
                data = bbgclient.bbgclient.get_secid2field([spread_index], 'tickers', ['PX_LAST'],
                                                           req_type='refdata', api_host=api_host)
                if data and data.get(spread_index) and data.get(spread_index).get('PX_LAST'):
                    px_last = data[spread_index]['PX_LAST']
                    if len(px_last) > 0:
                        px_last = float(px_last[0])
                        response = {'msg': 'Success', 'px_last': px_last}
            except Exception as e:
                return JsonResponse(response)
    return JsonResponse(response)


def fetch_peer_index_by_tradegroup(request):
    """
        Endpoint for frontend to fetch equal weighted peer index value by tradegroup and proxy_name
    """
    if request.method == "POST":
        tradegroup = request.POST.get("tradegroup")
        underlying = request.POST.get("underlying")
        proxy_name = request.POST.get("proxy_name")
        if tradegroup:
            try:
                peer_value, peer_name = get_peerdownside_by_tradegroup(tradegroup, proxy_name, underlying, return_proxy_name=True)
                return JsonResponse({"msg": "Success", "data": {"peer_value": peer_value, "peer_name": peer_name}})
            except Exception as e:
                traceback.print_exc()
                return JsonResponse({"msg": "Failed", "data": traceback.format_exc()})


def get_custom_input(request):
    response = {'msg': 'Failed', 'data': 'Invalid request'}
    if request.method == 'GET':
        deal_name = request.GET.get('tradegroup')
        if deal_name:
            try:
                data = str(get_tradegroup_peer_index_value(deal_name))
                if data:
                    response = {'msg': 'Success', 'data': data}
                else:
                    raise Exception('No data found for ' + deal_name)
            except Exception as e:
                return JsonResponse({'msg': 'Failed', 'data': traceback.format_exc()})
    return JsonResponse(response)


def get_tradegroup_peer_index_value(deal_name):
    """
     Helper function to fetch value for Custom Input when "Peer Index"/"CIX Index" i.s selected as DowndsideType
     in formula_downsides
     It searches for the deal_name in the MA_Deals then find the matching MaDealsActionIdDetails and fetches the
     min value of either unaffected_price or unaffected_downside
     @param deal_name: deal_name to search for in MA_Deals
     @return: min value of either unaffected_price or unaffected_downside of the MaDealsActionIdDetails object
    """
    try:
        action_id = MA_Deals.objects.get(deal_name=deal_name).action_id
        action_id_detail = MaDealsActionIdDetails.objects.filter(action_id=action_id).first()
        data = min(x for x in [action_id_detail.unaffected_price,
                               action_id_detail.unaffected_downside,
                               action_id_detail.unaffected_90d_vwap] if x is not None)
    except Exception as e:
        data = None
    return data


def credit_deals_csv_import(request):
    credit_deals_df = pd.DataFrame.from_records(CreditDealsUpsideDownside.objects.all().values())
    try:
        last_refreshed = CreditDealsUpsideDownside.objects.latest('last_refreshed').last_refreshed.strftime(
            '%B %d %Y %H: %M %p')
    except CreditDealsUpsideDownside.DoesNotExist:
        last_refreshed = 'Unknown'
    credit_deals_df.drop(columns=['last_updated', 'last_refreshed'], inplace=True)
    credit_deals_df.rename(columns={'tradegroup': 'TradeGroup', 'ticker': 'Ticker', 'analyst': 'Analyst',
                                    'origination_date': 'Origination Date', 'spread_index': 'Spread Index',
                                    'deal_value': 'Deal Value', 'last_price': 'Last Price',
                                    'is_excluded': 'Is Excluded',
                                    'risk_limit': 'Risk Limit', 'downside_type': 'Downside Type',
                                    'downside': 'Downside',
                                    'downside_notes': 'Downside Notes', 'upside_type': 'Upside Type',
                                    'upside': 'Upside',
                                    'upside_notes': 'Upside Notes', 'bloomberg_id': 'Bloomberg ID'}, inplace=True)

    credit_deals_df = credit_deals_df[['TradeGroup', 'Ticker', 'Analyst', 'Origination Date', 'Spread Index',
                                       'Deal Value', 'Last Price', 'Is Excluded', 'Risk Limit', 'Downside Type',
                                       'Downside', 'Downside Notes', 'Upside Type', 'Upside', 'Upside Notes',
                                       'Bloomberg ID']]
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=CreditDealsUpDown.csv'
    response.write('Credit Deals Upside Downside\n')
    response.write('Last Refreshed: {0}\n\n'.format(last_refreshed))
    credit_deals_df.to_csv(path_or_buf=response, index=False)
    return response


class RiskFactorsSummaryView(TemplateView):
    template_name = 'risk_factors_summary.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['summary_data'] = json.dumps(risk_factors_summary.get_summary_for_risk_factors())
        return context


class BaseCaseNavByApprovalsView(TemplateView):
    """
    View to display risk report requested by Nikhil.
    """
    template_name = 'basecase_nav_by_approvals.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # requirement columns to swap with frontend keywords
        conversion_sector_dct = {'HSR': 'hsr', 'MOFCOM': 'mofcom', 'CIFIUS': 'cifius', 'EC': 'ec', 'CMA': 'uk_cma',
                                 'Acquirer Vote': 'acq_sh_vote'}
        ##getting the data
        pe_df = pd.DataFrame.from_records(MA_Deals.objects.all().values('id', 'action_id'))  # find private equity deals
        # find matching Strategic PE value
        strategic_pe_df = pd.DataFrame.from_records(
            MaDealsActionIdDetails.objects.all().values('action_id', 'acquirer_industry'))
        # join action_id_df and strategic_pe_df
        pe_df = pe_df.merge(strategic_pe_df, on='action_id', how='left')
        pe_df['deal_id'] = pe_df['id']
        del pe_df['id']

        ma_deals_df = pd.DataFrame.from_records(
            MA_Deals.objects.filter(archived=False).values('id', 'deal_name', 'status'))  # filter(archived=False)
        ma_deals_approvals_df = pd.DataFrame.from_records(
            MA_Deals_Approvals.objects.values())
        DailyNAV_values_df = pd.DataFrame.from_records(
            DailyNAVImpacts.objects.all().values('TradeGroup', 'BASE_CASE_NAV_IMPACT_ARB'))
        DailyNAV_values_df['TradeGroup'] = DailyNAV_values_df['TradeGroup'].str.upper()
        ma_deals_df['deal_name'] = ma_deals_df['deal_name'].str.upper()
        final_df = pd.DataFrame()
        ##merging
        ma_deals_df['deal_id'] = ma_deals_df['id']
        final_df = pd.merge(ma_deals_df, ma_deals_approvals_df, on='deal_id', how='outer')
        final_df = pd.merge(final_df, pe_df, on='deal_id', how='left')
        # final_df.drop('deal_id', axis=1, inplace=True)
        # final_df.rename({'id_y':'deal_id'},axis=1)
        final_df = final_df[final_df['status'].str.upper() == 'ACTIVE']
        final_df = pd.merge(final_df, DailyNAV_values_df, left_on='deal_name', right_on='TradeGroup', how='left')
        final_df.drop('id_y', axis=1, inplace=True)
        final_df.rename({'id_x': 'id'})

        ##filter non-existent rows
        final_df = final_df[(pd.notna(final_df['BASE_CASE_NAV_IMPACT_ARB'])) & (final_df['TradeGroup'])]
        final_df['BASE_CASE_NAV_IMPACT_ARB'] = final_df['BASE_CASE_NAV_IMPACT_ARB'].fillna('').apply(
            lambda v: round(float(v), 3) if str(
                v).lower() != '' else None)  # ['BASE_CASE_NAV_IMPACT_ARB'].astype(float).round(3)

        def get_nav_val(row):
            row['strategic_pe_new'] = np.NaN
            if row['approval_sector'] in conversion_sector_dct.keys():
                val = conversion_sector_dct[row['approval_sector']]
                row[val + '_requirement'] = row['BASE_CASE_NAV_IMPACT_ARB']
            if row['approval_sector'] == 'Shareholder' and row['approval_category'] == 'Acquirer approval':
                row['acq_sh_vote_requirement'] = row['BASE_CASE_NAV_IMPACT_ARB']
            if str(row['acquirer_industry']) == 'Private Equity':
                row['strategic_pe_new'] = row['BASE_CASE_NAV_IMPACT_ARB']
            if row['BASE_CASE_NAV_IMPACT_ARB']:
                row['target_sh_vote_required_percentage'] = row['BASE_CASE_NAV_IMPACT_ARB']
            return row

        final_df = final_df.apply(get_nav_val, axis=1)
        final_df['strategic_pe'] = final_df['strategic_pe_new']
        req_cols = list(conversion_sector_dct.values())
        final_df = final_df.groupby('deal_name').agg("first").reset_index()

        req_cols = [col + '_requirement' for col in req_cols]
        for req in req_cols:
            if req not in final_df.columns.to_list():
                final_df[req] = ''

        final_df = final_df[['deal_name'] + req_cols + ['strategic_pe', 'target_sh_vote_required_percentage']]
        final_df.fillna("", inplace=True)
        context.update({'data': final_df.to_json(orient='records')})

        return context


def get_realtime_downsides(request):
    api_host = bbgclient.bbgclient.get_next_available_host()
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)

    con = engine.connect()
    df = pd.DataFrame()
    try:

        query = "CALL " + settings.CURRENT_DATABASE + ".GET_ARB_DOWNSIDES_DF_TARGET()"
        impacts_query = "SELECT TradeGroup, BASE_CASE_NAV_IMPACT_ARB, OUTLIER_NAV_IMPACT_ARB FROM " + settings.CURRENT_DATABASE \
                        + ".risk_reporting_dailynavimpacts"
        df = pd.read_sql_query(query, con=con)
        df.columns = ['deal_name', 'unaffected_date', 'risk_limit', 'cix_ticker', 'underlying',
                      'original_base_case', 'original_outlier']

        impacts_df = pd.read_sql_query(impacts_query, con=con)
        impacts_df.columns = ['deal_name', 'base_case_nav_impact', 'outlier_nav_impact']
        slicer = dfutils.df_slicer()
        df['unaffected_date'] = df['unaffected_date'].apply(pd.to_datetime)
        df['unaffected_date'] = df['unaffected_date'].apply(
            lambda x: x if slicer.is_business_day(x) else slicer.prev_n_business_days(1, x))
        df = df[~pd.isna(df['cix_ticker'])]
        min_date = df['unaffected_date'].min().strftime('%Y%m%d')
        max_date = datetime.datetime.now().date().strftime('%Y%m%d')
        # Make a historical data request....
        tickers = df['underlying'].unique().tolist() + df['cix_ticker'].unique().tolist()
        historical_prices = requests.get("http://" + api_host + "/wic/api/v1.0/general_histdata",
                                         params={'idtype': "tickers", "fields": "PX_LAST",
                                                 "tickers": ','.join(tickers),
                                                 "override": "",
                                                 "start_date": min_date,
                                                 "end_date": max_date},
                                         timeout=15)  # Set a 15 secs Timeout
        hist_data_results = historical_prices.json()['results']
        final_df = pd.DataFrame(columns=['Date', 'PX_LAST', 'Ticker'])

        for each_dict in hist_data_results:
            for key, value in each_dict.items():
                try:
                    prices = value['fields']['PX_LAST']
                    dates = value['fields']['date']
                    data = list(zip(dates, prices))
                    intermediate_df = pd.DataFrame(columns=['Date', 'PX_LAST'], data=data)
                    intermediate_df['Ticker'] = key
                except KeyError as ke:
                    print(key)
                    intermediate_df = pd.DataFrame(columns=['Date', 'PX_LAST', 'Ticker'])
            final_df = pd.concat([final_df, intermediate_df])
        final_df['PX_LAST'] = final_df['PX_LAST'].apply(float)
        # final_df['Date'] = final_df['Date'].apply(str)

        max_date = final_df['Date'].max()
        price_not_found = []
        df['unaffected_date'] = df['unaffected_date'].apply(lambda x: x.strftime('%Y-%m-%d'))

        def get_price_on_unaffected_date(row, today=False):
            if today:
                unaffected_date = max_date
            else:
                unaffected_date = row['unaffected_date']

            ticker = row['underlying']
            try:
                px = final_df[((final_df['Ticker'] == ticker) & (final_df['Date'] == unaffected_date))].iloc[0][
                    'PX_LAST']
            except IndexError as ie:
                px = 0
                price_not_found.append(ticker)

            return px

        df['Price on Unaffected Date'] = df.apply(lambda x: get_price_on_unaffected_date(x), axis=1)
        df['Price Today'] = df.apply(lambda x: get_price_on_unaffected_date(x, today=True), axis=1)

        # df['Outlier Price on Unaffected Date'] = df.apply(lambda x: get_price_on_unaffected_date(x, outlier=True), axis=1)
        # df['Outlier Price Today'] = df.apply(lambda x: get_price_on_unaffected_date(x, today=True), axis=1)

        def get_cix_price_on_unaffected_date(row, today=False):
            if today:
                unaffected_date = max_date
            else:
                unaffected_date = row['unaffected_date']
            ticker = row['cix_ticker']
            try:
                px = final_df[((final_df['Ticker'] == ticker) & (final_df['Date'] == unaffected_date))].iloc[0][
                    'PX_LAST']
            except IndexError as ie:
                px = 0
                price_not_found.append(ticker)
            return px

        df['CIX Price on Unaffected Date'] = df.apply(lambda x: get_cix_price_on_unaffected_date(x), axis=1)
        df['CIX Price Today'] = df.apply(lambda x: get_cix_price_on_unaffected_date(x, today=True), axis=1)

        df['Needs Attention'] = df.apply(
            lambda x: True if x['Price on Unaffected Date'] == 0 or x['CIX Price on Unaffected Date'] == 0 or x[
                'Price Today'] == 0 or x['CIX Price Today'] == 0 else False, axis=1)
        df['CIX % Change'] = np.round(
            1e2 * (df['CIX Price Today'] - df['CIX Price on Unaffected Date']) / df['CIX Price on Unaffected Date'],
            decimals=2)

        df['CIX Implied Downside'] = np.round(df['Price on Unaffected Date'] * (1 + df['CIX % Change'] / 100),
                                              decimals=3)

        # def get_realtime_outlier_downside(row):
        #     if pd.isna(row['acquirer']):
        #         return row['Realtime Downside']
        #
        #     return np.round(row['Outlier Price Today'] * (1 + row['CIX % Change']/100), decimals=3)
        #
        # df['Realtime Outlier Downside'] = df.apply(lambda x: get_realtime_outlier_downside(x), axis=1)

        # Calculate NAV Risk for Base Case and Outlier
        df = pd.merge(df, impacts_df, how='left', on='deal_name')

        df[['base_case_nav_impact', 'original_base_case', 'original_outlier', 'outlier_nav_impact']] = df[
            ['base_case_nav_impact', 'original_base_case', 'original_outlier', 'outlier_nav_impact']].apply(
            pd.to_numeric)
        df['nav_risk_base_case'] = (df['CIX Implied Downside'] * df['base_case_nav_impact']) / df['original_base_case']
        # df['nav_risk_outlier'] = (df['Realtime Outlier Downside'] * df['outlier_nav_impact']) / df['original_outlier']

        df['nav_risk_base_case'] = df['nav_risk_base_case'].apply(lambda x: np.round(x, decimals=2))

        df.columns = ['deal_name', 'unaffected_date', 'risk_limit', 'cix_ticker', 'underlying',
                      'original_base_case', 'original_outlier', 'price_on_unaffected',
                      'price_today', 'cix_price_on_unaffected', 'cix_price_today', 'needs_attention',
                      'cix_pct_change', 'realtime_downside', 'base_case_nav_impact',
                      'outlier_nav_impact', 'nav_risk_base_case']

    except Exception as e:
        print(e)
    finally:
        con.close()

    return render(request, 'realtime_downsides.html', context={'realtime_downsides': df.to_json(orient='records')})


def get_arb_aed_risk_multiple():
    multiples_query = "SELECT date_updated, tradegroup,1 AS ArbRiskMultiple,  aed_risk_mult FROM " + \
                      settings.CURRENT_DATABASE + ".portfolio_optimization_hardfloatoptimization WHERE " + \
                      "date_updated = (SELECT MAX(date_updated) FROM " + settings.CURRENT_DATABASE + \
                      ".portfolio_optimization_hardfloatoptimization);"

    shares_query = "SELECT tradegroup, target_ticker, TargetLastPrice, AllInSpread, fund, amount, PctOfSleeveCurrent FROM " + \
                   "wic.daily_flat_file_db WHERE Flat_file_as_of = (SELECT MAX(flat_file_as_of) from " + \
                   "wic.daily_flat_file_db) AND AlphaHedge = 'Alpha' AND fund in ('ARB', 'AED') AND Sleeve like " + \
                   "'Merger Arbitrage' AND amount<>0;"

    nav_impacts_query = "SELECT TradeGroup, BASE_CASE_NAV_IMPACT_ARB, OUTLIER_NAV_IMPACT_ARB, " \
                        "BASE_CASE_NAV_IMPACT_AED, OUTLIER_NAV_IMPACT_AED FROM " + settings.CURRENT_DATABASE + \
                        ".risk_reporting_dailynavimpacts"
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()

    multiples_df = pd.read_sql_query(multiples_query, con=con)
    shares_df = pd.read_sql_query(shares_query, con=con)
    nav_impacts_df = pd.read_sql_query(nav_impacts_query, con=con)

    con.close()

    if multiples_df.empty or shares_df.empty or nav_impacts_df.empty:
        return {}

    shares_df.sort_values(by=['tradegroup', 'target_ticker'], inplace=True)
    shares_df = shares_df.groupby(['tradegroup', 'TargetLastPrice', 'AllInSpread', 'fund']).agg(
        {'target_ticker': lambda x: x.iloc[0], 'amount': 'sum', 'PctOfSleeveCurrent': 'sum'}).reset_index()
    shares_df = pd.pivot_table(shares_df, index=['tradegroup', 'target_ticker', 'TargetLastPrice', 'AllInSpread'],
                               columns=['fund'], values=['amount', 'PctOfSleeveCurrent']).reset_index()
    shares_df.columns = ["".join((i, j)) if j else i for i, j in shares_df.columns.values]
    nav_impacts_df.rename(columns={'TradeGroup': 'tradegroup'}, inplace=True)
    final_df = pd.merge(multiples_df, shares_df, how='outer', on='tradegroup')
    final_df = pd.merge(final_df, nav_impacts_df, how='outer', on='tradegroup')
    final_df = final_df[~pd.isna(final_df['ArbRiskMultiple'])]
    calculated_on = final_df['date_updated'].max().strftime('%Y-%m-%d')
    final_df.rename(columns={
        'tradegroup': 'Deal', 'ArbRiskMultiple': 'ARB Risk Multiple', 'aed_risk_mult': 'AED Risk Multiple',
        'TargetLastPrice': 'Last Price (USD)', 'AllInSpread': 'Net Spread (USD)', 'PctOfSleeveCurrentAED': 'AED % AUM',
        'PctOfSleeveCurrentARB': 'ARB % AUM', 'amountAED': 'AED Position', 'amountARB': 'ARB Position',
        'BASE_CASE_NAV_IMPACT_ARB': 'ARB Base-Case NAV-Impact', 'OUTLIER_NAV_IMPACT_ARB': 'ARB Outlier NAV-Impact',
        'BASE_CASE_NAV_IMPACT_AED': 'AED Base-Case NAV-Impact', 'OUTLIER_NAV_IMPACT_AED': 'AED Outlier NAV-Impact',
        'target_ticker': 'Target Ticker'},
        inplace=True)
    final_df = final_df[['Deal', 'Target Ticker', 'Last Price (USD)', 'Net Spread (USD)', 'ARB Position', 'ARB % AUM',
                         'ARB Base-Case NAV-Impact', 'ARB Outlier NAV-Impact', 'ARB Risk Multiple', 'AED Position',
                         'AED % AUM', 'AED Base-Case NAV-Impact', 'AED Outlier NAV-Impact', 'AED Risk Multiple']].copy()
    response = {}
    response['AsOf'] = calculated_on
    response['data'] = final_df.to_json(orient='records')
    return response


class ArbAedRiskMultipleView(TemplateView):
    template_name = 'arb_aed_risk_multiple.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        response = get_arb_aed_risk_multiple()
        context['data'] = response.get('data', json.dumps({}))
        context['as_of'] = response.get('AsOf', '')
        return context


def get_ess_cix_nav_risk_data(request):
    sum_df, details_df = get_ess_nav_impacts()
    return JsonResponse({'data': sum_df.to_json(orient='records'),
                         'details': details_df.to_json(orient='records')})


def peer_downsides(request):
    """
    Page for displaying downsides
    """
    if request.method == 'GET':
        request_latest = True
        latest_date = Downside_Trendlines.objects.latest('date').date
        if 'selected_date' in request.GET:
            selected_date = request.GET['selected_date']
            # request latest data if selected date is later than latest date
            if datetime.datetime.strptime(selected_date, '%Y-%m-%d').date() > latest_date:
                selected_date = latest_date.strftime('%Y-%m-%d')
            else:
                request_latest = False
                trendline_obj = Downside_Trendlines.objects.filter(date__lte=selected_date).latest('date')
                selected_date = trendline_obj.date.strftime('%Y-%m-%d')
        else:
            selected_date = Downside_Trendlines.objects.latest('date').date.strftime('%Y-%m-%d')
        from positions_and_pnl.models import TradeGroupMaster
        try:
            df = pd.DataFrame.from_records(TradeGroupMaster.objects.filter(date=selected_date).values())
            df = df[['fund', 'tradegroup']]

            def etf_filter(x):  # find deals that are only in ARBETF or EVNT
                funds = x['fund'].unique().tolist()
                if 'EVNT' in funds:
                    funds.remove('EVNT')
                if 'ARBETF' in funds:
                    funds.remove('ARBETF')
                return not bool(funds)  # if the deal is only in etf it the list should be empty

            funds_agg = df.groupby('tradegroup').apply(etf_filter).reset_index()
            etf_only_tg = funds_agg[funds_agg[0] == True]['tradegroup'].tolist()

            all_downsides_df = pd.DataFrame.from_records(
                Downside_Trendlines.objects.filter(date=selected_date).values())
            all_downsides_df = all_downsides_df.round(3)
            all_downsides_df['color'] = all_downsides_df['notes'].str.split('|').str[0]
            all_downsides_df['notes'] = all_downsides_df['notes'].str.split('|').str[1]
            all_downsides_df['acquirer_ticker'] = all_downsides_df['tradegroup'].str.split(' - ').str[1]
            all_downsides_df.fillna('', inplace=True)

            # use live downsides if requesting latest data
            if request_latest:
                formula_df = pd.DataFrame.from_records(FormulaeBasedDownsides.objects.filter(IsExcluded='No').values())
                formula_df = formula_df[~formula_df['Underlying'].str.upper().str.contains('CVR EQUITY')]
                target_df = formula_df[formula_df['TargetAcquirer'] == 'Target']
                acq_df = formula_df[formula_df['TargetAcquirer'] == 'Acquirer']
                merge_df = pd.merge(target_df, acq_df, on='TradeGroup', how='left')
                merge_df = merge_df[['TradeGroup', 'base_case_x', 'base_case_y', 'outlier_y']]
                merge_df.rename(columns={'TradeGroup': 'tradegroup',
                                         'base_case_x': 'target_downside_base',
                                         'base_case_y': 'acquirer_downside_base',
                                         'outlier_y': 'acquirer_downside_outlier'}, inplace=True)

                # convert str to float and round to 3 decimals
                merge_df['target_downside_base'] = merge_df['target_downside_base'].astype(float).round(3)
                merge_df['acquirer_downside_base'] = merge_df['acquirer_downside_base'].astype(float).round(3)
                merge_df['acquirer_downside_outlier'] = merge_df['acquirer_downside_outlier'].astype(float).round(3)
                merge_df = merge_df.round(3)
                merge_df = merge_df.replace(np.nan, '')

                # iterate through all_downsides_df and replace with live downsides if available
                for index, row in all_downsides_df.iterrows():
                    tg = row['tradegroup']
                    if tg in merge_df['tradegroup'].tolist():
                        tg_df = merge_df[merge_df['tradegroup'] == tg]
                        # check if value is valid before insertion
                        fields_list = ['target_downside_base', 'acquirer_downside_base', 'acquirer_downside_outlier']
                        for field in fields_list:
                            if tg_df[field].values[0]:
                                all_downsides_df.loc[index, field] = tg_df[field].values[0]

            downsides_df = all_downsides_df[~all_downsides_df['tradegroup'].isin(etf_only_tg)]
            # filter for ARBETF only tradegroups
            etf_downsides_df = all_downsides_df[all_downsides_df['tradegroup'].isin(etf_only_tg)]
            error_msg = ""

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            downsides_df = pd.DataFrame()
            etf_downsides_df = pd.DataFrame()
            print('Error found:' + str(e)[0:250])
        return render(request, 'peer_downsides.html',
                      context={'downsides': downsides_df, 'etf_downsides': etf_downsides_df,
                               'date': selected_date, 'error_msg': error_msg})
    elif request.method == 'POST':
        entry = request.POST
        try:
            trend_obj = Downside_Trendlines.objects.get(id=entry['id'])
            trend_obj.ufp_base = entry['ufp_target'] if entry['ufp_target'] else None
            trend_obj.jefferies_base = entry['jefferies_target'] if entry['jefferies_target'] else None
            trend_obj.cowen_base = entry['cowen_target'] if entry['cowen_target'] else None
            trend_obj.jefferies_acquirer = entry['jefferies_acquirer'] if entry['jefferies_acquirer'] else None
            trend_obj.ufp_acquirer = entry['ufp_acquirer'] if entry['ufp_acquirer'] else None
            trend_obj.cowen_acquirer = entry['cowen_acquirer'] if entry['cowen_acquirer'] else None
            trend_obj.notes = entry['notes']
            trend_obj.save()
            return JsonResponse({'status': 'ok'})
        except Exception as e:
            return JsonResponse({'status': repr(e)})


def add_peer_downsides(request):
    """
    Page for creating new peer downside
    """
    if request.method == 'GET':
        curr_date = datetime.datetime.now().strftime('%Y-%m-%d')
        formula_df = pd.DataFrame.from_records(FormulaeBasedDownsides.objects.filter(IsExcluded='No').values())
        formula_df = formula_df[~formula_df['Underlying'].str.upper().str.contains('CVR EQUITY')]
        target_df = formula_df[formula_df['TargetAcquirer'] == 'Target']
        acq_df = formula_df[formula_df['TargetAcquirer'] == 'Acquirer']
        merge_df = pd.merge(target_df, acq_df, on='TradeGroup', how='left')
        merge_df = merge_df[['TradeGroup', 'base_case_x', 'base_case_y', 'outlier_y']]
        merge_df.rename(columns={'base_case_x': 'target_downside_base', 'base_case_y': 'acquirer_downside_base',
                                 'outlier_y': 'acquirer_downside_outlier'}, inplace=True)
        # cast rows to str
        merge_df.fillna('', inplace=True)
        merge_df['date'] = curr_date
        merge_df['Target'] = merge_df['TradeGroup'].str.split('-', expand=True)[0]
        merge_df['Acquirer'] = merge_df['TradeGroup'].str.split('-', expand=True)[1]
        return render(request, 'add_peer_downsides.html', context={'downsides': merge_df,
                                                                   'date': curr_date})
    elif request.method == 'POST':
        downsides_data = json.loads(request.POST['data'])
        downside_date = downsides_data['date']
        entry_list = []
        # remove previous saved data for the date
        Downside_Trendlines.objects.filter(date=downside_date).delete()
        for entry in downsides_data['downsides']:
            trend_obj = Downside_Trendlines(date=downside_date,
                                            tradegroup=entry['tradegroup'],
                                            target_ticker=entry['target_ticker'],
                                            )
            if entry['target_downside_base']:
                trend_obj.target_downside_base = entry['target_downside_base']
            if entry['ufp_target']:
                trend_obj.ufp_base = entry['ufp_target']
            if entry['jeff_target']:
                trend_obj.jefferies_base = entry['jeff_target']
            if entry['cowen_target']:
                trend_obj.cowen_base = entry['cowen_target']
            if entry['acquirer_downside_base']:
                trend_obj.acquirer_downside_base = entry['acquirer_downside_base']
            if entry['acquirer_downside_outlier']:
                trend_obj.acquirer_downside_outlier = entry['acquirer_downside_outlier']
            if entry['jeff_acquirer']:
                trend_obj.jefferies_acquirer = entry['jeff_acquirer']
            if entry['ufp_acquirer']:
                trend_obj.ufp_acquirer = entry['ufp_acquirer']
            if entry['cowen_acquirer']:
                trend_obj.cowen_acquirer = entry['cowen_acquirer']
            entry_list.append(trend_obj)
        Downside_Trendlines.objects.bulk_create(entry_list)

        return JsonResponse({'status': 'ok'})


def weekday_check_fix(diff_days=1):
    '''return last weekday as diff in days'''
    # of type datetime.datetime.today()
    date_no = diff_days
    weekno = (datetime.datetime.today() - timedelta(days=diff_days)).weekday()
    while weekno > 4:  # 5 Sat, 6 Sun
        date_no += 1
        weekno -= 1
    return date_no


def create_downside_history_streamer(request):
    """ downside history page created on 9/15. Tracks tradegroup live-Prices price-Deltas  """

    test = False
    exp_snap_fund = 'ARB'
    numerical_cols = ['ARB_exposure', 'ARB_nav_risk', 'last_price', 'day_one_downside', 'live_downside',
                      'one_day_delta', 'five_day_delta', 'thirty_day_delta', 'inter_day_delta', 'day_one_day_delta']
    final_cols = ['Datestamp', 'Underlying', 'outlier', 'diff_days', 'base_case', 'TargetAcquirer', 'base_case_f',
                  'LastPrice_f', 'Target', 'TradeGroup', 'ARB_exposure', 'ARB_nav_risk', 'last_price',
                  'day_one_downside', 'live_downside', 'one_day_delta', 'five_day_delta', 'thirty_day_delta',
                  'inter_day_delta', 'day_one_day_delta', 'BaseCaseDownsideType', 'action_id']
    formulae_columns = ['TradeGroup', 'base_case', 'LastPrice', 'Underlying', 'TargetAcquirer', 'day_one_downside',
                        'BaseCaseDownsideType']
    as_of_yyyy_mm_dd = None
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                           + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()
    if as_of_yyyy_mm_dd is None:
        as_of_yyyy_mm_dd = ExposuresSnapshot.objects.all().aggregate(Max('date'))['date__max'].strftime('%Y-%m-%d')

    # get all data
    if test:
        as_of_yyyy_mm_dd = '2022-08-18'
    # hist_downside_df = pd.DataFrame.from_records(HistoricalFormulaeBasedDownsides.objects.filter(Datestamp=as_of_yyyy_mm_dd).values())
    ma_deals_df = pd.DataFrame.from_records(MA_Deals.objects.filter(archived=False).values())
    hist_downside_df = pd.DataFrame.from_records(
        HistoricalFormulaeBasedDownsides.objects.filter(Datestamp__lte=as_of_yyyy_mm_dd).values())
    daily_nav_df = pd.DataFrame.from_records(DailyNAVImpacts.objects.all().values())
    exp_snap_df = pd.DataFrame.from_records(
        list(ExposuresSnapshot.objects.filter(date=as_of_yyyy_mm_dd).filter(fund=exp_snap_fund).values()))
    current_downside_df = pd.DataFrame.from_records(FormulaeBasedDownsides.objects.all().values())

    # restructure
    ma_deals_df = ma_deals_df[['deal_name', 'target_ticker', 'acquirer_ticker', 'id']]
    ma_deals_df['action_id'] = ma_deals_df['id']
    # ma_deals_df.reset_index(inplace=True)
    # ma_deals_df.rename(columns={'id': 'action_id'},inplace=True)
    # edge case where US with spaces in deal-names arent equivalent in naming convention #ZEN - HELLMAN & FRIEDMAN
    ma_deals_df['deal_name'] = ma_deals_df['deal_name'].apply(
        lambda x: x.replace(' US', '') if "& FRIEDMAN" not in x else x.replace('HELLMAN & FRIEDMAN', 'HELLMAN'))
    # ma_deals_df['deal_name'] = ma_deals_df['deal_name'].apply(lambda x: x.replace('WSP LN', 'WSP CN') if "WSP LN" in x else x)
    api_host = bbgclient.bbgclient.get_next_available_host()
    df = pd.read_sql_query('call wic.GET_POSITIONS_FOR_DOWNSIDE_FORMULAE()', con=con)
    df.index.names = ['id']
    df.rename(columns={'TG': 'TradeGroup'}, inplace=True)
    exp_snap_df.rename(columns={'tradegroup': 'TradeGroup'}, inplace=True)
    exp_snap_df = exp_snap_df[['TradeGroup', 'alpha_exposure', 'date']]
    ma_deals_df.rename(columns={'deal_name': 'TradeGroup'}, inplace=True)

    # restructure2
    df['TradeGroup'] = df['TradeGroup'].apply(lambda x: x.strip().upper())
    df['Underlying'] = df['Underlying'].apply(lambda x: x.strip().upper())
    df['Underlying'] = df['Underlying'].apply(lambda x: x + ' EQUITY' if 'EQUITY' not in x else x)
    all_unique_tickers = list(df['Underlying'].unique())
    live_price_df = pd.DataFrame.from_dict(bbgclient.bbgclient.get_secid2field(
        all_unique_tickers, 'tickers', ['PX_LAST'], req_type='refdata', api_host=api_host), orient='index')
    live_price_df = live_price_df.reset_index()
    live_price_df['PX_LAST'] = live_price_df['PX_LAST'].apply(lambda x: x[0])
    live_price_df.columns = ['Underlying', 'PX_LAST']
    current_downside_df = current_downside_df[formulae_columns]
    tody = date.today()
    as_of_dt = tody
    current_downside_df['Datestamp'] = as_of_dt
    # rename: separate formulae from historical
    current_downside_df['LastPrice_f'] = current_downside_df['LastPrice']
    current_downside_df['base_case_f'] = current_downside_df['base_case']
    # current_downside_df.rename(columns={'LastPrice': 'LastPrice_f'}, inplace=True)

    # Merge all data
    df = pd.merge(ma_deals_df, hist_downside_df, on='TradeGroup', how='left')
    # df[df['TradeGroup']=="IEA - MTZ"].to_csv('test1.csv', index=False)
    current_downside_df = current_downside_df[
        current_downside_df['TradeGroup'].isin(ma_deals_df['TradeGroup'].tolist())]
    df = df.append(current_downside_df)
    df = pd.merge(df, daily_nav_df, on='TradeGroup', how='left')
    # df.drop_duplicates(inplace=True)
    df = pd.merge(df, exp_snap_df, on='TradeGroup', how='left')
    df = pd.merge(df, live_price_df, how='left', on='Underlying')
    df.drop_duplicates(inplace=True)
    df_temp = df[df['base_case'].notna()]
    df_early_date = df_temp.sort_values('Datestamp').groupby('TradeGroup').apply(lambda x: x.head(1))
    df_early_date.reset_index(drop=True, inplace=True)
    df_early_date.rename(columns={'base_case': 'early_base_case'}, inplace=True)
    df = pd.merge(df, df_early_date[['TradeGroup', 'early_base_case']], on='TradeGroup', how='left')

    # define pre-calcs
    def delta_calc(row, diff=0):
        if row['diff_days'] == diff:
            price_base_case = row['base_case']
        else:
            return np.nan
        if isinstance(price_base_case, str):
            price_base_case = float(price_base_case)
        if isinstance(row['live_downside'], str):
            row['live_downside'] = float(row['live_downside'])
        if row['live_downside'] is not None and price_base_case is not None:
            if row['live_downside'] - price_base_case == 0:
                return 0
        if price_base_case != 0 and row['live_downside'] and price_base_case:
            return 100 * ((row['live_downside'] - price_base_case) / price_base_case)
        else:
            return np.nan

    def inter_delta_calc(row, diff=0):
        if row['PX_LAST']:
            if isinstance(row['live_downside'], str):
                if row['live_downside']:
                    row['live_downside'] = float(row['live_downside'])
                else:
                    row['live_downside'] = 0
            if isinstance(row['early_base_case'], str):
                if row['early_base_case']:
                    row['early_base_case'] = float(row['early_base_case'])
                else:  # catch for empty string
                    row['early_base_case'] = 0
            if row['early_base_case'] != 0 and row['early_base_case'] and row['live_downside']:
                return 100 * (row['live_downside'] - row['early_base_case']) / row[
                    'early_base_case']  # live_down - first_downside fix
            else:
                return np.nan
        else:
            return np.nan

    def day_one_calc(row, diff=0):
        if row['TargetAcquirer'] == 'Acquirer':
            return np.nan
        if isinstance(row['live_downside'], str):
            row['live_downside'] = float(row['live_downside'])
        if isinstance(row['day_one_downside_new'], str):
            row['day_one_downside_new'] = float(row['day_one_downside_new'])
        if row['day_one_downside_new'] != 0 and row['day_one_downside_new'] and row['live_downside']:
            return 100 * (row['live_downside'] - row['day_one_downside_new']) / row['day_one_downside_new']
        else:
            return np.nan

    # calc values
    # as_of_dt = date(year=2022, month=8, day=18)
    diff_days_num = weekday_check_fix()
    diff_days_five = weekday_check_fix(5)
    diff_days_thirty = weekday_check_fix(30)
    as_of_dt = datetime.datetime.today()
    # subtract today from historical-formulae date
    df['diff_days'] = df['Datestamp'].apply(lambda x: (as_of_dt.date() - x).days if pd.notnull(x) else '')
    df['live_downside'] = df.groupby(['TradeGroup', 'TargetAcquirer'], sort=False)['base_case_f'].apply(
        lambda x: x.ffill().bfill())  # x.ffill().bfill()
    df['Target'] = df['Underlying'].apply(lambda x: x)
    df['ARB_exposure'] = df['alpha_exposure']
    df['ARB_nav_risk'] = df['BASE_CASE_NAV_IMPACT_ARB']
    df['last_price'] = df['LastPrice']
    # df['live_downside'] = df['base_case_f'] # this will override the ffill bfill
    df['day_one_downside_new'] = df['day_one_downside']
    df['one_day_delta'] = df.apply(lambda x: delta_calc(x, diff_days_num), axis=1)
    df['five_day_delta'] = df.apply(lambda x: delta_calc(x, diff_days_five),
                                    axis=1)  # Live Downside - histFormulae t-5 base_case /  histFormulae t-5 base_case
    df['thirty_day_delta'] = df.apply(lambda x: delta_calc(x, diff_days_thirty), axis=1)
    df['inter_day_delta'] = df.apply(lambda x: inter_delta_calc(x), axis=1)
    df['day_one_day_delta'] = df.apply(lambda x: day_one_calc(x), axis=1)
    df['one_day_delta'] = df.groupby(['TradeGroup', 'TargetAcquirer'], sort=False)['one_day_delta'].apply(
        lambda x: x.ffill().bfill())
    df['five_day_delta'] = df.groupby(['TradeGroup', 'TargetAcquirer'], sort=False)['five_day_delta'].apply(
        lambda x: x.bfill().ffill())
    df['thirty_day_delta'] = df.groupby(['TradeGroup', 'TargetAcquirer'], sort=False)['thirty_day_delta'].apply(
        lambda x: x.ffill().bfill())
    df['action_id'] = df.groupby(['TradeGroup', 'TargetAcquirer'], sort=False)['action_id'].apply(
        lambda x: x.ffill().bfill())

    # final formatting
    df = df.sort_values(by=['TradeGroup', 'Datestamp', 'TargetAcquirer'], ascending=True)
    df = df[final_cols]
    df.reset_index(drop=True)
    df = df[df['Datestamp'] == holiday_utils.get_todays_date()]
    # remove cases where exposure snapshot mapped over and has zeroes. exp-snap database has duplicate tradegroup rows with zeroes across the row for other columns.
    not_list = df.columns.difference(['alpha_exposure']).tolist()
    df = df[(~df.duplicated(not_list, keep=False)) & (df['ARB_exposure'] != 0)]
    # remove zero arb exposures
    df = df[(df['ARB_exposure'] != 0)]
    df = df.dropna(subset=['ARB_exposure'])

    # drop edge case where arb is nan and live/day1 downsides are nan
    # df = df[((~df['ARB_nav_risk'].isna()) & (~df['day_one_downside'].isna()))]
    # get top row for color coding and create new color coding column. STd dev
    new_topten = []
    for c in numerical_cols:
        if 'ARB_nav_risk' == c:
            # difficult to fix
            df[c] = df[c].astype(float)
            m, ss = df[c].mean(), df[c].std()
            nth = m + ss
            nth_low = m - ss
            # df_top = df[c].sort_values(ascending=False).head(10).reset_index(drop=True)
            # nth = df_top.iloc[-1]
        elif "day_one_downside" == c:
            # df_top = df[c].dropna().astype(float).sort_values(ascending=False).head(10).reset_index(drop=True)
            df_top = df[c].dropna().astype(float)
            m, ss = df_top.mean(), df_top.std()
            nth = m + ss
            nth_low = m - ss
            df[c] = df[c].dropna().astype(float)
            # nth = df_top.iloc[-1]
        else:
            try:
                df[c] = df[c].astype(float)
                m, ss = df[c].mean(), df[c].std()
                nth = m + ss
                nth_low = m - ss
                # df_top = df[c].sort_values(ascending=False).head(10).reset_index(drop=True)
            except:
                # df_top = df[c].sort_values(ascending=False).head(10).reset_index(drop=True)
                m, ss = df[c].mean(), df[c].std()
                nth = m + ss
                nth_low = m - ss

            # nth = df_top.iloc[-1]
        new_topten.append({'topten_' + c: nth})
        new_topten.append({'lowten_' + c: nth_low})
        df_top = pd.DataFrame()
        m, ss = None, None
    new_dict_hyperlink = {}

    for rr, key in zip(list(df['action_id']), df['TradeGroup']):
        try:
            new_dict_hyperlink[key.replace(' ', '_').replace('-', '')] = str(int(rr))
        except:
            pass
    df['day_one_downside'].fillna('Na', inplace=True)
    # combine data into single dictionary
    final_dct = {'history_data': df.to_json(orient='records')}
    for d in new_topten:
        final_dct.update(d)
    final_dct['action_ids'] = new_dict_hyperlink
    final_dct.update({'last_synced_on': datetime.datetime.today().strftime('%Y-%m-%d %H:%M')})
    if request.is_ajax():
        # return_data = {'data': final_live_df.to_json(orient='records'),
        #                'final_live_itd_df': final_live_itd_df.to_json(orient='records'),
        #                'daily_pnl': final_daily_pnl.to_json(orient='records'),
        #                'position_level_pnl': position_level_pnl.to_json(orient='records'),
        #                'final_position_level_ytd_pnl': final_position_level_ytd_pnl.to_json(orient='records'),
        #                'final_position_level_itd_pnl': final_position_level_itd_pnl.to_json(orient='records'),
        #                'fund_drilldown_details': fund_drilldown_details.to_json(orient='records'),
        #                'last_synced_on': datetime.today().strftime('%Y-%m-%d')}
        final_dct.update({'history_data': df.to_json(orient='records')})
        final_dct.update({'last_synced_on': datetime.datetime.today().strftime('%Y-%m-%d %H:%M')})
        return JsonResponse(final_dct)
    return render(request, 'downside_history.html', context=final_dct)
