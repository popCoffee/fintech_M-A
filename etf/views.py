import datetime
import pandas as pd
import numpy as np
import json

from django.db.models import Max
from django.shortcuts import render
from django.db import connection
from django.utils.datastructures import MultiValueDictKeyError

from securities.models import SecurityMaster
from urllib.parse import urlencode
from django.http import HttpResponse, JsonResponse
from django.views.generic.edit import FormView
from django.views.generic import TemplateView
from django.urls import reverse
from .forms import ETFTrackingErrorInputForm,ETFDailyTrackingInputForm
from .tasks import get_etf_tracking_error, update_etf_pnl_and_bskt
from .models import EtfRecRecords, EtfRecSummary, ETFLivePnL, ETFMonitors, CustomUserInputs, MarketOnClose
from celery.result import AsyncResult
from django.conf import settings


# Create your views here.


def get_etf_positions(request):
    as_of = "(SELECT MAX(Flat_file_as_of) from wic.daily_flat_file_db)"
    if 'as_of' in request.GET:
        as_of = "'" + request.GET['as_of'] + "'"
    positions_df = SecurityMaster.objects.raw("SELECT 1 as id,flat_file_as_of, fund, Sleeve, Bucket, AlphaHedge, "
                                              "CatalystTypeWIC, CatalystRating, TradeGroup, Ticker, amount, Price, "
                                              "CurrentMktVal, aum, CurrentMktVal_Pct, CCY FROM "
                                              "wic.daily_flat_file_db where Fund = 'ARBETF' and flat_file_as_of=" +
                                              as_of)

    return render(request, 'etf_positions.html', {'etf_positions': positions_df})


def get_etf_performances(request):
    def get_tg_cum_pnl(row, y):
        cum_pnl = y[y['TradeGroup'] == row]['pct_pnl'].sum()
        return cum_pnl

    def create_pnl_chart_url(row):
        url = '../etf/get_tradegroup_etf_pnl?'
        tg_fund = {'TradeGroup': row['TradeGroup'], 'Fund': row['Fund']}
        url = url + urlencode(tg_fund)
        return "<button class='btn btn-sm' data-url=" + url + ">View Line Graph </button>"

    perf_df = pd.read_sql_query("SELECT date, Fund, TradeGroup, pnl FROM " + settings.CURRENT_DATABASE +
                                ".positions_and_pnl_tradegroupmaster where Fund like 'ARBETF'",
                                con=connection)

    fund_aum_df = pd.read_sql_query("SELECT DISTINCT flat_file_as_of as `Date`,Fund, `aum` FROM "
                                    "wic.daily_flat_file_db where fund like 'ARBETF'"
                                    , con=connection)

    tradegroups_pnl_df = pd.read_sql_query("SELECT `Date`, Fund, TradeGroup, pnl, aum FROM " \
                                           + settings.CURRENT_DATABASE + ".positions_and_pnl_tradegroupmaster WHERE " \
                                                                         "Fund LIKE 'ARBETF' AND `Date` = (SELECT MAX(`Date`) FROM "
                                           + settings.CURRENT_DATABASE + ".positions_and_pnl_tradegroupmaster)",
                                           con=connection)

    tg_perf_max_date = str(tradegroups_pnl_df['Date'].max())

    tg_perf_df = pd.read_sql_query("SELECT `date` as Date,TradeGroup, Fund, `pnl` FROM " \
                                   + settings.CURRENT_DATABASE + ".positions_and_pnl_tradegroupmaster WHERE "
                                                                 "Fund like 'ARBETF' ", con=connection)

    pct_of_assets = pd.read_sql_query("SELECT DISTINCT flat_file_as_of as `Date`, TradeGroup, Fund, "
                                      "100*(SUM(CurrentMktVal) / aum) AS "
                                      "`pct_of_assets` FROM wic.daily_flat_file_db WHERE Fund LIKE 'ARBETF' AND "
                                      "Flat_file_as_of = '" + tg_perf_max_date + "' "
                                                                                 "GROUP BY Date, TradeGroup, Fund",
                                      con=connection)

    tg_perf_df['pnl'] = tg_perf_df['pnl'].apply(lambda x: int(float(x)))
    # Add AUM
    tg_perf_df = pd.merge(tg_perf_df, fund_aum_df, on=['Date', 'Fund'])
    tg_perf_df['pct_pnl'] = 1e2 * (tg_perf_df['pnl'] / tg_perf_df['aum'])
    del tg_perf_df['aum']
    pct_of_assets['pct_of_assets'] = pct_of_assets['pct_of_assets'].apply(lambda x: str(np.round(x, decimals=2)) + "%")

    tradegroups_pnl_df = pd.merge(tradegroups_pnl_df, pct_of_assets, how='inner', on=['Date', 'TradeGroup', 'Fund'])

    tradegroups_pnl_df['Date'] = tradegroups_pnl_df['Date'].apply(str)
    tradegroups_pnl_df['pnl'] = tradegroups_pnl_df['pnl'].apply(lambda x: int(float(x)))
    tradegroups_pnl_df['pct_pnl'] = 1e2 * (tradegroups_pnl_df['pnl'] / tradegroups_pnl_df['aum'])

    tradegroups_pnl_df['pnl_chart_url'] = tradegroups_pnl_df.apply(create_pnl_chart_url, axis=1)

    perf_df.rename(columns={'date': 'Date', 'fund': 'Fund'}, inplace=True)
    perf_df = pd.merge(perf_df, fund_aum_df, on=['Date', 'Fund'])
    perf_df['Date'] = perf_df['Date'].apply(str)
    # Remove CCY

    perf_df['pnl_bps'] = perf_df.apply(convert_to_bps, axis=1)

    del perf_df['aum']
    del perf_df['pnl']

    tradegroups_pnl_dict = {}

    fund_level_df = perf_df.groupby(['Date', 'Fund']).agg('sum').reset_index()
    perf_dict = {}
    unique_funds = fund_level_df['Fund'].unique()
    for fund in unique_funds:
        fund_cum_pnl = fund_level_df[fund_level_df['Fund'] == fund]
        fund_cum_pnl['cum_pnl'] = fund_cum_pnl['pnl_bps'].cumsum()
        fund_cum_pnl[['pnl_bps', 'cum_pnl']] = fund_cum_pnl[['pnl_bps', 'cum_pnl']].round(2)
        perf_dict[fund] = fund_cum_pnl.to_json(orient='records')
        tg_fund_perf = tg_perf_df[tg_perf_df['Fund'] == fund]
        tradegroups_df = tradegroups_pnl_df[tradegroups_pnl_df['Fund'] == fund]
        tradegroups_df['cum_pnl'] = tradegroups_df['TradeGroup'].apply(lambda x: get_tg_cum_pnl(x, tg_fund_perf))
        tradegroups_df['pct_cum_pnl'] = tradegroups_df['cum_pnl']
        tradegroups_df['pct_cum_pnl'] = tradegroups_df['pct_cum_pnl'].apply(
            lambda x: str(np.round(x, decimals=2)) + "%")
        tradegroups_df['pct_pnl'] = tradegroups_df['pct_pnl'].apply(lambda x: str(np.round(x, decimals=2)) + "%")
        del tradegroups_df['aum']
        del tradegroups_df['Fund']
        del tradegroups_df['pnl']
        del tradegroups_df['cum_pnl']
        tradegroups_pnl_dict[fund] = tradegroups_df.to_json(orient='records')

    return render(request, 'etf_performances.html', {'fund_level_performance': fund_level_df.to_json(orient='records'),
                                                     'etf_cum_pnl': json.dumps(perf_dict),
                                                     'tradegroups_pnl': json.dumps(tradegroups_pnl_dict)
                                                     })


def get_tradegroup_etf_pnl(request):
    fund = request.GET['Fund']
    tradegroup = request.GET['TradeGroup']

    tg_perf_df = pd.read_sql_query("SELECT `date`, `pnl` FROM prod_wic_db.positions_and_pnl_tradegroupmaster where "
                                   "Fund like '" + fund + "' AND TradeGroup like '" + tradegroup + "'", con=connection)

    fund_aum_df = pd.read_sql_query(
        "SELECT DISTINCT flat_file_as_of as `date`, `aum` FROM wic.daily_flat_file_db where "
        "Fund like '" + fund + "'", con=connection)

    tg_perf_df = pd.merge(tg_perf_df, fund_aum_df, how='inner', on=['date'])
    tg_perf_df['pnl'] = tg_perf_df.apply(lambda x: np.round(convert_to_bps(x), decimals=2), axis=1)
    tg_perf_df['cum_pnl'] = tg_perf_df['pnl'].cumsum()
    tg_perf_df['date'] = tg_perf_df['date'].apply(str)

    return HttpResponse(tg_perf_df.to_json(orient='records'))


def convert_to_bps(row, default_col='pnl'):
    return 1e4 * (float(row[default_col]) / float(row['aum']))


class ETFTrackingErrorView(FormView):
    """
    View for submitting the Pnl Attribution Analysis form details
    """
    template_name = 'etf_tracking_error.html'
    form_class = ETFTrackingErrorInputForm
    fields = '__all__'
    task_id_dict = dict()

    def get_success_url(self):
        return reverse('etf:get_tracking_error')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['task_id'] = self.task_id_dict.get('values', None)
        context['form_data'] = json.dumps(self.task_id_dict.get('form_data', {}))
        context['start_date'] = self.task_id_dict.get('start_date', '')
        context['end_date'] = self.task_id_dict.get('end_date', '')
        self.task_id_dict['values'] = None
        self.task_id_dict['form_data'] = json.dumps({})
        self.task_id_dict['start_date'] = ''
        self.task_id_dict['end_date'] = ''
        return context

    def form_valid(self, form):
        form_data = form.cleaned_data
        start_date = form_data['start_date'].strftime('%Y-%m-%d')
        end_date = form_data['end_date'].strftime('%Y-%m-%d')
        # task_id_response = tracking_error_helper(form_data)
        task_id = None #task_id_response.get('task_id',None) # if task_id_response.get('task_id') else None
        self.task_id_dict['values'] = task_id
        self.task_id_dict['start_date'] = start_date
        self.task_id_dict['end_date'] = end_date
        self.task_id_dict['form_data'] = {'fund': form_data['fund'],
                                          'start_date': start_date, 'end_date': end_date,
                                          }
        return super(ETFTrackingErrorView, self).form_valid(form)


def tracking_error_helper(form_data):
    start_date = form_data['start_date'].strftime('%Y-%m-%d')
    end_date = form_data['end_date'].strftime('%Y-%m-%d')
    start_date_tradar = form_data['start_date'].strftime('%Y%m%d')
    end_date_tradar = form_data['end_date'].strftime('%Y%m%d')
    task = get_etf_tracking_error.delay(start_date, end_date, start_date_tradar, end_date_tradar)
    return {'task_id': task.id}


def get_tracking_error_results_json(request):
    if request.method == 'POST':
        task_id = request.POST.get('task_id')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')
        start_date_tradar = datetime.datetime.strptime(request.POST.get('start_date'), '%Y-%m-%d').strftime('%Y%m%d')
        end_date_tradar = datetime.datetime.strptime(request.POST.get('end_date'), '%Y-%m-%d').strftime('%Y%m%d')
        # if not task_id or task_id in ['null', 'None']:
        #     return JsonResponse({'error': 'No task ID found. Refresh and try again.'})
        # else:
        data_json = get_etf_tracking_error(start_date, end_date, start_date_tradar, end_date_tradar)
        return JsonResponse({'data': data_json})


#####
class ETFDailyTrackingView(FormView):
    """
    View for submitting the Pnl Attribution Analysis form details
    """
    template_name = 'etf_daily_tracking_error.html'
    form_class = ETFDailyTrackingInputForm
    fields = '__all__'
    task_id_dict = dict()

    def get_success_url(self):
        return reverse('etf:check_daily_pnl')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # context['task_id'] = self.task_id_dict.get('values', None)
        # context['form_data'] = json.dumps(self.task_id_dict.get('form_data', {}))
        context['start_date'] = self.task_id_dict.get('start_date', '')
        # context['end_date'] = self.task_id_dict.get('end_date', '')
        # self.task_id_dict['values'] = None
        # self.task_id_dict['form_data'] = json.dumps({})
        self.task_id_dict['start_date'] = ''
        # self.task_id_dict['end_date'] = ''
        return context

    def form_valid(self, form):
        form_data = form.cleaned_data
        start_date = form_data['start_date'].strftime('%Y-%m-%d')
        # end_date = form_data['end_date'].strftime('%Y-%m-%d')
        result = daily_tracking_helper(form_data)
        task_id = 'None' #task_id_response.get('task_id',None) # if task_id_response.get('task_id') else None
        self.task_id_dict['values'] = result
        self.task_id_dict['start_date'] = start_date
        # self.task_id_dict['end_date'] = end_date
        # self.task_id_dict['form_data'] = {'fund': form_data['fund'],
        #                                   'start_date': start_date, 'end_date': end_date,
        #                                   }
        return super(ETFTrackingErrorView, self).form_valid(form)


def daily_tracking_helper(form_data):
    from etf.tests import compare_pnl_date
    start_date = form_data['start_date'].strftime('%Y-%m-%d')
    start_date_tradar = form_data['start_date'].strftime('%Y%m%d')
    result = compare_pnl_date(start_date, start_date_tradar )
    result = result.get('detailed_tracking_error',[])
    return result


def daily_pnl_crosscheck_post(request):
    if request.method == 'POST':
        from etf.tests import compare_pnl_date
        print('running views.py!!')
        # task_id = request.POST.get('task_id')
        start_date = request.POST.get('start_date')
        # end_date = request.POST.get('end_date')
        start_date_tradar = datetime.datetime.strptime(request.POST.get('start_date'), '%Y-%m-%d').strftime('%Y%m%d')
        # end_date_tradar = datetime.datetime.strptime(request.POST.get('end_date'), '%Y-%m-%d').strftime('%Y%m%d')
        # if not task_id or task_id in ['null', 'None']:
        #     return JsonResponse({'error': 'No task ID found. Refresh and try again.'})
        # else:

        data_json = compare_pnl_date(start_date, start_date_tradar)
        print(data_json)
        return JsonResponse({'data': data_json})



# ETF REC TAB View

class EtfRecs(TemplateView):
    template_name = 'recs.html'

    def post(self, request, *args, **kwargs):
        response = 'Failed'
        if 'edit_id' in request.POST:
            try:
                edit_id = request.POST['edit_id']
                edit_note = request.POST['note']
                current_record = EtfRecRecords.objects.get(id=edit_id)
                current_record.notes = edit_note
                current_record.save()
                response = 'Success'
            except MultiValueDictKeyError as e:
                # id not found it the ETF records, do not add comment
                pass
        return HttpResponse(response)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if len(EtfRecRecords.objects.all()) == 0 or len(
                EtfRecSummary.objects.all()) == 0:  # return if no entires in records
            return context

        as_of_date = self.request.GET.get('as_of_date')

        if not as_of_date:
            selected_date = EtfRecRecords.objects.order_by('-date').first().date
        else:
            selected_date = datetime.datetime.strptime(as_of_date, '%Y-%m-%d').date()

        columns_order = ['date', 'sedol', 'eze_ticker', 'deal', 'basket', 'index', 'eze', 'index_eze',
                         'weight_tracked', 'pct_tracked', 'additional_etf_exposure', 'notes']
        try:
            recs_df = pd.DataFrame.from_records(list(EtfRecRecords.objects.filter(date=selected_date).all().values()),
                                                index='id')
        except KeyError:  # when selecting on an invalid date
            recs_df = pd.DataFrame()

        summary_object = None
        if not recs_df.empty:
            context['data_found'] = True
            # Get the Summary DataFrame too
            summary_object = EtfRecSummary.objects.get(date=selected_date)
            recs_df = recs_df[columns_order]
            context['latest_date'] = recs_df.iloc[0]['date'].strftime('%Y-%m-%d')
            recs_df['date'] = recs_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            context['recs_df'] = recs_df.to_dict(orient='index')  # include entry id for index
        else:
            context['data_found'] = False
            context['latest_date'] = as_of_date
            context['recs_df'] = json.dumps({})

        if summary_object:
            # summary_df['date'] = summary_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            context['summary_object'] = summary_object
        else:
            context['summary_df'] = json.dumps({})

        return context


# CBV for PNL
class EtfPnl(TemplateView):

    template_name = 'etf_pnl.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if len(ETFLivePnL.objects.all()) == 0:
            return context
        custom_user_inputs = None
        try:
            custom_user_inputs = CustomUserInputs.objects.first()
            etf_pnl_df = pd.DataFrame.from_records(list(ETFLivePnL.objects.all().values()))
            monitors_df = pd.DataFrame.from_records(list(ETFMonitors.objects.all().values()))
            market_on_close_df = pd.DataFrame.from_records(list(MarketOnClose.objects.all().values()))
        except KeyError:  # when selecting on an invalid date
            etf_pnl_df = pd.DataFrame()
            monitors_df = pd.DataFrame()

        if not etf_pnl_df.empty:
            context['custom_user_inputs'] = custom_user_inputs
            context['data_found'] = True
            # Get the Summary DataFrame too
            context['updated_on'] = etf_pnl_df['updated_on'].max().strftime('%Y-%m-%d %H:%M:%S')
            context['deal_pnl_df'] = etf_pnl_df[['tradegroup', 'deal_return', 'one_day_return', 'five_day_return',
                                                 'live_ytd_return']].dropna().drop_duplicates().to_dict(orient='index')
            context['etf_pnl_df'] = etf_pnl_df.to_dict(orient='index')  # include entry id for index
            context['market_on_close_df'] = market_on_close_df.to_dict(orient='index')
            if not monitors_df.empty:
                context['spread_monitor'] = json.loads(monitors_df.iloc[0]['spread_monitor'])
                context['bid_ask_monitor'] = json.loads(monitors_df.iloc[0]['bid_ask_monitor'])
                context['bid_ask_spread_monitor'] = json.loads(monitors_df.iloc[0]['bid_ask_spread_monitor'])
                context['nav_monitor'] = json.loads(monitors_df.iloc[0]['nav_monitor'])
                context['basket_valuation_monitor'] = json.loads(monitors_df.iloc[0]['basket_valuation_monitor'])
                context['unit_activity_monitor'] = json.loads(monitors_df.iloc[0]['unit_activity_monitor'])
                context['spread_estimate_monitor'] = json.loads(monitors_df.iloc[0]['spread_estimate_monitor'])

        else:
            context['data_found'] = False

        return context


def calculate_live_etf_pnl(request):
    response = {}
    if request.method == 'POST':
        task = update_etf_pnl_and_bskt.delay(record_progress=True)
        response['task_id'] = task.id
    return JsonResponse(response)


def update_custom_user_inputs(request):
    response = {}
    date = datetime.datetime.now().date()
    if request.method == 'POST':
        try:
            collateral_buffer = request.POST.get('collateral_buffer')
            net_td_creations = request.POST.get('net_td_creations')
            net_td_redemptions = request.POST.get('net_td_redemptions')
            tax = request.POST.get('tax')
            fees = request.POST.get('fees')
            object = CustomUserInputs.objects.get(id=CustomUserInputs.objects.last().id)

            if object:
                object.date_updated = date
                object.collateral_buffer = collateral_buffer
                object.net_td_creations = net_td_creations
                object.net_td_redemptions = net_td_redemptions
                object.tax = tax
                object.fees = fees
                object.save()
                response['success'] = 'Success'
            else:
                CustomUserInputs.objects.create(date_updated=date, collateral_buffer=collateral_buffer,
                                                net_td_creations=net_td_creations,
                                                net_td_redemptions=net_td_redemptions, tax=tax,
                                                fees=fees).save()
        except Exception as e:
            response['error'] = str(e)
    return JsonResponse(response)