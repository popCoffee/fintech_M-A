import datetime
try:
    from io import BytesIO as IO # for modern python
except ImportError:
    from io import StringIO as IO # for legacy python
from itertools import chain
import json
import numpy as np
import pandas as pd
import requests
from urllib.parse import urlencode

from django.conf import settings
from django.db import connection, transaction
from django.db.models import Max
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.generic import FormView, ListView
from sqlalchemy import create_engine

from .models import *
from .tasks import arb_hard_float_optimization, credit_hard_float_optimization, update_pnl_ess_constituents
from portfolio_optimization.forms import EssDealTypeParametersForm
from portfolio_optimization.tasks import update_pnl_ess_constituents_function
from portfolio_optimization.utils import format_data, get_ess_df, get_aed_sleeves_performance
from risk.models import MA_Deals, MA_Deals_Risk_Factors


def ess_target_configs(request):
    # Render a response to View the current ESS configs.
    deal_type_parameters = EssDealTypeParameters.objects.all()
    normalized_sizing = NormalizedSizingByRiskAdjProb.objects.all()
    soft_catalyst_sizing = SoftCatalystNormalizedRiskSizing.objects.all()

    return render(request, 'ess_targets_configs.html', {'deal_type_paramters': deal_type_parameters,
                                                        'normalized_sizing': normalized_sizing,
                                                        'soft_catalyst_sizing': soft_catalyst_sizing
                                                        }
                  )


class EssDealTypeParametersView(FormView):
    template_name = "ess_targets_configs.html"
    form_class = EssDealTypeParametersForm
    success_url = '#'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        deal_type_parameters = EssDealTypeParameters.objects.all()
        normalized_sizing = NormalizedSizingByRiskAdjProb.objects.all()
        soft_catalyst_sizing = SoftCatalystNormalizedRiskSizing.objects.all()
        context.update({'deal_type_parameters': deal_type_parameters, 'normalized_sizing': normalized_sizing,
                        'soft_catalyst_sizing': soft_catalyst_sizing})
        return context

    def form_valid(self, form):
        data = form.cleaned_data
        deal_type_id_to_edit = self.request.POST.get('deal_type_id')
        create_new_deal_type = not deal_type_id_to_edit
        if not create_new_deal_type:
            try:
                deal_type_obj = EssDealTypeParameters.objects.get(id=deal_type_id_to_edit)
                deal_type_obj.__dict__.update(data)
                deal_type_obj.save()
                create_new_deal_type = False
            except EssDealTypeParameters.DoesNotExist:
                create_new_deal_type = True
        if create_new_deal_type:
            if 'deal_type_id' in data.keys():
                data.pop('deal_type_id')
            EssDealTypeParameters.objects.create(**data)
        return super(EssDealTypeParametersView, self).form_valid(form)


def get_deal_type_details(request):
    """ Retreives all the details for the requested Deal Type """
    if request.method == 'POST':
        deal_type_id_to_edit = request.POST['deal_type_id_to_edit']
        deal_type_details = {}
        try:
            deal_type = EssDealTypeParameters.objects.get(id=deal_type_id_to_edit)
            deal_type_details['deal_type'] = deal_type.deal_type
            deal_type_details['long_probability'] = deal_type.long_probability
            deal_type_details['long_irr'] = deal_type.long_irr
            deal_type_details['long_max_risk'] = deal_type.long_max_risk
            deal_type_details['long_max_size'] = deal_type.long_max_size
            deal_type_details['short_probability'] = deal_type.short_probability
            deal_type_details['short_irr'] = deal_type.short_irr
            deal_type_details['short_max_risk'] = deal_type.short_max_risk
            deal_type_details['short_max_size'] = deal_type.short_max_size
        except EssDealTypeParameters.DoesNotExist:
            deal_type_details = []

    return JsonResponse({'deal_type_details': deal_type_details})


def delete_deal_type(request):
    response = None
    if request.method == 'POST':
        # Take the ID and Delete
        id_to_delete = request.POST['id']
        try:
            EssDealTypeParameters.objects.get(id=id_to_delete).delete()
            response = 'deal_type_deleted'
        except EssDealTypeParameters.DoesNotExist:
            response = 'deal_does_not_exist'

    return HttpResponse(response)


def update_soft_catalyst_risk_sizing(request):
    response = 'Failed'
    if request.method == 'POST':
        try:
            tier = request.POST['tier']
            win_probability = request.POST['win_probability']
            loss_probability = request.POST['loss_probability']
            max_risk = request.POST['max_risk']
            avg_position = request.POST['avg_position']

            SoftCatalystNormalizedRiskSizing.objects.filter(tier__contains=tier).delete()
            SoftCatalystNormalizedRiskSizing(tier=tier, win_probability=win_probability,
                                             loss_probability=loss_probability, max_risk=max_risk,
                                             avg_position=avg_position).save()
            response = 'Success'
        except Exception as e:
            print(e)

    return HttpResponse(response)


def update_normlized_sizing_by_risk_adj_prob(request):
    response = 'Failed'
    if request.method == 'POST':
        try:
            win_probability = request.POST['win_prob']
            loss_probability = request.POST['loss_prob']
            arb_max_risk = request.POST['arb_max_risk']
            risk_adj_loss = request.POST['risk_adj_loss']
            NormalizedSizingByRiskAdjProb.objects.all().delete()
            NormalizedSizingByRiskAdjProb(win_probability=win_probability, loss_probability=loss_probability,
                                          arb_max_risk=arb_max_risk, risk_adj_loss=risk_adj_loss).save()
            response = 'Success'
        except Exception as e:
            print(e)

    return HttpResponse(response)


class EssLongShortView(ListView):
    template_name = 'ess_potential_long_shorts.html'
    queryset = EssPotentialLongShorts.objects.all().order_by('-Date')
    context_object_name = 'esspotentiallongshorts_list'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        as_of = self.request.GET.get('as_of')
        if as_of:
            as_of = datetime.datetime.strptime(as_of, '%Y-%m-%d')
        else:
            as_of = EssPotentialLongShorts.objects.latest('Date').Date
        queryset = EssPotentialLongShorts.objects.filter(Date=as_of)
        context.update({'esspotentiallongshorts_list': queryset, 'as_of': as_of})
        return context


class EssImpliedProbabilityView(ListView):
    template_name = 'implied_probability_track.html'
    queryset = EssUniverseImpliedProbability.objects.all()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        as_of = self.request.GET.get('as_of')
        if as_of:
            as_of = datetime.datetime.strptime(as_of, '%Y-%m-%d')

        else:
            as_of = EssUniverseImpliedProbability.objects.latest('Date').Date

        queryset = EssUniverseImpliedProbability.objects.all()
        field_names = []
        implied_probability_chart = {}

        implied_probabilities_df = pd.DataFrame().from_records(queryset.
                                                           values('Date', 'deal_type', 'implied_probability'))

        # Get SPX Index Returnns
        start_date = implied_probabilities_df['Date'].min().strftime('%Y%m%d')
        end_date = implied_probabilities_df['Date'].max().strftime('%Y%m%d')

        r = requests.get("http://192.168.0.23:8080/wic/api/v1.0/general_histdata",
                         params={'idtype': "tickers", "fields": "PX_LAST",
                                 "tickers": "SPX INDEX",
                                 "override": "", "start_date": start_date, "end_date": end_date},
                         timeout=15)  # Set a 15 secs Timeout
        results = r.json()['results']
        spx_prices = pd.DataFrame.from_dict(results[0]['SPX INDEX']['fields'])
        spx_prices['PX_LAST'] = spx_prices['PX_LAST'].astype(float)
        spx_prices['implied_probability'] = ((spx_prices['PX_LAST'] / spx_prices['PX_LAST'].shift(
            1)) - 1.0) * 100  # daily change
        spx_prices['implied_probability'] = spx_prices['implied_probability'].cumsum()
        spx_prices['implied_probability'].fillna(0, inplace=True)
        spx_prices['deal_type'] = "SPX INDEX Ret(%)"
        del spx_prices['PX_LAST']
        spx_prices.rename(columns={'date': 'Date'}, inplace=True)

        implied_probabilities_df = pd.concat([implied_probabilities_df, spx_prices])
        if not implied_probabilities_df.empty:
            field_names = list(implied_probabilities_df['deal_type'].unique())
            implied_probabilities_df['implied_probability'] = implied_probabilities_df['implied_probability']. \
                apply(lambda x: np.round(x, decimals=2))

            implied_probabilities_df['Date'] = implied_probabilities_df['Date'].astype(str)
            implied_probabilities_df = implied_probabilities_df.pivot_table(columns=['deal_type'], index='Date'). \
                reset_index()

            implied_probabilities_df.columns = ["".join(('', j)) for i, j in implied_probabilities_df.columns]
            implied_probabilities_df.columns.values[0] = 'Date'
            implied_probabilities_df.reset_index(inplace=True)
            implied_probability_chart = implied_probabilities_df.to_json(orient='records')

        ess_implied_prb_universe = queryset.filter(Date=as_of)

        context.update({'implied_probability_chart': implied_probability_chart, 'field_names': json.dumps(field_names),
                        'ess_implied_prob': ess_implied_prb_universe, 'as_of': as_of})
        return context


def ess_implied_prob_drilldown(request):
    return_data = None
    if request.method == 'POST':
        try:
            date = request.POST['date']
            deal_type = request.POST['deal_type']
            date_adj = None
            if date == datetime.datetime.now().strftime('%Y-%m-%d'):
                date_adj = '(SELECT MAX(flat_file_as_of) from wic.daily_flat_file_db)'
            else:
                date_adj = "'" + date + "'"
            query = "SELECT DISTINCT flat_file_as_of as `Date`, TradeGroup, Fund, Ticker,Price, LongShort, SecType, " \
                    "DealUpside, DealDownside FROM wic.daily_flat_file_db WHERE Flat_file_as_of = " + date_adj + " AND " \
                    "Fund IN ('AED', 'TAQ') and AlphaHedge = 'Alpha' AND LongShort IN ('Long', 'Short') " \
                    "AND SecType = 'EQ' AND Sleeve = 'Equity Special Situations' and amount <> 0;"
            filtered_df = pd.read_sql_query(query, con=connection)
            return_data = get_implied_prob_df(filtered_df, date, deal_type)

        except Exception as e:
            print(e)
            return_data = None

    return JsonResponse({'data': return_data})


def get_implied_prob_df(imp_prob_tracker_df, date, deal_type, get_df=False):
    if deal_type in ['AED Long', 'AED Short', 'TAQ Long', 'TAQ Short']:

        if 'Long' in deal_type:
            imp_prob_tracker_df = imp_prob_tracker_df[imp_prob_tracker_df['LongShort'] == 'Long']
        else:
            imp_prob_tracker_df = imp_prob_tracker_df[imp_prob_tracker_df['LongShort'] == 'Short']

        # Slice for the Fund
        fund_code = deal_type.split(' ')[0]
        imp_prob_tracker_df = imp_prob_tracker_df[imp_prob_tracker_df['Fund'] == fund_code]

        imp_prob_tracker_df['implied_probability'] = 1e2 * (imp_prob_tracker_df['Price'] -
                                                            imp_prob_tracker_df['DealDownside']) / \
                                                     (imp_prob_tracker_df['DealUpside'] -
                                                      imp_prob_tracker_df['DealDownside'])

        imp_prob_tracker_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace Inf values

        imp_prob_tracker_df['Date'] = imp_prob_tracker_df['Date'].astype(str)

        imp_prob_tracker_df = imp_prob_tracker_df[['Date', 'Ticker', 'Price', 'TradeGroup',
                                                   'implied_probability']]

        # For Funds, provide a link to the TradeGroup story
        imp_prob_tracker_df['idea_link'] = imp_prob_tracker_df.apply(lambda x:
                                                                     "<td><a href='../position_stats/"
                                                                     "get_tradegroup_story?" +
                                                                     urlencode({'TradeGroup': x['TradeGroup'],
                                                                                'Fund': fund_code}) +
                                                                     "' target='_blank'>Story</a></td>",
                                                                     axis=1)
        imp_prob_tracker_df.columns = ['Date', 'alpha_ticker', 'price', 'deal_type', 'implied_probability',
                                       'idea_link']
        imp_prob_tracker_df['implied_probability'] = imp_prob_tracker_df['implied_probability'].round(2)
        if get_df:
            return imp_prob_tracker_df
        return imp_prob_tracker_df.to_json(orient='records')
    else:
        # Gather Data from Potential Long short timeseries..
        if deal_type == 'ESS IDEA Universe':
            implied_drilldowwn = pd.DataFrame.from_records(EssPotentialLongShorts.objects.all().filter(
                Date=date).values('Date', 'alpha_ticker', 'price', 'deal_type', 'implied_probability',
                                  'ess_idea_id'))

        elif deal_type == 'Universe (Long)':
            implied_drilldowwn = pd.DataFrame.from_records(EssPotentialLongShorts.objects.all().filter(
                Date=date, potential_long='Y').values('Date', 'alpha_ticker', 'price', 'deal_type',
                                                      'implied_probability', 'ess_idea_id'))

        elif deal_type == 'Universe (Short)':
            implied_drilldowwn = pd.DataFrame.from_records(EssPotentialLongShorts.objects.all().filter(
                Date=date, potential_short='Y').values('Date', 'alpha_ticker', 'price', 'deal_type',
                                                       'implied_probability', 'ess_idea_id'))

        elif deal_type == 'Universe (Unclassified)':
            implied_drilldowwn = pd.DataFrame.from_records(EssPotentialLongShorts.objects.all().filter(
                Date=date, potential_short='', potential_long='').values('Date', 'alpha_ticker', 'price', 'deal_type',
                                                                         'implied_probability', 'ess_idea_id'))

        elif deal_type == 'Soft Universe Imp. Prob':
            implied_drilldowwn = pd.DataFrame.from_records(EssPotentialLongShorts.objects.all().filter(
                Date=date, catalyst='Soft').values('Date', 'alpha_ticker', 'price', 'deal_type',
                                                   'implied_probability', 'ess_idea_id'))

        elif deal_type in ['Hard-1', 'Hard-2', 'Hard-3', 'Soft-1', 'Soft-2', 'Soft-3']:
            cat = deal_type.split('-')[0]
            catalyst_tier = deal_type.split('-')[1]
            implied_drilldowwn = pd.DataFrame.from_records(EssPotentialLongShorts.objects.all().filter(
                Date=date, catalyst=cat, catalyst_tier=catalyst_tier).values('Date', 'alpha_ticker', 'price',
                                                                             'deal_type', 'implied_probability',
                                                                             'ess_idea_id'))

        else:
            implied_drilldowwn = pd.DataFrame.from_records(EssPotentialLongShorts.objects.all().filter(
                Date=date, deal_type=deal_type).values('Date', 'alpha_ticker', 'price', 'deal_type',
                                                       'implied_probability', 'ess_idea_id'))
        if not implied_drilldowwn.empty:
            implied_drilldowwn['implied_probability'] = implied_drilldowwn['implied_probability'].round(2)
            implied_drilldowwn['Date'] = implied_drilldowwn['Date'].astype(str)

            implied_drilldowwn['idea_link'] = implied_drilldowwn['ess_idea_id'].apply(
                lambda x: "<td><a href='" + reverse('risk:show_ess_idea') + "?ess_idea_id=" + str(x) +
                "' target='_blank'>Open IDEA</a></td>")

        if get_df:
            return implied_drilldowwn
        return implied_drilldowwn.to_json(orient='records')


class WicUniverseRorView(ListView):
    template_name = 'merger_arb_ror.html'
    queryset = ArbOptimizationUniverse.objects.all().order_by('-ann_ror')
    context_object_name = 'arboptimizationuniverse_list'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        as_of = self.request.GET.get('as_of')
        if as_of:
            as_of = datetime.datetime.strptime(as_of, '%Y-%m-%d')
        else:
            as_of = ArbOptimizationUniverse.objects.latest('date_updated').date_updated
        queryset = ArbOptimizationUniverse.objects.filter(date_updated=as_of).order_by('-ann_ror')
        context.update({'arboptimizationuniverse_list': queryset, 'as_of': as_of})
        return context


class ArbHardOptimizationView(ListView):
    template_name = 'arb_hard_optimization.html'
    queryset = HardFloatOptimization.objects.filter(sleeve='Merger Arbitrage', date_updated=Max('date_updated'))

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        as_of = self.request.GET.get('as_of')
        if as_of:
            as_of = datetime.datetime.strptime(as_of, '%Y-%m-%d')
            as_of_summary = as_of
        else:
            as_of = HardFloatOptimization.objects.latest('date_updated').date_updated
            as_of_summary = HardOptimizationSummary.objects.latest('date_updated').date_updated
        arb_hard_df = pd.DataFrame.from_records(HardFloatOptimization.objects.filter(sleeve='Merger Arbitrage',
                                                                                     date_updated=as_of).values())
        merge_df = attach_ranking_cols_to_df(arb_hard_df)
        try:
            summary_queryset = HardOptimizationSummary.objects.get(date_updated=as_of_summary)
        except HardOptimizationSummary.DoesNotExist:
            summary_queryset = []
            print('Hard Optimization summary not available for ' + str(as_of_summary))
        context.update({'hard_optimization_list': json.loads(merge_df.to_json(orient='records')), 'as_of': as_of,
                        'summary_queryset': summary_queryset})
        return context


def attach_ranking_cols_to_df(arb_hard_df):
    tradegroup_list = arb_hard_df.tradegroup.unique().tolist()
    ma_deals_df = pd.DataFrame.from_records(MA_Deals.objects.filter(deal_name__in=tradegroup_list)
                                            .values('id', 'deal_name'))
    arb_hard_df['tradegroup'] = arb_hard_df['tradegroup'].str.upper()
    ma_deals_df['deal_name'] = ma_deals_df['deal_name'].str.upper()
    merge_df = pd.merge(arb_hard_df, ma_deals_df, left_on=['tradegroup'], right_on=['deal_name'], how='left')
    mna_factors_df = pd.DataFrame.from_records(MA_Deals_Risk_Factors.objects.all()
                                               .values('deal_id', 'fundamental_downside', 'timing_improvement',
                                                       'solutions_to_risk', 'acquirer_quality'))
    merge_df['id_y'] = merge_df['id_y'].fillna(0).astype(int)
    mna_factors_df['deal_id'] = mna_factors_df['deal_id'].fillna(0).astype(int)
    merge_df = pd.merge(merge_df, mna_factors_df, left_on=['id_y'], right_on=['deal_id'], how='left')
    merge_df = merge_df.sort_values(by=['ann_ror'], ascending=False)
    ranking_cols = ['fundamental_downside', 'timing_improvement', 'solutions_to_risk', 'acquirer_quality']
    merge_df[ranking_cols] = merge_df[ranking_cols].astype(float)
    merge_df['ranking_score'] = round(1e2 * merge_df[ranking_cols].sum(axis=1) / 12.00, 2)
    merge_df['ranking_score'] = merge_df['ranking_score'].astype(str) + ' %'
    merge_df[ranking_cols] = merge_df[ranking_cols].replace({np.nan: '-', 1: 'Low - 1', 2: 'Med - 2', 3: 'High - 3'})
    merge_df['closing_date'] = pd.to_datetime(merge_df['closing_date'], errors='coerce')
    merge_df['closing_date'] = merge_df['closing_date'].dt.strftime('%Y-%m-%d')
    merge_df.rename(columns={'id_x': 'id'}, inplace=True)
    return merge_df


def download_hard_opt_data(request):
    as_of = HardFloatOptimization.objects.latest('date_updated').date_updated
    hard_opt_df = pd.DataFrame.from_records(HardFloatOptimization.objects.filter(date_updated=as_of)
                                            .order_by('-ann_ror').values())

    ranking_cols = ['fundamental_downside', 'timing_improvement', 'solutions_to_risk', 'acquirer_quality',
                    'ranking_score']
    pre_ranking_cols = ['tradegroup', 'deal_status', 'catalyst', 'catalyst_rating']
    post_ranking_cols = ['target_last_price', 'deal_value', 'all_in_spread', 'closing_date', 'days_to_close',
                         'gross_ror', 'ann_ror', 'risk_pct', 'risk_pct_unhedged', 'expected_vol',
                         'current_pct_of_float', 'firm_pct_float_mstrat_1x', 'firm_pct_float_mstrat_2x',
                         'arb_outlier_risk', 'arb_pct_of_aum', 'aed_outlier_risk', 'aed_pct_of_aum', 'rebal_multiples',
                         'rebal_target', 'weighted_gross_nav_potential', 'curr_rtn_wt_duration', 'curr_rwd_ror',
                         'lg_outlier_risk', 'lg_pct_of_aum', 'rebal_multiples_litman', 'rebal_target_litman',
                         'weighted_gross_nav_potential_litman', 'curr_rtn_wt_duration_litman',
                         'curr_rwd_ror_litman', 'notes', 'is_excluded']
    hard_opt_df = hard_opt_df[pre_ranking_cols + post_ranking_cols + ['id']]
    hard_opt_df = attach_ranking_cols_to_df(hard_opt_df)
    hard_opt_df.drop(columns=['id', 'id_y'], inplace=True)
    hard_opt_df = hard_opt_df[pre_ranking_cols + ranking_cols + post_ranking_cols]
    hard_opt_df.columns = ['TradeGroup', 'Deal Status', 'Catalyst', 'Cat. Rating', 'Fundamental Downside',
                           'Timing Improvement', 'Solutions To Risk', 'Acquirer Quality', 'Ranking Score', 'Tgt PX',
                           'Deal Value', 'Spread', 'Closing', 'DTC', 'Gross RoR', 'Ann. RoR', 'Risk(%)',
                           'Risk(%) Unhedged', 'Imp Vol', 'Firm Current (% of Float)', '% Float if Mstrats(1x)',
                           '% Float if Mstrats(2x)', 'ARB NAV Risk', 'ARB % AUM', 'AED NAV Risk', 'AED % AUM',
                           'Rebal Risk Mult', 'Rebal Target', 'Wgt Gross NAV Potential', 'Curr Rt Weight Duration',
                           'RWD RoR', 'LG NAV Risk', 'LG % AUM', 'Rebal Mult (LG)', 'Rebal Target (LG)',
                           '(LG) Wgt Gross NAV Potential', '(LG) Curr Rt Weight Duration', '(LG) RWD RoR', 'Notes',
                           'RoR Excluded']

    excel_file = IO()
    xlwriter = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    hard_opt_df.to_excel(xlwriter, 'HardOpt')
    xlwriter.save()
    xlwriter.close()
    excel_file.seek(0)
    response = HttpResponse(excel_file.read(),
                            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=HardOptimization-'+str(as_of)+'.xlsx'
    return response


def download_credit_hard_opt_data(request):
    as_of = CreditHardFloatOptimization.objects.latest('date_updated').date_updated
    credit_df = pd.DataFrame.from_records(CreditHardFloatOptimization.objects.filter(date_updated=as_of)
                                          .order_by('catalyst_rating', '-ann_ror').values())

    credit_df = credit_df[['tradegroup', 'target_ticker', 'sleeve', 'deal_status', 'catalyst', 'catalyst_rating',
                           'target_last_price', 'px_ask_price', 'deal_upside', 'coupon', 'closing_date',
                           'days_to_close', 'gross_ror', 'ann_ror', 'risk_pct_unhedged', 'expected_vol', 'nav_impact',
                           'pct_of_sleeve_current', 'm_strat_pct_aum', 'weighted_gross_nav_potential',
                           'non_excluded_pct_aum', 'curr_rtn_wt_duration', 'curr_rwd_ror', 'notes', 'is_excluded']]

    credit_df.columns = ['TradeGroup', 'Target Ticker', 'Sleeve', 'Deal Status', 'Catalyst', 'Cat. Rating', 'Tgt PX',
                         'PX Ask', 'Deal Upside', 'Coupon', 'Closing', 'DTC', 'Gross RoR', 'Ann. RoR',
                         'Risk(%) Unhedged', 'Imp Vol', 'TACO % AUM', 'M-strat Risk', 'M-strat % AUM',
                         'Wgt Gross NAV Potential', 'Non excluded % AUM', 'Curr Rt Weight Duration', 'Curr RWD RoR',
                         'Notes', 'RoR Excluded']

    excel_file = IO()
    xlwriter = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    credit_df.to_excel(xlwriter, 'CreditHardOpt')
    xlwriter.save()
    xlwriter.close()
    excel_file.seek(0)
    response = HttpResponse(excel_file.read(),
                            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=CreditHardOptimization-'+str(as_of)+'.xlsx'
    return response


class CreditHardOptimizationView(ListView):
    template_name = 'credit_hard_optimization.html'
    queryset = HardFloatOptimization.objects.filter(sleeve='Credit Opportunities', date_updated=Max('date_updated'))

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        as_of = self.request.GET.get('as_of')
        if as_of:
            as_of = datetime.datetime.strptime(as_of, '%Y-%m-%d')
        else:
            as_of = CreditHardFloatOptimization.objects.latest('date_updated').date_updated
        queryset = CreditHardFloatOptimization.objects.filter(sleeve='Credit Opportunities', date_updated=as_of)
        context.update({'credit_hard_opt_list': queryset.order_by('catalyst_rating', '-ann_ror'), 'as_of': as_of})
        return context


def save_hard_opt_comment(request):
    response = 'Failed'
    if request.method == 'POST':
        try:
            id = request.POST['id']
            note = request.POST['note']
            is_credit = request.POST.get('is_credit') or 'false'
            is_credit = is_credit == 'true'
            is_arb = request.POST.get('is_arb') or 'false'
            is_arb = is_arb == 'true'
            # get the Object
            if is_arb:
                deal_object = HardFloatOptimization.objects.get(id=id)
            elif is_credit:
                deal_object = CreditHardFloatOptimization.objects.get(id=id)
            deal_object.notes = note
            deal_object.save()
            response = 'Success'
        except Exception as e:
            print(e)

    return HttpResponse(response)


def save_exclusion_hard_opt(request):
    response = 'Failed'
    if request.method == 'POST':
        try:
            is_excluded = False
            id = request.POST['id']
            is_credit = request.POST.get('is_credit') or 'false'
            is_credit = is_credit == 'true'
            is_arb = request.POST.get('is_arb') or 'false'
            is_arb = is_arb == 'true'
            exclusion = request.POST['is_excluded']
            if exclusion.lower() == 'yes':
                is_excluded = True
            # get the Object
            if is_arb:
                deal_object = HardFloatOptimization.objects.get(id=id)
            elif is_credit:
                deal_object = CreditHardFloatOptimization.objects.get(id=id)
            deal_object.is_excluded = is_excluded
            deal_object.save()
            response = 'Success'
        except Exception as e:
            print(e)

    return HttpResponse(response)


def save_rebal_paramaters(request):
    response = 'Failed'
    if request.method == 'POST':
        try:
            id = request.POST['id']
            rebal_multiple = request.POST['rebal_multiple']
            rebal_target = request.POST['rebal_target']
            fund = request.POST.get('fund')
            is_credit = request.POST.get('is_credit') or 'false'
            is_credit = is_credit == 'true'
            is_arb = request.POST.get('is_arb') or 'false'
            is_arb = is_arb == 'true'
            # get the Object
            if is_arb:
                deal_object = HardFloatOptimization.objects.get(id=id)
                if fund and fund.lower() == 'aed':
                    deal_object.rebal_multiples = rebal_multiple if rebal_multiple else None
                    deal_object.rebal_target = rebal_target if rebal_target else deal_object.aed_aum_mult
                else:
                    deal_object.rebal_multiples_litman = rebal_multiple if rebal_multiple else None
                    deal_object.rebal_target_litman = rebal_target if rebal_target else deal_object.aed_aum_mult
            elif is_credit:
                deal_object = CreditHardFloatOptimization.objects.get(id=id)
                deal_object.rebal_multiples = rebal_multiple if rebal_multiple else None
                deal_object.rebal_target = rebal_target if rebal_target else deal_object.m_strat_pct_aum
            deal_object.save()
            response = 'Success'
        except Exception as e:
            print(e)

    return HttpResponse(response)


def rebalance_portfolio(request):
    task_id = 'Error'
    if request.method == 'POST':
        try:
            is_credit = request.POST.get('is_credit') or 'false'
            is_credit = is_credit == 'true'
            is_arb = request.POST.get('is_arb') or 'false'
            is_arb = is_arb == 'true'
            if is_arb:
                task = arb_hard_float_optimization.delay(post_to_slack=False, record_progress=True)
                task_id = task.task_id
            elif is_credit:
                task = credit_hard_float_optimization.delay(post_to_slack=False, record_progress=True)
                task_id = task.task_id
        except Exception as exception:
            print(exception)

    return JsonResponse({'task_id': task_id})


class PnlPotentialView(ListView):
    template_name = 'pnl_potential.html'
    queryset = PnlPotentialDate.objects.all()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        merger_df = pd.DataFrame()
        credit_df = pd.DataFrame()
        as_of_date = self.request.GET.get('as_of')
        df1, df2, df3, df4 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        aed_df, aed_credit_df = pd.DataFrame(), pd.DataFrame()
        date_queryset, exclusions_queryset = pd.DataFrame(), pd.DataFrame()
        scenario_queryset, incremental_queryset = pd.DataFrame(), pd.DataFrame()
        empty_data = False
        ess_required_return = 8
        credit_mktval_updated, aed_mktval_updated = False, False
        if as_of_date:
            show_edit_buttons = False
            summary_df = pd.DataFrame.from_records(PnlPotentialDailySummary.objects.filter(date_updated=as_of_date).values())
            if not summary_df.empty:
                date_queryset = PnlPotentialDateHistory.objects.filter(date_updated=as_of_date).order_by(
                    '-sleeve', 'start_date')
                exclusions_queryset = PnlPotentialExclusionsHistory.objects.filter(date_updated=as_of_date).order_by(
                    '-sleeve', 'deal_name')
                scenario_queryset = PnlPotentialScenariosHistory.objects.filter(date_updated=as_of_date).order_by(
                    'scenario_name', 'date_deal_name', '-sleeve')
                incremental_queryset = PnlPotentialIncrementalHistory.objects.filter(date_updated=as_of_date)
                df1 = pivot_table(summary_df[summary_df['sleeve'].isin(['Merger Arbitrage', 'Credit Opportunities'])])
                df2 = pivot_table(summary_df[summary_df['sleeve'].isin(['Merger Arbitrage + Credit Opportunities'])])
                df3 = pivot_table(summary_df[summary_df['sleeve'].isin(['ESS Capital'])])
                df4 = pd.DataFrame()
                aed_df = pd.DataFrame()
                aed_credit_df = pd.DataFrame()
                if not df1.empty:
                    merger_df = df1[df1['sleeve'] == 'Merger Arbitrage']
                    credit_df = df1[df1['sleeve'] == 'Credit Opportunities']
            else:
                empty_data = True
        else:
            show_edit_buttons = True
            date_queryset = PnlPotentialDate.objects.all().order_by('-sleeve', 'start_date')
            exclusions_queryset = PnlPotentialExclusions.objects.all().order_by('-sleeve', 'deal_name')
            scenario_queryset = PnlPotentialScenarios.objects.all().order_by('scenario_name', 'date_deal_name',
                                                                             '-sleeve')
            incremental_queryset = PnlPotentialIncremental.objects.all()
            df_dict, dict_values = format_data()
            df1 = df_dict.get('scenario_response_df')
            df2 = df_dict.get('scenario_processing_df')
            df3 = df_dict.get('ess_achievement_returns_df')
            df4 = df_dict.get('aed_ess_df')
            aed_df = df_dict.get('aed_df')
            aed_credit_df = df_dict.get('aed_credit_df')
            ess_required_return = dict_values.get('ess_required_return')
            implied_prob_deduct = dict_values.get('implied_prob_deduct')
            if not aed_credit_df.empty and True in aed_credit_df['Is MktVal updated'].unique():
                credit_mktval_updated = True
            if not aed_df.empty and True in aed_df['Is MktVal updated'].unique():
                aed_mktval_updated = True
            if not df1.empty:
                merger_df = df1[df1['sleeve'] == 'Merger Arbitrage']
                credit_df = df1[df1['sleeve'] == 'Credit Opportunities']
            else:
                empty_data = True
        context.update({'date_queryset': date_queryset, 'exclusions_queryset': exclusions_queryset,
                        'incremental_queryset': incremental_queryset, 'scenario_queryset': scenario_queryset,
                        'merger_df': merger_df.to_json(orient='records'), 'aed_df': aed_df.to_json(orient='records'),
                        'credit_df': credit_df.to_json(orient='records'), 'df2': df2.to_json(orient='records'),
                        'df3': df3.to_json(orient='records'),
                        'aed_credit_df': aed_credit_df.to_json(orient='records'), 'as_of_date': as_of_date,
                        'show_edit_buttons': show_edit_buttons, 'empty_data': empty_data,
                        'credit_mktval_updated': credit_mktval_updated, 'aed_mktval_updated': aed_mktval_updated,
                        'dict_values': json.dumps(dict_values)})
        return context


def get_aed_sleeve_perf_view(request):
    sleeve_perf_dollar, sleeve_perf_bps = pd.DataFrame(), pd.DataFrame()
    if request.method == 'GET':
        sleeve_perf_dollar, sleeve_perf_bps, loss_budget_df = get_aed_sleeves_performance()

    return JsonResponse({'sleeve_perf_dollar': sleeve_perf_dollar.to_json(orient='records'),
                         'sleeve_perf_bps': sleeve_perf_bps.to_json(orient='records'),
                         'loss_budget_df': loss_budget_df.to_json(orient='records'),
                         'msg': 'success'})


def pivot_table(df):
    df.drop(columns=['date_updated', 'id'], inplace=True)
    df.rename(columns={'cut_name': 'Index'}, inplace=True)
    df = pd.pivot_table(df, index=['sleeve', 'Index'], columns=['scenario_name'], values='value')
    df.columns.name = ''
    df = df.reset_index()
    return df


def sync_ess_drilldown_with_ess_idea_db(request):
    """ View to Synchronize the P&L Potential ESS Constituents with the ESS IDEA DB """
    response = 'failed'
    if request.method == 'POST':

        try:
            update_pnl_ess_constituents()
            response = 'success'
        except Exception as e:
            print(e)

    return HttpResponse(response)


def save_pnl_potential_new_date(request):
    response = None
    if request.method == 'POST':
        data = request.POST
        start_date = data.get('start_date')
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = data.get('end_date')
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        date_name = data.get('date_name')
        sleeves = data.get('sleeve')
        if sleeves and sleeves.lower() == 'all sleeves':
            sleeves = ['Merger Arbitrage', 'Credit Opportunities']
        else:
            sleeves = [sleeves]
        scenario_names = PnlPotentialScenarios.objects.all().values('scenario_name')
        scenario_names = [i['scenario_name'] for i in scenario_names]
        scenario_names = set(scenario_names)
        for sleeve in sleeves:
            PnlPotentialDate.objects.create(start_date=start_date, end_date=end_date, date_name=date_name,
                                            sleeve=sleeve)
            for scenario_name in scenario_names:
                PnlPotentialScenarios.objects.create(scenario_name=scenario_name, sleeve=sleeve,
                                                     date_deal_name=date_name, scenario_value=100)
        response = 'success'
    return HttpResponse(response)


def save_pnl_potential_exclusion_deal(request):
    response = None
    if request.method == 'POST':
        data = request.POST
        deal_name = data.get('deal_name').upper()
        sleeves = data.get('sleeve')
        if sleeves and sleeves.lower() == 'all sleeves':
            sleeves = ['Merger Arbitrage', 'Credit Opportunities']
        else:
            sleeves = [sleeves]
        scenario_names = PnlPotentialScenarios.objects.all().values('scenario_name')
        scenario_names = [i['scenario_name'] for i in scenario_names]
        scenario_names = set(scenario_names)
        for sleeve in sleeves:
            PnlPotentialExclusions.objects.create(deal_name=deal_name, sleeve=sleeve)
            for scenario_name in scenario_names:
                PnlPotentialScenarios.objects.create(scenario_name=scenario_name, sleeve=sleeve,
                                                     date_deal_name=deal_name, scenario_value=100)
        response = 'success'
    return HttpResponse(response)


def get_deal_names(request):
    deal_names = []
    if request.method == 'GET':
        engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" + settings.WICFUNDS_DATABASE_PASSWORD
                               + "@" + settings.WICFUNDS_DATABASE_HOST + "/" + settings.WICFUNDS_DATABASE_NAME)
        con = engine.connect()
        deal_names_query = "SELECT distinct TradeGroup FROM wic.daily_flat_file_db WHERE Flat_file_as_of = (SELECT " \
                           "MAX(Flat_file_as_of) FROM wic.daily_flat_file_db);"
        deal_names = con.execute(deal_names_query)
        deal_names = [i[0].upper() if i[0] else i[0] for i in deal_names]
        if None in deal_names:
            deal_names.remove(None)
        deal_names = sorted(deal_names)
    return JsonResponse({'deal_names': deal_names})


def save_required_return(request):
    """
    Deletes all the required_return from the table and save a new required_return in the table.
    PnlPotentialOtherValues will always contain a single row with required_return and date_updated
    """
    response = 'failed'
    if request.method == 'POST':
        required_return = request.POST.get('required_return') or 8
        implied_prob = request.POST.get('implied_prob_deduction') or 5
        date_updated = datetime.datetime.now()
        obj_list = []
        obj_list.append(PnlPotentialOtherValues(date_updated=date_updated, field_name='implied_prob',
                                                field_value=implied_prob))
        obj_list.append(PnlPotentialOtherValues(date_updated=date_updated, field_name='required_return',
                                                field_value=required_return))
        PnlPotentialOtherValues.objects.all().delete()
        PnlPotentialOtherValues.objects.bulk_create(obj_list)
        response = 'success'
    return HttpResponse(response)


def save_pnl_potential_scenario(request):
    response = None
    if request.method == 'POST':
        data = request.POST
        scenario_name = data.get('scenario_name').upper()
        engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" +
                               settings.WICFUNDS_DATABASE_PASSWORD + "@" + settings.WICFUNDS_DATABASE_HOST +
                               "/" + settings.WICFUNDS_DATABASE_NAME)
        con = engine.connect()
        for key in data:
            if key == 'ess_scenario':
                key_data = json.loads(data[key])
                aed_ess_df = get_ess_df(new_scenario=True, scenario_name=scenario_name)
                aed_ess_df.drop(columns=['sleeve'], inplace=True)
                aed_ess_df.rename(columns={'AED NAV': 'aed_nav', 'CurrentMktVal_Pct': 'current_mkt_val_pct',
                                           'PX_LAST':'px_last', 'Pnl Potential 100%': 'pnl_potential_100',
                                           'Pnl Potential 50%': 'pnl_potential_50',
                                           'Pnl Potential 0%': 'pnl_potential_0'}, inplace=True)
                aed_ess_df['up_probability'] = 100
                aed_ess_df['down_probability'] = 100
                aed_ess_df['upside_field'] = 'model_up'
                aed_ess_df['downside_field'] = 'model_down'
                aed_ess_df['downside_value'] = aed_ess_df['model_down']
                aed_ess_df['upside_value'] = aed_ess_df['model_up']
                aed_ess_df['scenario_name'] = scenario_name
                for ess_key in key_data:
                    aed_ess_df['scenario_type'] = ess_key
                    aed_ess_df.to_sql(con=con, name='portfolio_optimization_pnlpotentialessconstituents', index=False,
                                      schema=settings.CURRENT_DATABASE, if_exists='append')
            elif key != 'scenario_name':
                key_data = json.loads(data[key])
                date_deal_name = key_data.get('date_deal_name').upper()
                scenario_value = key_data.get('scenario_value')
                sleeve = key_data.get('sleeve')
                PnlPotentialScenarios.objects.create(scenario_name=scenario_name, date_deal_name=date_deal_name,
                                                     scenario_value=scenario_value, sleeve=sleeve)

        con.close()
        response = 'success'
    return HttpResponse(response)


@transaction.atomic
def save_pnl_potential_ess_scenario(request):
    response = 'failed'
    if request.method == 'POST':
        data = request.POST.copy()
        scenario_name = data.get('scenario_name')
        scenario_type = data.get('scenario_type')
        del data['scenario_name']
        del data['scenario_type']
        objs = PnlPotentialESSConstituents.objects.filter(scenario_name=scenario_name, scenario_type=scenario_type)
        if len(objs) != len(data.keys()):
            return HttpResponse('mismatch')
        for key in data.keys():
            key_data = json.loads(data[key])
            try:
                with transaction.atomic():
                    obj = objs.get(id=key_data['row_id'])
                    upside_value = float(key_data['up_value'])
                    downside_value = float(key_data['down_value'])
                    px_last = obj.px_last or 0
                    obj.up_probability = key_data['up_pct']
                    obj.down_probability = key_data['down_pct']
                    obj.upside_field = key_data['up_select']
                    obj.downside_field = key_data['down_select']
                    obj.upside_value = upside_value
                    obj.downside_value = downside_value
                    current_mkt_val_pct = abs(obj.current_mkt_val_pct)
                    is_customized = key_data['custom_checkbox']
                    custom_mkt_val = key_data['customized_mkt_val']
                    obj.customized_mkt_val_pct = current_mkt_val_pct
                    obj.is_customized = is_customized
                    mkt_val = current_mkt_val_pct
                    if is_customized:
                        obj.customized_mkt_val_pct = custom_mkt_val
                        mkt_val = custom_mkt_val
                    aed_nav = obj.aed_nav
                    try:
                        obj.pnl_potential_100 = ((upside_value / px_last) - 1) * mkt_val / 100 * aed_nav
                        obj.pnl_potential_50 = (((((obj.model_up - obj.model_down) * 0.5) + obj.model_down) /
                                                 px_last) - 1) * mkt_val / 100 * aed_nav
                        obj.pnl_potential_0 = ((downside_value / px_last) - 1) * mkt_val / 100 * aed_nav
                    except ZeroDivisionError:
                        obj.pnl_potential_100, obj.pnl_potential_50, obj.pnl_potential_0 = 0, 0, 0
                    obj.save()
            except PnlPotentialESSConstituents.DoesNotExist:
                return HttpResponse('failed')
        response = 'success'
    return HttpResponse(response)


@transaction.atomic
def save_pnl_potential_credit_arb_scenario(request):
    response = 'failed'
    if request.method == 'POST':
        data = request.POST.copy()
        sleeve = data.get('sleeve')
        del data['sleeve']
        objs = ArbCreditPnLPotentialDrilldown.objects.filter(sleeve=sleeve)
        if len(objs) != len(data.keys()):
            return HttpResponse('mismatch')
        with transaction.atomic():
            for key in data.keys():
                key_data = json.loads(data[key])
                try:
                    obj = objs.get(id=key)
                    is_customized = key_data['is_customized']
                    custom_mkt_val = key_data['custom_mkt_val']
                    current_mkt_val = obj.current_mkt_val_pct
                    obj.customized_mkt_val_pct = current_mkt_val
                    obj.is_customized = is_customized
                    mkt_val = current_mkt_val
                    if is_customized:
                        obj.customized_mkt_val_pct = custom_mkt_val
                        mkt_val = custom_mkt_val
                    obj.pnl_potential = float(mkt_val) * float(obj.gross_ror) * float(obj.aed_nav) * 0.0001
                    obj.save()
                except ArbCreditPnLPotentialDrilldown.DoesNotExist:
                    return HttpResponse('failed')
        response = 'success'
    return HttpResponse(response)


def update_pnl_potential_scenario(request):
    response = 'failed'
    if request.method == 'POST':
        data = request.POST.get('data')
        if data:
            data = json.loads(data)
        else:
            return HttpResponse('failed')
        row_id = data.get('row_id')
        scenario_value = data.get('scenario_value')
        try:
            obj = PnlPotentialScenarios.objects.get(id=row_id)
            obj.scenario_value = scenario_value
            obj.save()
            response = 'success'
        except PnlPotentialScenarios.DoesNotExist:
            response = 'failed'
    return HttpResponse(response)


def save_pnl_potential_incremental(request):
    response = 'failed'
    if request.method == 'POST':
        data = request.POST.get('incremental')
        if data:
            data = json.loads(data)
        else:
            return HttpResponse('failed')
        incremental_value = data.get('incremental_value')
        incremental_name = data.get('incremental_name').upper()
        sleeves = data.get('sleeve')
        if sleeves and sleeves.lower() == 'all sleeves':
            sleeves = ['Merger Arbitrage', 'Credit Opportunities']
        else:
            sleeves = [sleeves]
        scenario_names = PnlPotentialScenarios.objects.all().values('scenario_name')
        scenario_names = [i['scenario_name'] for i in scenario_names]
        scenario_names = set(scenario_names)
        for sleeve in sleeves:
            PnlPotentialIncremental.objects.create(incremental_name=incremental_name, sleeve=sleeve,
                                                   incremental_value=incremental_value)
            for scenario_name in scenario_names:
                PnlPotentialScenarios.objects.create(scenario_name=scenario_name, sleeve=sleeve,
                                                     date_deal_name=incremental_name, scenario_value=100)
        response = 'success'
    return HttpResponse(response)


def update_pnl_potential_incremental(request):
    response = 'failed'
    if request.method == 'POST':
        data = request.POST
        row_id = data.get('row_id')
        incremental_value = data.get('incremental_value')
        try:
            obj = PnlPotentialIncremental.objects.get(id=row_id)
            obj.incremental_value = incremental_value
            obj.save()
            response = 'success'
        except PnlPotentialIncremental.DoesNotExist:
            response = 'failed'
    return HttpResponse(response)


def get_scenario_detail(request):
    if request.method == 'GET':
        row_id = request.GET.get('row_id')
        is_ess = request.GET.get('is_ess') == 'true'
        try:
            obj = PnlPotentialScenarios.objects.get(id=row_id)
            scenario_name = obj.scenario_name
            date_deal_name = obj.date_deal_name
            data = {'scenario_name': scenario_name, 'date_deal_name': date_deal_name,
                    'scenario_value': obj.scenario_value, 'sleeve': obj.sleeve}
            if is_ess:
                ess_scenario_df = pd.DataFrame.from_records(PnlPotentialESSConstituents.objects.filter(
                    scenario_name=scenario_name, scenario_type=date_deal_name).values())
                if not ess_scenario_df.empty:
                    ess_scenario_df = ess_scenario_df[['tradegroup', 'alpha_ticker', 'aed_nav', 'current_mkt_val_pct',
                                                       'customized_mkt_val_pct', 'is_customized',
                                                       'pt_up', 'pt_wic', 'pt_down', 'model_up', 'model_wic',
                                                       'model_down', 'px_last', 'pnl_potential_100', 'pnl_potential_50',
                                                       'pnl_potential_0', 'up_probability', 'upside_field',
                                                       'upside_value', 'down_probability', 'downside_field',
                                                       'downside_value', 'id']]
                    ess_scenario_df.rename(
                        columns={'aed_nav': 'AED NAV', 'alpha_ticker': 'Alpha Ticker',
                                 'current_mkt_val_pct': 'Current MktValPct', 'is_customized': 'Is MktVal updated',
                                 'customized_mkt_val_pct': 'Customized MktValPct', 'down_probability': 'Down %',
                                 'downside_field': 'Downside Select', 'downside_value': 'Downside',
                                 'model_down': 'Model Down', 'model_up': 'Model UP', 'model_wic': 'Model WIC',
                                 'pnl_potential_0': 'PNL Potential 0%', 'pnl_potential_100': 'PNL Potential 100%',
                                 'pnl_potential_50': 'PNL Potential 50%', 'pt_down': 'PT Down', 'pt_up': 'PT UP',
                                 'pt_wic': 'PT WIC', 'px_last': 'PX LAST', 'scenario_name': 'Scenario Name',
                                 'scenario_type': 'Scenario Type', 'tradegroup': 'Tradegroup', 'up_probability': 'UP %',
                                 'upside_field': 'Upside Select', 'upside_value': 'Upside'}, inplace=True)
                    data['ess_scenario'] = ess_scenario_df.to_json(orient='records')
        except PnlPotentialScenarios.DoesNotExist:
            data = []
    return JsonResponse({'results': data})


def delete_scenario(request):
    response = 'failed'
    if request.method == 'POST':
        row_id = request.POST.get('row_id')
        scenario_name = request.POST.get('scenario_name')
        try:
            obj = PnlPotentialScenarios.objects.get(id=row_id)
            if scenario_name == obj.scenario_name:
                PnlPotentialScenarios.objects.filter(scenario_name=scenario_name).delete()
                PnlPotentialESSConstituents.objects.filter(scenario_name=scenario_name).delete()
                response = 'success'
        except PnlPotentialScenarios.DoesNotExist:
            response = 'failed'
    return HttpResponse(response)


def delete_pnl_potential_row(request):
    response = None
    if request.method == 'POST':
        data = request.POST
        row_id = data.get('row_id')
        is_deal = data.get('is_deal')
        is_deal = is_deal == 'true'
        is_date = data.get('is_date')
        is_date = is_date == 'true'
        is_incremental = data.get('is_incremental')
        is_incremental = is_incremental == 'true'
        try:
            scenario_sleeve, scenario_date_deal_name = None, None
            if is_deal:
                deal_obj = PnlPotentialExclusions.objects.get(id=row_id)
                scenario_sleeve = deal_obj.sleeve
                scenario_date_deal_name = deal_obj.deal_name
                deal_obj.delete()
                response = 'success'
            elif is_date:
                date_obj = PnlPotentialDate.objects.get(id=row_id)
                scenario_sleeve = date_obj.sleeve
                scenario_date_deal_name = date_obj.date_name
                date_obj.delete()
                response = 'success'
            elif is_incremental:
                obj = PnlPotentialIncremental.objects.get(id=row_id)
                scenario_sleeve = obj.sleeve
                scenario_date_deal_name = obj.incremental_name
                obj.delete()
                response = 'success'
            if scenario_sleeve and scenario_date_deal_name:
                PnlPotentialScenarios.objects.filter(sleeve=scenario_sleeve, date_deal_name=scenario_date_deal_name).delete()
        except PnlPotentialExclusions.DoesNotExist:
            response = 'failed'
        except PnlPotentialDate.DoesNotExist:
            response = 'failed'
        except PnlPotentialScenarios.DoesNotExist:
            response = 'failed'
    return HttpResponse(response)


def get_pnl_potential_scenarios(request):
    queryset = {}
    if request.method == 'GET':
        date_queryset = PnlPotentialDate.objects.extra(select={'name': 'date_name'}).values('name', 'sleeve')
        deal_queryset = PnlPotentialExclusions.objects.extra(select={'name': 'deal_name'}).values('name', 'sleeve')
        incremental_qs = PnlPotentialIncremental.objects.extra(select={'name': 'incremental_name'}).values('name', 'sleeve')
        queryset = chain(date_queryset, deal_queryset, incremental_qs)
    return JsonResponse({'results': list(queryset)})


def get_incremental_detail(request):
    if request.method == 'GET':
        queryset = PnlPotentialIncremental.objects.get(id=request.GET.get('row_id'))
        return JsonResponse({'sleeve': queryset.sleeve, 'incremental_value': queryset.incremental_value })
    return HttpResponse('failed')


def update_pnl_ess_constituents_view(request):
    """
    This function is also used in tasks for Pnl Potential.
    """
    response = update_pnl_ess_constituents_function(post_to_slack=False)
    return HttpResponse(response)
