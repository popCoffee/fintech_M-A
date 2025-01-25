import logging

import bbgclient
from datetime import date, datetime, timedelta
import json
import pandas as pd

from django.db import connection
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView

from credit_idea.forms import CreditIdeaForm, CreditIdeaCommentsForm
from credit_idea.models import (CreditIdea, CreditIdeaComments, CreditIdeaCreditDetails, CreditIdeaDetails,
                                CreditIdeaCreditScenario, CreditIdeaScenario, CreditIdeaCreditScenarioComments,
                                CreditStaticScreen)
from credit_idea.utils import (append_equity_to_ticker, calculate_number_of_days, convert_to_float_else_zero,
                               convert_to_str_decimal, round_decimal_fields, replace_boolean_fields)
from risk.models import MA_Deals
from risk_reporting.models import FormulaeBasedDownsides

ACQUIRER_PREMIUM = 30
NAV_PCT_IMPACT = -0.15
ACQ_PB_RATE = 0.40
TARGET_PB_RATE = 0.40
FACE_VALUE_OF_BONDS = 1000000
PROPOSED_RATIO = 5.00
BBG_FED_FUND_INDEX = 'FEDL01 INDEX'
logger = logging.getLogger(__name__)


def get_dict_from_flat_file(tradegroup_name, arb_underlying=None):
    query = "SELECT * FROM wic.daily_flat_file_db where Fund='ARB' and TradeGroup='" + \
            tradegroup_name.upper() + "' and Sleeve = 'Merger Arbitrage' and AlphaHedge = 'Alpha' order by " + \
            "Flat_file_as_of desc LIMIT 1;"
    flat_file_df = pd.read_sql_query(query, con=connection)
    try:
        formulaue_downside_values = FormulaeBasedDownsides.objects.filter(TradeGroup__iexact=tradegroup_name,
                                                                          TargetAcquirer__iexact='Target')
        if arb_underlying and formulaue_downside_values and len(formulaue_downside_values) > 1:
            formulaue_downside_values = formulaue_downside_values.filter(Underlying__icontains=arb_underlying)
        if formulaue_downside_values:
            formulaue_downside_values = formulaue_downside_values.first()
        else:
            raise FormulaeBasedDownsides.DoesNotExist
        topping_big_upside = convert_to_float_else_zero(formulaue_downside_values.DealValue)
        base_case_downside = convert_to_float_else_zero(formulaue_downside_values.base_case)
        outlier_downside = convert_to_float_else_zero(formulaue_downside_values.outlier)
        base_case_downside_type = formulaue_downside_values.BaseCaseDownsideType
        outlier_downside_type = formulaue_downside_values.OutlierDownsideType
    except FormulaeBasedDownsides.DoesNotExist:
        if not flat_file_df.empty:
            topping_big_upside = flat_file_df.at[flat_file_df.index[0], 'DealValue']
            base_case_downside = flat_file_df.at[flat_file_df.index[0], 'pm_base_case']
            outlier_downside = flat_file_df.at[flat_file_df.index[0], 'outlier']
            base_case_downside_type = ''
            outlier_downside_type = ''
        else:
            topping_big_upside = 0.00
            base_case_downside = 0.00
            outlier_downside = 0.00
            base_case_downside_type = ''
            outlier_downside_type = ''

    if not flat_file_df.empty:
        fund_assets = flat_file_df.at[flat_file_df.index[0], 'AUM']
        closing_date = flat_file_df.at[flat_file_df.index[0], 'ClosingDate']
        deal_value = flat_file_df.at[flat_file_df.index[0], 'DealValue']
        target_ticker = append_equity_to_ticker(flat_file_df.at[flat_file_df.index[0], 'Target_Ticker'])
        acq_ticker = append_equity_to_ticker(flat_file_df.at[flat_file_df.index[0], 'Hedge_Security'])
        cash_consideration = flat_file_df.at[flat_file_df.index[0], 'DealCashTerms']
        share_consideration = flat_file_df.at[flat_file_df.index[0], 'StockTerms']
        fx_local_to_base = flat_file_df.at[flat_file_df.index[0], 'FXCurrentLocalToBase']
        ccy = flat_file_df.at[flat_file_df.index[0], 'CCY']
        exp_target_dividend = convert_to_float_else_zero(flat_file_df.at[flat_file_df.index[0], 'ExpectedTargetDVDWIC'])
        no_target_dividend = convert_to_float_else_zero(flat_file_df.at[flat_file_df.index[0], 'NoTgtDVD'])
        exp_acq_dividend = convert_to_float_else_zero(flat_file_df.at[flat_file_df.index[0], 'ExpectedShortDVDWIC'])
        no_acq_dividend = convert_to_float_else_zero(flat_file_df.at[flat_file_df.index[0], 'NoAcqDVD'])
        target_dividend = exp_target_dividend * no_target_dividend
        acq_dividend = exp_acq_dividend * no_acq_dividend
    else:
        query = "SELECT * FROM wic.daily_flat_file_db where Fund = 'ARB' and Sleeve = 'Merger Arbitrage' and " + \
                "AlphaHedge = 'Alpha' order by Flat_file_as_of desc LIMIT 1;"
        flat_file_df = pd.read_sql_query(query, con=connection)
        if not flat_file_df.empty:
            fund_assets = flat_file_df.at[flat_file_df.index[0], 'AUM']
        else:
            fund_assets = 0.00
        closing_date = date.today()
        deal_value = 0.00
        target_ticker = ''
        acq_ticker = ''
        cash_consideration = 0.00
        share_consideration = 0.00
        fx_local_to_base = 1.00
        ccy = 'USD'
        target_dividend = 0.00
        acq_dividend = 0.00
    result_dict = {
        'topping_big_upside': topping_big_upside,
        'base_case_downside': base_case_downside,
        'outlier_downside': outlier_downside,
        'base_case_downside_type': base_case_downside_type,
        'outlier_downside_type': outlier_downside_type,
        'fund_assets': fund_assets,
        'closing_date': closing_date,
        'deal_value': deal_value,
        'target_ticker': target_ticker,
        'acq_ticker': acq_ticker,
        'cash_consideration': cash_consideration,
        'share_consideration': share_consideration,
        'fx_local_to_base': fx_local_to_base,
        'ccy': ccy,
        'target_dividend': target_dividend,
        'acq_dividend': acq_dividend
    }
    return result_dict


def create_new_credit_idea(new_credit_idea_dict):
    deal_bucket = new_credit_idea_dict.get('deal_bucket')
    deal_strategy_type = new_credit_idea_dict.get('deal_strategy_type')
    catalyst = new_credit_idea_dict.get('catalyst')
    catalyst_tier = new_credit_idea_dict.get('catalyst_tier')
    target_sec_cusip = new_credit_idea_dict.get('target_sec_cusip')
    coupon = new_credit_idea_dict.get('coupon')
    hedge_sec_cusip = new_credit_idea_dict.get('hedge_sec_cusip')
    analyst = new_credit_idea_dict.get('analyst')
    estimated_closing_date = new_credit_idea_dict.get('estimated_closing_date')
    upside_price = new_credit_idea_dict.get('upside_price')
    downside_price = new_credit_idea_dict.get('downside_price')
    arb_tradegroup = new_credit_idea_dict.get('arb_tradegroup')
    other_tradegroup = new_credit_idea_dict.get('other_tradegroup')
    deal_category = new_credit_idea_dict.get('deal_category')
    credit_idea = CreditIdea.objects.create(deal_bucket=deal_bucket, deal_strategy_type=deal_strategy_type,
                                            catalyst=catalyst, catalyst_tier=catalyst_tier,
                                            target_sec_cusip=target_sec_cusip, coupon=coupon,
                                            hedge_sec_cusip=hedge_sec_cusip, analyst=analyst,
                                            estimated_closing_date=estimated_closing_date,
                                            upside_price=upside_price, downside_price=downside_price,
                                            arb_tradegroup=arb_tradegroup, other_tradegroup=other_tradegroup,
                                            deal_category=deal_category)
    flat_file_dict = get_dict_from_flat_file(arb_tradegroup)
    topping_big_upside = flat_file_dict.get('topping_big_upside')
    base_case_downside = flat_file_dict.get('base_case_downside')
    outlier_downside = flat_file_dict.get('outlier_downside')
    base_case_downside_type = flat_file_dict.get('base_case_downside_type')
    outlier_downside_type = flat_file_dict.get('outlier_downside_type')
    fund_assets = flat_file_dict.get('fund_assets')
    closing_date = flat_file_dict.get('closing_date')
    deal_value = flat_file_dict.get('deal_value')
    target_ticker = flat_file_dict.get('target_ticker')
    acq_ticker = flat_file_dict.get('acq_ticker')
    cash_consideration = flat_file_dict.get('cash_consideration')
    share_consideration = flat_file_dict.get('share_consideration')
    target_dividend = flat_file_dict.get('target_dividend')
    acq_dividend = flat_file_dict.get('acq_dividend')

    CreditIdeaDetails.objects.create(target_ticker=target_ticker, topping_big_upside=topping_big_upside,
                                     base_case_downside=base_case_downside, outlier_downside=outlier_downside,
                                     acq_ticker=acq_ticker, cash_consideration=cash_consideration,
                                     share_consideration=share_consideration, deal_value=deal_value,
                                     target_dividend=target_dividend, acq_dividend=acq_dividend,
                                     fund_assets=fund_assets, float_so=0, acq_pb_rate=ACQ_PB_RATE,
                                     target_pb_rate=TARGET_PB_RATE, nav_pct_impact=NAV_PCT_IMPACT,
                                     base_case_downside_type=base_case_downside_type, credit_idea=credit_idea,
                                     outlier_downside_type=outlier_downside_type)
    CreditIdeaCreditDetails.objects.create(credit_idea=credit_idea, face_value_of_bonds=FACE_VALUE_OF_BONDS,
                                           proposed_ratio=PROPOSED_RATIO)
    scenario_keys = ['Earlier Date', 'Base Date', 'Worst Date']
    scenario_hedge_keys = ['Bonds Called (Redemption)', 'Change of Control (CoC)', 'No Deal (Base Case)',
                           'No Deal (Conservative Case)']
    scenario_is_deal_closed = ['Yes', 'Yes', 'No', 'No']
    scenario_redemption_type = ['Call Price', 'Change of Control', 'Base Break Price', 'Conservative Break Price']
    scenario_upside = [True, False, False, False]
    scenario_downside = [False, False, True, False]
    if closing_date and isinstance(closing_date, (date, datetime)):
        days_to_close = calculate_number_of_days(closing_date)
        date_list = [closing_date - timedelta(days=31), closing_date, closing_date + timedelta(days=31)]
    else:
        days_to_close = 0
        date_list = ['', '', '']
    for index, scenario in enumerate(scenario_keys):
        CreditIdeaScenario.objects.create(scenario=scenario, credit_idea=credit_idea,
                                          estimated_closing_date=date_list[index])
    for index, scenario in enumerate(scenario_hedge_keys):
        CreditIdeaCreditScenario.objects.create(scenario=scenario, credit_idea=credit_idea, is_hedge=False,
                                                returns_estimated_closing_date=closing_date,
                                                returns_days_to_close=days_to_close,
                                                is_deal_closed=scenario_is_deal_closed[index],
                                                bond_redemption_type=scenario_redemption_type[index],
                                                is_upside=scenario_upside[index], is_downside=scenario_downside[index])
        CreditIdeaCreditScenario.objects.create(scenario=scenario, credit_idea=credit_idea, is_hedge=True,
                                                returns_estimated_closing_date=closing_date,
                                                returns_days_to_close=days_to_close,
                                                is_deal_closed=scenario_is_deal_closed[index],
                                                bond_redemption_type=scenario_redemption_type[index],
                                                is_upside=scenario_upside[index], is_downside=scenario_downside[index])
        CreditIdeaCreditScenarioComments.objects.create(scenario=scenario, credit_idea=credit_idea, comments="")


class CreditIdeaView(FormView):
    """
    View for creating, editing and deleting Credit Idea.
    """
    template_name = 'credit_idea_db.html'
    form_class = CreditIdeaForm
    fields = '__all__'
    success_url = '#'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        queryset = CreditIdea.objects.all()
        credit_details_qs = CreditIdeaDetails.objects.values_list('target_ticker', 'credit_idea_id',
                                                                  'base_case_downside')
        tickers_list = [i[0] for i in credit_details_qs]
        tickers_list = [append_equity_to_ticker(i) for i in tickers_list]
        bbg_fields = ['PX_LAST']
        api_host = bbgclient.bbgclient.get_next_available_host()
        live_price = bbgclient.bbgclient.get_secid2field(tickers_list, 'tickers', bbg_fields, req_type='refdata',
                                                         api_host=api_host)
        equity_downside_dict = {}
        for item in credit_details_qs:
            ticker = append_equity_to_ticker(item[0])
            live_price_target_price = live_price.get(ticker)['PX_LAST']
            live_price_target_price = live_price_target_price[0] if live_price_target_price else 0
            target_price = convert_to_float_else_zero(live_price_target_price)
            arb = convert_to_float_else_zero(item[2])
            try:
                temp_equity_downside = (target_price - arb) / target_price * 100
            except ZeroDivisionError:
                temp_equity_downside = 0.00
            equity_downside_dict[item[1]] = temp_equity_downside
        for obj in queryset:
            try:
                obj_credit_details = CreditIdeaCreditDetails.objects.get(credit_idea_id=obj.id)
                security_field = obj_credit_details.bbg_security_name
                last_price = obj_credit_details.bbg_last_price
            except CreditIdeaCreditDetails.DoesNotExist:
                security_field = ''
                last_price = 0.00
            try:
                obj_scenarios = CreditIdeaCreditScenario.objects.filter(credit_idea_id=obj.id)
                closing_date = obj_scenarios.first().returns_estimated_closing_date
                upside_scenario = obj_scenarios.filter(is_upside=True)
                upside_price = upside_scenario.first().bond_redemption if upside_scenario.first() else 0.00
                downside_scenario = obj_scenarios.filter(is_downside=True)
                downside_price = downside_scenario.first().bond_redemption if downside_scenario.first() else 0.00
                unhedged_upside_scenario = upside_scenario.filter(is_hedge=False)
                unhedged_return = unhedged_upside_scenario.first().returns_annual_pct if unhedged_upside_scenario.first() else 0.00
                hedged_upside_scenario = upside_scenario.filter(is_hedge=True)
                hedged_return = hedged_upside_scenario.first().returns_annual_pct if hedged_upside_scenario.first() else 0.00
            except Exception:
                closing_date = ''
                upside_price = 0.00
                downside_price = 0.00
                unhedged_return = 0.00
                hedged_return = 0.00
            try:
                obj_details = CreditIdeaScenario.objects.filter(credit_idea_id=obj.id)
                base_scenario = obj_details.filter(scenario__icontains='base').first()
                equity_return = base_scenario.annual_pct if base_scenario else 0.00
            except Exception:
                equity_return = 0.00
            obj.security_field = security_field
            obj.last_price = convert_to_str_decimal(last_price, 3)
            obj.closing_date = closing_date
            if calculate_number_of_days(closing_date) <= 7:
                obj.flag_row = True
            else:
                obj.flag_row = False
            obj.upside_price = convert_to_str_decimal(upside_price, 3)
            obj.downside_price = convert_to_str_decimal(downside_price, 3)
            obj.unhedged_return = convert_to_str_decimal(unhedged_return, 3)
            obj.hedged_return = convert_to_str_decimal(hedged_return, 3)
            obj.equity_return = convert_to_str_decimal(equity_return, 3)
            obj.equity_downside = convert_to_str_decimal(equity_downside_dict.get(obj.id), 2)
        context['credit_idea_list'] = queryset

        return context

    def form_valid(self, form):
        data = form.cleaned_data
        idea_id_to_edit = self.request.POST.get('id')
        analyst = data.get('analyst')
        arb_tradegroup = data.get('arb_tradegroup')
        arb_tradegroup = arb_tradegroup.upper() if arb_tradegroup else arb_tradegroup
        other_tradegroup = data.get('other_tradegroup')
        other_tradegroup = other_tradegroup.upper() if other_tradegroup else ''
        other_tradegroup = '' if arb_tradegroup.lower() != 'other' else other_tradegroup
        deal_bucket = data.get('deal_bucket')
        deal_strategy_type = data.get('deal_strategy_type')
        catalyst = data.get('catalyst')
        catalyst_tier = data.get('catalyst_tier')
        target_sec_cusip = data.get('target_sec_cusip')
        coupon = data.get('coupon')
        hedge_sec_cusip = data.get('hedge_sec_cusip')
        estimated_closing_date = data.get('estimated_closing_date')
        upside_price = data.get('upside_price')
        downside_price = data.get('downside_price')
        comments = data.get('comments')
        deal_category = data.get('deal_category')
        create_new_idea = False if idea_id_to_edit else True
        if not create_new_idea:
            try:
                account_obj = CreditIdea.objects.get(id=idea_id_to_edit)
                account_obj.analyst = analyst
                account_obj.arb_tradegroup = arb_tradegroup
                account_obj.other_tradegroup = other_tradegroup
                account_obj.deal_bucket = deal_bucket
                account_obj.deal_strategy_type = deal_strategy_type
                account_obj.catalyst = catalyst
                account_obj.catalyst_tier = catalyst_tier
                account_obj.target_sec_cusip = target_sec_cusip
                account_obj.coupon = coupon
                account_obj.hedge_sec_cusip = hedge_sec_cusip
                account_obj.estimated_closing_date = estimated_closing_date
                account_obj.upside_price = upside_price
                account_obj.downside_price = downside_price
                account_obj.comments = comments
                account_obj.deal_category = deal_category
                account_obj.save()
                create_new_idea = False
            except CreditIdea.DoesNotExist:
                create_new_idea = True
        if create_new_idea:
            create_new_credit_idea(data)
        return super(CreditIdeaView, self).form_valid(form)


def get_credit_idea_details(request):
    """ Retreives all the details for the requested Credit IDEA """
    credit_idea_details = []
    if request.method == 'GET':
        credit_idea_id = request.GET.get('credit_idea_id')
        if credit_idea_id:
            try:
                credit_idea_details = {}
                credit_idea = CreditIdea.objects.get(id=credit_idea_id)
                credit_idea_details['analyst'] = credit_idea.analyst
                credit_idea_details['arb_tradegroup'] = credit_idea.arb_tradegroup
                credit_idea_details['other_tradegroup'] = credit_idea.other_tradegroup
                credit_idea_details['deal_bucket'] = credit_idea.deal_bucket
                credit_idea_details['deal_strategy_type'] = credit_idea.deal_strategy_type
                credit_idea_details['catalyst'] = credit_idea.catalyst
                credit_idea_details['catalyst_tier'] = credit_idea.catalyst_tier
                credit_idea_details['target_sec_cusip'] = credit_idea.target_sec_cusip
                credit_idea_details['coupon'] = credit_idea.coupon
                credit_idea_details['hedge_sec_cusip'] = credit_idea.hedge_sec_cusip
                credit_idea_details['estimated_closing_date'] = credit_idea.estimated_closing_date
                credit_idea_details['upside_price'] = credit_idea.upside_price
                credit_idea_details['downside_price'] = credit_idea.downside_price
                credit_idea_details['comments'] = credit_idea.comments
                credit_idea_details['deal_category'] = credit_idea.deal_category
            except CreditIdea.DoesNotExist:
                credit_idea_details = []

    return JsonResponse({'credit_idea_details': credit_idea_details})


def delete_credit_idea(request):
    response = None
    deal_category = None
    if request.method == 'POST':
        # Take the ID and Delete
        id_to_delete = request.POST['id']
        try:
            credit_idea = CreditIdea.objects.get(id=id_to_delete)
            deal_category = credit_idea.deal_category
            credit_idea.delete()
            response = 'credit_idea_deleted'
        except CreditIdea.DoesNotExist:
            response = 'refresh_window'

    return JsonResponse({'response': response, 'deal_category': deal_category})


class CreditIdeaCommentsView(FormView):
    """
    View for vieweing, editing, saving comments on Credit Idea.
    """
    template_name = 'credit_idea_comments.html'
    form_class = CreditIdeaCommentsForm
    fields = '__all__'
    success_url = '#'

    def get_initial(self):
        initial = super(CreditIdeaCommentsView, self).get_initial()
        credit_idea_id = self.request.GET.get('credit_idea_id')
        if credit_idea_id:
            comments_obj, created = CreditIdeaComments.objects.get_or_create(credit_idea_id=credit_idea_id)
            if not created:
                initial['summary_comments'] = comments_obj.summary_comments
                initial['press_release_comments'] = comments_obj.press_release_comments
                initial['proxy_statement_comments'] = comments_obj.proxy_statement_comments
                initial['change_of_control_comments'] = comments_obj.change_of_control_comments
                initial['restricted_payments_comments'] = comments_obj.restricted_payments_comments
                initial['liens_indebtedness_comments'] = comments_obj.liens_indebtedness_comments
                initial['other_comments'] = comments_obj.other_comments
        return initial

    def form_valid(self, form):
        data = form.cleaned_data
        credit_idea_id = self.request.GET.get('credit_idea_id')
        if credit_idea_id:
            comments_obj, created = CreditIdeaComments.objects.get_or_create(credit_idea_id=credit_idea_id)
            comments_obj.summary_comments = data.get('summary_comments')
            comments_obj.press_release_comments = data.get('press_release_comments')
            comments_obj.proxy_statement_comments = data.get('proxy_statement_comments')
            comments_obj.change_of_control_comments = data.get('change_of_control_comments')
            comments_obj.restricted_payments_comments = data.get('restricted_payments_comments')
            comments_obj.liens_indebtedness_comments = data.get('liens_indebtedness_comments')
            comments_obj.other_comments = data.get('other_comments')
            comments_obj.save()
        return super(CreditIdeaCommentsView, self).form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        credit_idea_id = self.request.GET.get('credit_idea_id')
        if credit_idea_id:
            try:
                credit_idea = CreditIdea.objects.get(id=credit_idea_id)
                context.update({
                    'arb_tradegroup': credit_idea.arb_tradegroup,
                    'other_tradegroup': credit_idea.other_tradegroup
                })
            except CreditIdea.DoesNotExist:
                return context
        return context


class CreditIdeaDetailsView(TemplateView):
    template_name = 'view_credit_idea.html'

    def post(self, request, *args, **kwargs):
        try:
            context = self.get_context_data()
            credit_idea_id = request.POST.get('credit_idea_id')
            master_data = json.loads(request.POST.get('master_data'))
            response = save_credit_idea_data(context, master_data, credit_idea_id)
        except Exception:
            response = 'failed'
        return HttpResponse(response)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        credit_idea_id = self.request.GET.get('credit_idea_id')
        if credit_idea_id:
            try:
                credit_idea_object = CreditIdea.objects.get(id=credit_idea_id)
                arb_tradegroup = credit_idea_object.arb_tradegroup
                other_tradegroup = credit_idea_object.other_tradegroup
                if arb_tradegroup.lower() != 'other':
                    flat_file_dict = get_dict_from_flat_file(arb_tradegroup)
                    flat_file_closing_date = flat_file_dict.get('closing_date')
                else:
                    flat_file_dict = {}
                    flat_file_closing_date = ''

                credit_idea_scenarios = CreditIdeaScenario.objects.filter(credit_idea_id=credit_idea_id)
                base_scenario = credit_idea_scenarios.filter(scenario='Base Date').first()
                if base_scenario:
                    if arb_tradegroup.lower() != 'other':
                        estimated_closing_date = flat_file_closing_date
                    else:
                        estimated_closing_date = base_scenario.estimated_closing_date
                    if isinstance(estimated_closing_date, (date, datetime)):
                        estimated_closing_date = estimated_closing_date.strftime('%m/%d/%Y')
                    base_rebate = convert_to_float_else_zero(base_scenario.rebate)
                    base_spread = convert_to_float_else_zero(base_scenario.spread)
                    base_days_to_close = calculate_number_of_days(base_scenario.estimated_closing_date)
                else:
                    estimated_closing_date = ''
                    base_rebate = 0.00
                    base_spread = 0.00
                    base_days_to_close = 0
            except CreditIdea.DoesNotExist:
                raise Http404('Credit Idea not available')

            try:
                credit_idea_details = CreditIdeaDetails.objects.get(credit_idea_id=credit_idea_id)
                if arb_tradegroup.lower() != 'other':
                    arb_base_case = convert_to_float_else_zero(flat_file_dict.get('base_case_downside')) or 0.00
                    deal_value = convert_to_float_else_zero(flat_file_dict.get('deal_value')) or 0.00
                    arb_outlier = convert_to_float_else_zero(flat_file_dict.get('outlier_downside')) or 0.00
                    target_ticker = append_equity_to_ticker(flat_file_dict.get('target_ticker'))
                    if not target_ticker:
                        target_ticker = append_equity_to_ticker(credit_idea_details.target_ticker)
                    acq_ticker = append_equity_to_ticker(flat_file_dict.get('acq_ticker'))
                    if not acq_ticker:
                        acq_ticker = append_equity_to_ticker(credit_idea_details.acq_ticker)
                    cash_terms = convert_to_float_else_zero(flat_file_dict.get('cash_consideration'))
                    share_terms = convert_to_float_else_zero(flat_file_dict.get('share_consideration'))
                    fx_local_to_base = flat_file_dict.get('fx_local_to_base') or 1.00
                    ccy = flat_file_dict.get('ccy') or 'USD'
                    target_dividends = convert_to_float_else_zero(flat_file_dict.get('target_dividend'))
                    acq_dividends = convert_to_float_else_zero(flat_file_dict.get('acq_dividend'))
                    fund_assets = flat_file_dict.get('fund_assets')
                else:
                    arb_base_case = convert_to_float_else_zero(credit_idea_details.base_case_downside) or 0.00
                    deal_value = convert_to_float_else_zero(credit_idea_details.deal_value) or 0.00
                    arb_outlier = convert_to_float_else_zero(credit_idea_details.outlier_downside) or 0.00
                    target_ticker = append_equity_to_ticker(credit_idea_details.target_ticker)
                    acq_ticker = append_equity_to_ticker(credit_idea_details.acq_ticker)
                    cash_terms = convert_to_float_else_zero(credit_idea_details.cash_consideration)
                    share_terms = convert_to_float_else_zero(credit_idea_details.share_consideration)
                    fx_local_to_base = 1.00
                    ccy = 'USD'
                    target_dividends = convert_to_float_else_zero(credit_idea_details.target_dividend)
                    acq_dividends = convert_to_float_else_zero(credit_idea_details.acq_dividend)
                    fund_assets = credit_idea_details.fund_assets

                arb_base_case_downside_type = credit_idea_details.base_case_downside_type
                arb_outlier_downside_type = credit_idea_details.outlier_downside_type
                acq_pb_rate = convert_to_float_else_zero(credit_idea_details.acq_pb_rate)
                target_pb_rate = convert_to_float_else_zero(credit_idea_details.target_pb_rate)
                float_so_value = credit_idea_details.float_so
                nav_pct_impact = convert_to_float_else_zero(credit_idea_details.nav_pct_impact)
            except CreditIdeaDetails.DoesNotExist:
                arb_base_case = 0.00
                deal_value = 0.00
                arb_outlier = 0.00
                arb_base_case_downside_type = ''
                arb_outlier_downside_type = ''
                target_ticker = ''
                acq_ticker = ''
                cash_terms = 0.00
                share_terms = 0.00
                fx_local_to_base = 1.00
                ccy = 'USD'
                target_dividends = 0.00
                acq_dividends = 0.00
                acq_pb_rate = 0.00
                target_pb_rate = 0.00
                fund_assets = 0.00
                float_so_value = 0.00
                nav_pct_impact = 0.00

            try:
                credit_idea_creditdetails = CreditIdeaCreditDetails.objects.get(credit_idea_id=credit_idea_id)
                bond_ticker = credit_idea_creditdetails.bond_ticker or ''
                bbg_security_name = credit_idea_creditdetails.bbg_security_name or ''
                face_value_of_bonds = convert_to_float_else_zero(credit_idea_creditdetails.face_value_of_bonds)
                bond_est_purchase_price = convert_to_float_else_zero(credit_idea_creditdetails.bond_est_purchase_price)
                bbg_est_daily_vol = convert_to_float_else_zero(credit_idea_creditdetails.bbg_est_daily_vol)
                bbg_actual_thirty_day = convert_to_float_else_zero(credit_idea_creditdetails.bbg_actual_thirty_day)
                credit_team_view = credit_idea_creditdetails.credit_team_view
                credit_team_view = int(credit_team_view) if credit_team_view else credit_team_view
                base_break_price = convert_to_float_else_zero(credit_idea_creditdetails.base_break_price)
                conservative_break_price = convert_to_float_else_zero(
                    credit_idea_creditdetails.conservative_break_price)
                call_price = convert_to_float_else_zero(credit_idea_creditdetails.call_price)
                make_whole_price = convert_to_float_else_zero(credit_idea_creditdetails.make_whole_price)
                equity_claw_percent = convert_to_float_else_zero(credit_idea_creditdetails.equity_claw_percent)
                equity_claw_value = convert_to_float_else_zero(credit_idea_creditdetails.equity_claw_value)
                blend = convert_to_float_else_zero(credit_idea_creditdetails.blend)
                change_of_control = convert_to_float_else_zero(credit_idea_creditdetails.change_of_control)
                acq_credit = convert_to_float_else_zero(credit_idea_creditdetails.acq_credit)
                other_acq_credit = convert_to_float_else_zero(credit_idea_creditdetails.other_acq_credit)
                proposed_ratio = convert_to_float_else_zero(credit_idea_creditdetails.proposed_ratio)
                break_spread = convert_to_float_else_zero(credit_idea_creditdetails.break_spread)
            except CreditIdeaCreditDetails.DoesNotExist:
                bond_ticker = ''
                bbg_security_name = ''
                face_value_of_bonds = 0.00
                bond_est_purchase_price = 0.00
                bbg_est_daily_vol = 0.00
                bbg_actual_thirty_day = 0.00
                credit_team_view = 1
                base_break_price = 0.00
                conservative_break_price = 0.00
                call_price = 0.00
                make_whole_price = 0.00
                equity_claw_percent = 0.00
                equity_claw_value = 0.00
                blend = 0.00
                change_of_control = 0.00
                acq_credit = 0.00
                other_acq_credit = 0.00
                proposed_ratio = 0.00
                break_spread = 0.00

            credit_idea_creditscenario = CreditIdeaCreditScenario.objects.filter(credit_idea_id=credit_idea_id)
            scenario_with_hedge = credit_idea_creditscenario.filter(is_hedge=True)
            scenario_without_hedge = credit_idea_creditscenario.filter(is_hedge=False)

            credit_idea_creditcomments = CreditIdeaCreditScenarioComments.objects.filter(credit_idea_id=credit_idea_id)
            try:
                api_host = bbgclient.bbgclient.get_next_available_host()
                bbg_target_ticker = append_equity_to_ticker(target_ticker)
                bbg_acq_ticker = append_equity_to_ticker(acq_ticker)
                tickers_live_price = [bbg_target_ticker, BBG_FED_FUND_INDEX, bond_ticker, bbg_acq_ticker]

                bbg_fields = ['PX_LAST', 'DVD_SH_LAST', 'SECURITY_NAME', 'COUPON', 'AMT_OUTSTANDING', 'PX_BID',
                              'PX_ASK']
                live_price = bbgclient.bbgclient.get_secid2field(tickers_live_price, 'tickers',
                                                                 bbg_fields, req_type='refdata', api_host=api_host)

                target_live_price, acq_last_price, fed_funds_last_price = 0, 0, 0
                target_ticker_price = live_price.get(bbg_target_ticker)
                if target_ticker_price:
                    px_last_value = target_ticker_price.get('PX_LAST')
                    target_live_price = px_last_value[0] if len(px_last_value) > 0 else 0.00
                    target_live_price = convert_to_float_else_zero(target_live_price)
                    if 'ln' in bbg_target_ticker.lower().split(' '):
                        target_live_price = target_live_price * 0.01
                    dvd_last_value = target_ticker_price.get('DVD_SH_LAST')
                    tgt_dvd = dvd_last_value[0] if len(dvd_last_value) > 0 else 0.00
                    tgt_dvd = convert_to_float_else_zero(tgt_dvd)

                acq_ticker_price = live_price.get(bbg_acq_ticker)
                if acq_ticker_price:
                    px_last_value = acq_ticker_price.get('PX_LAST')
                    acq_last_price = px_last_value[0] if len(px_last_value) > 0 else 0.00
                    acq_last_price = convert_to_float_else_zero(acq_last_price)
                    if 'ln' in bbg_acq_ticker.lower().split(' '):
                        acq_last_price = acq_last_price * 0.01
                    dvd_last_value = target_ticker_price.get('DVD_SH_LAST')
                    acq_dvd = dvd_last_value[0] if len(dvd_last_value) > 0 else 0.00
                    acq_dvd = convert_to_float_else_zero(acq_dvd)

                fed_fund_price = live_price.get(BBG_FED_FUND_INDEX)
                if fed_fund_price:
                    px_last_value = fed_fund_price.get('PX_LAST')
                    fed_funds_last_price = px_last_value[0] if len(px_last_value) > 0 else 0.00
                    fed_funds_last_price = convert_to_float_else_zero(fed_funds_last_price)

                bond_ticker_price = live_price.get(bond_ticker)
                if bond_ticker_price:
                    if not bbg_security_name:
                        bbg_security_name = bond_ticker_price.get('SECURITY_NAME')
                        bbg_security_name = bbg_security_name[0] if len(bbg_security_name) > 0 else ''
                        bbg_security_name = bbg_security_name if bbg_security_name else ''
                    bbg_interest_rate = bond_ticker_price.get('COUPON')
                    bbg_interest_rate = convert_to_float_else_zero(bbg_interest_rate[0]) if len(
                        bbg_interest_rate) > 0 else 0.00
                    bbg_issue_size = bond_ticker_price.get('AMT_OUTSTANDING')
                    bbg_issue_size = convert_to_float_else_zero(bbg_issue_size[0]) / 1000000 if len(
                        bbg_issue_size) > 0 else 0.00
                    bbg_bid_price = bond_ticker_price.get('PX_BID')
                    bbg_bid_price = convert_to_float_else_zero(bbg_bid_price[0]) if len(bbg_bid_price) > 0 else 0.00
                    bbg_ask_price = bond_ticker_price.get('PX_ASK')
                    bbg_ask_price = convert_to_float_else_zero(bbg_ask_price[0]) if len(bbg_ask_price) > 0 else 0.00
                    bbg_last_price = bond_ticker_price.get('PX_LAST')
                    bbg_last_price = convert_to_float_else_zero(bbg_last_price[0]) if len(bbg_last_price) > 0 else 0.00

                else:
                    bbg_security_name = ''
                    bbg_interest_rate = 0.00
                    bbg_issue_size = 0.00
                    bbg_bid_price = 0.00
                    bbg_ask_price = 0.00
                    bbg_last_price = 0.00

                # if not convert_to_float_else_zero(bond_est_purchase_price):
                bond_est_purchase_price = bbg_last_price
                equity_claw = convert_to_float_else_zero(100 + bbg_interest_rate)
                if make_whole_price == 0:
                    blend = 0.00
                else:
                    blend = convert_to_float_else_zero(
                        equity_claw * equity_claw_percent * 0.01 + (1 - equity_claw_percent * 0.01) * make_whole_price)

                arb_spend = convert_to_float_else_zero(face_value_of_bonds * bond_est_purchase_price * 0.01)
                passive_phase_arb_data = [
                    {'id': 'arb_spend', 'type_input': 'false', 'key': 'Spend',
                     'value': convert_to_str_decimal(arb_spend, 0)},
                    {'id': 'face_value_of_bonds', 'type_input': 'true', 'key': 'Face Value of Bonds',
                     'value': convert_to_str_decimal(face_value_of_bonds, 0)},
                ]

                potential_outcomes_data = [
                    {'id': 'base_break_price', 'key': 'Base Break Price', 'type_input': 'false', 'type': '',
                     'type_input2': 'true', 'value': convert_to_str_decimal(base_break_price, 3)},
                    {'id': 'conservative_break_price', 'key': 'Conservative Break Price', 'type_input': 'false',
                     'type': '', 'type_input2': 'true', 'value': convert_to_str_decimal(conservative_break_price, 3)},
                    {'id': 'call_price', 'key': 'Call Price', 'type_input': 'false', 'type': '', 'type_input2': 'true',
                     'value': convert_to_str_decimal(call_price, 3)},
                    {'id': 'make_whole_price', 'key': 'Make-Whole Price', 'type_input': 'false', 'type': '',
                     'type_input2': 'true', 'value': convert_to_str_decimal(make_whole_price, 3)},
                    {'id': 'equity_claw_value', 'key': 'Equity Claw', 'type_input': 'true', 'type': equity_claw_percent,
                     'type_input2': 'false', 'value': convert_to_str_decimal(equity_claw, 3)},
                    {'id': 'blend', 'key': 'Blend', 'type_input': 'false', 'type': '', 'type_input2': 'true',
                     'type_input2': 'false', 'value': convert_to_str_decimal(blend, 3)},
                    {'id': 'change_of_control', 'key': 'Change of Control', 'type_input': 'false', 'type': '',
                     'type_input2': 'true', 'value': convert_to_str_decimal(change_of_control, 3)},
                    {'id': 'acq_credit', 'key': 'Other (1)', 'type_input': 'false', 'type': '', 'type_input2': 'true',
                     'value': convert_to_str_decimal(acq_credit, 3)},
                    {'id': 'other_acq_credit', 'key': 'Other (2)', 'type_input': 'false', 'type': '',
                     'type_input2': 'true',
                     'value': convert_to_str_decimal(other_acq_credit, 3)},
                ]

                estimated_liquidity_data = [
                    {'id': 'bbg_est_daily_vol', 'key': 'BBG Est. Daily Vol. ($)',
                     'value': convert_to_str_decimal(bbg_est_daily_vol)},
                    {'id': 'bbg_actual_thirty_day', 'key': 'BBG Actual 30-day ($)',
                     'value': convert_to_str_decimal(bbg_actual_thirty_day)},
                    {'id': 'credit_team_view', 'key': 'Credit Team View', 'value': credit_team_view},
                ]

                bond_information_data = [
                    {'id': 'bbg_security_name', 'type_input': 'true', 'key': 'Security', 'value': bbg_security_name},
                    {'id': 'bond_ticker', 'type_input': 'true', 'key': 'Bond Ticker', 'value': bond_ticker},
                    {'id': 'bbg_interest_rate', 'type_input': 'false', 'key': 'Interest Rate (%)',
                     'value': convert_to_str_decimal(bbg_interest_rate, 3)},
                    {'id': 'bbg_issue_size', 'type_input': 'false', 'key': 'Issue Size ($)',
                     'value': convert_to_str_decimal(bbg_issue_size)},
                ]

                bond_price_data = [
                    {'id': 'est_purchase_price', 'type_input': 'true', 'key': 'Est. Purchased Price',
                     'value': convert_to_str_decimal(bond_est_purchase_price, 3)},
                    {'id': 'bbg_bid_price', 'type_input': 'false', 'key': 'Bid Price',
                     'value': convert_to_str_decimal(bbg_bid_price, 3)},
                    {'id': 'bbg_ask_price', 'type_input': 'false', 'key': 'Ask Price',
                     'value': convert_to_str_decimal(bbg_ask_price, 3)},
                    {'id': 'bbg_last_price', 'type_input': 'false', 'key': 'Last Price',
                     'value': convert_to_str_decimal(bbg_last_price, 3)},
                ]

                hedge_in_dollars = arb_spend * proposed_ratio * 0.01
                if target_live_price != 0:
                    shares_target_shorts = hedge_in_dollars / (target_live_price * fx_local_to_base)
                else:
                    shares_target_shorts = 0.00
                acq_rebate_pct = fed_funds_last_price - acq_pb_rate
                target_rebate_pct = fed_funds_last_price - target_pb_rate
                less_short_rebate = target_live_price * fx_local_to_base * -1 * target_rebate_pct * base_days_to_close / 365 * 0.01
                short_spread = base_spread + base_rebate + less_short_rebate
                arb_spread = base_spread * fx_local_to_base
                less_rebate = base_rebate * fx_local_to_base * -1
                hedging_data = [
                    {'id': 'proposed_ratio', 'type_input': 'true', 'key': 'Proposed Ratio',
                     'value': convert_to_str_decimal(proposed_ratio)},
                    {'id': 'hedge', 'type_input': 'false', 'key': 'Hedge in $',
                     'value': convert_to_str_decimal(hedge_in_dollars, 0)},
                    {'id': 'target_short', 'type_input': 'false', 'key': 'Shares of Target Short',
                     'value': convert_to_str_decimal(shares_target_shorts, 0)},
                    {'id': 'arb_spread', 'type_input': 'false', 'key': 'Arb Spread',
                     'value': convert_to_str_decimal(arb_spread)},
                    {'id': 'less_rebate', 'type_input': 'false', 'key': 'Less: Rebate',
                     'value': convert_to_str_decimal(less_rebate)},
                    {'id': 'less_short_rebate', 'type_input': 'false', 'key': 'Less: Short Rebate',
                     'value': convert_to_str_decimal(less_short_rebate)},
                    {'id': 'short_spread', 'type_input': 'false', 'key': 'Short Spread',
                     'value': convert_to_str_decimal(short_spread)},
                    {'id': 'break_spread', 'type_input': 'true', 'key': 'Alternative Break Spread',
                     'value': convert_to_str_decimal(break_spread)},
                ]

                acquirer_30_premium = (1 + (ACQUIRER_PREMIUM * 0.01)) * acq_last_price
                topping_break_spread = ((acquirer_30_premium * share_terms) + cash_terms) - arb_base_case
                downsides_data = [
                    {'id': 'upside', 'key': 'Topping Bid Upside', 'type': 'Deal Value',
                     'value': convert_to_str_decimal(deal_value, 2),
                     'usd_val': convert_to_str_decimal(deal_value * fx_local_to_base, 2)},
                    {'id': 'base_downside', 'key': 'Base Case Downside', 'type': arb_base_case_downside_type,
                     'value': convert_to_str_decimal(arb_base_case, 2),
                     'usd_val': convert_to_str_decimal(arb_base_case * fx_local_to_base, 2)},
                    {'id': 'outlier_downside', 'key': 'Outlier Downside', 'type': arb_outlier_downside_type,
                     'value': convert_to_str_decimal(arb_outlier, 2),
                     'usd_val': convert_to_str_decimal(arb_outlier * fx_local_to_base, 2)},
                    {'id': 'thirty_premium', 'key': 'Acquirer 30% Premium', 'type': str(ACQUIRER_PREMIUM) + ' %',
                     'value': convert_to_str_decimal(acquirer_30_premium, 2),
                     'usd_val': convert_to_str_decimal(acquirer_30_premium * fx_local_to_base, 2)},
                    {'id': 'normal_spread', 'key': 'Normal Break Spread', 'type': 'Break Spread',
                     'value': convert_to_str_decimal(target_live_price - arb_base_case, 2),
                     'usd_val': convert_to_str_decimal((target_live_price - arb_base_case) * fx_local_to_base, 2)},
                    {'id': 'topping_spread', 'key': 'Topping Break Spread', 'type': 'Break Spread',
                     'value': convert_to_str_decimal(topping_break_spread, 2),
                     'usd_val': convert_to_str_decimal(topping_break_spread * fx_local_to_base, 2)}
                ]
                gross_spread = convert_to_float_else_zero(deal_value) - convert_to_float_else_zero(target_live_price)
                dvd_adjusted_spread = gross_spread + target_dividends
                dvd_adjusted_spread = convert_to_float_else_zero(
                    gross_spread + target_dividends - acq_dividends * share_terms)
                rebate_adjusted_spread = convert_to_float_else_zero(dvd_adjusted_spread + base_rebate)
                deal_value = convert_to_str_decimal(
                    convert_to_float_else_zero(acq_last_price) * convert_to_float_else_zero(
                        share_terms) + convert_to_float_else_zero(cash_terms), 2)
                spread_data = [
                    {'id': 'target_ticker', 'key': 'Target Ticker', 'type_input': 'true', 'type': target_ticker,
                     'type_input2': 'true', 'value': convert_to_str_decimal(target_live_price, 2),
                     'usd_val': convert_to_str_decimal(target_live_price * fx_local_to_base, 2)},
                    {'id': 'acq_ticker', 'key': 'Acq. Ticker (N/A for PE)', 'type_input': 'true', 'type': acq_ticker,
                     'type_input2': 'true', 'value': convert_to_str_decimal(acq_last_price, 2),
                     'usd_val': convert_to_str_decimal(acq_last_price * fx_local_to_base, 2)},
                    {'id': 'cash', 'key': 'Cash Consideration', 'type_input': 'false', 'type': '',
                     'type_input2': 'true', 'value': convert_to_str_decimal(cash_terms, 2),
                     'usd_val': convert_to_str_decimal(cash_terms * fx_local_to_base, 2)},
                    {'id': 'share', 'key': 'Share Considerations', 'type_input': 'false', 'type': '',
                     'type_input2': 'true', 'value': convert_to_str_decimal(share_terms, 4),
                     'usd_val': convert_to_str_decimal(share_terms, 4)},
                    {'id': 'deal_value', 'key': 'Current Deal Value', 'type_input': 'false', 'type': '',
                     'type_input2': 'false', 'value': deal_value,
                     'usd_val': convert_to_str_decimal(convert_to_float_else_zero(deal_value) * fx_local_to_base, 2)},
                    {'id': 'curr_price', 'key': target_ticker + ' Current Price', 'type_input': 'false', 'type': '',
                     'type_input2': 'false', 'value': convert_to_str_decimal(target_live_price, 2),
                     'usd_val': convert_to_str_decimal(target_live_price * fx_local_to_base, 2)},
                    {'id': 'gross_spread', 'key': 'Gross Spread', 'type_input': 'false', 'type': '',
                     'type_input2': 'false', 'value': convert_to_str_decimal(gross_spread, 2),
                     'usd_val': convert_to_str_decimal(gross_spread * fx_local_to_base, 2)},
                    {'id': 'target_dividend', 'key': 'Target Dividend', 'type_input': 'false', 'type': '',
                     'type_input2': 'true', 'value': convert_to_str_decimal(target_dividends, 2),
                     'usd_val': convert_to_str_decimal(target_dividends * fx_local_to_base, 2)},
                    {'id': 'acq_dividend', 'key': 'Acquirer Dividend', 'type_input': 'false', 'type': '',
                     'type_input2': 'true', 'value': convert_to_str_decimal(acq_dividends, 2),
                     'usd_val': convert_to_str_decimal(acq_dividends * fx_local_to_base, 2)},
                    {'id': 'dvd_adjusted_spread', 'key': 'DVD Adjusted Spread', 'type_input': 'false', 'type': '',
                     'type_input2': 'false', 'value': convert_to_str_decimal(dvd_adjusted_spread, 2),
                     'usd_val': convert_to_str_decimal(dvd_adjusted_spread * fx_local_to_base, 2)},
                    {'id': 'rebate_adjusted_spread', 'key': 'Rebate Adjusted Spread to', 'type_input': 'false',
                     'type': estimated_closing_date, 'type_input2': 'false',
                     'value': convert_to_str_decimal(rebate_adjusted_spread, 2),
                     'usd_val': convert_to_str_decimal(rebate_adjusted_spread * fx_local_to_base, 2)}
                ]

                rebate_data = [
                    {'id': 'funds_rate', 'type_input': 'false', 'key': 'Fed Funds Rate',
                     'acq_value': convert_to_str_decimal(fed_funds_last_price, 2),
                     'target_value': convert_to_str_decimal(fed_funds_last_price, 2)},
                    {'id': 'pb_rate', 'type_input': 'true', 'key': 'Less: PB Rate',
                     'acq_value': convert_to_str_decimal(acq_pb_rate, 2),
                     'target_value': convert_to_str_decimal(target_pb_rate, 2)},
                    {'id': 'rebate_pct', 'type_input': 'false', 'key': 'Rebate %',
                     'acq_value': convert_to_str_decimal(acq_rebate_pct),
                     'target_value': convert_to_str_decimal(target_rebate_pct)},
                ]

                five_percent_cap = convert_to_float_else_zero(float_so_value) * 0.05
                if convert_to_float_else_zero(fund_assets) != 0:
                    capacity = five_percent_cap * target_live_price * fx_local_to_base / fund_assets * 100
                else:
                    capacity = 0
                sizing_data = [
                    {'id': 'fund_assets', 'type_input': 'true', 'key': 'Fund Assets ($)', 'value': fund_assets},
                    {'id': 'float_so', 'type_input': 'true', 'key': 'Float S/O',
                     'value': convert_to_str_decimal(float_so_value, 2)},
                    {'id': 'five_cap', 'type_input': 'false', 'key': '5% cap',
                     'value': convert_to_str_decimal(five_percent_cap, 2)},
                    {'id': 'capacity', 'type_input': 'false', 'key': 'Capacity',
                     'value': convert_to_str_decimal(capacity, 2)}
                ]
                try:
                    size_in_shares = abs((nav_pct_impact * 0.01 * fund_assets) / (
                            topping_break_spread - gross_spread) / fx_local_to_base)
                except ZeroDivisionError:
                    size_in_shares = 0
                passive_spend = size_in_shares * target_live_price * fx_local_to_base
                try:
                    passive_pct_aum = passive_spend / fund_assets * 100
                except ZeroDivisionError:
                    passive_pct_aum = 0
                passive_data = [
                    {'id': 'nav_impact', 'type_input': 'true', 'key': 'NAV % impact', 'value': nav_pct_impact},
                    {'id': 'size_shares', 'type_input': 'false', 'key': 'Size in shares',
                     'value': convert_to_str_decimal(size_in_shares)},
                    {'id': 'spend', 'type_input': 'false', 'key': 'Spend (in USD)',
                     'value': convert_to_str_decimal(passive_spend)},
                    {'id': 'aum', 'type_input': 'false', 'key': '% AUM',
                     'value': convert_to_str_decimal(passive_pct_aum, 2)}
                ]

                scenario_data = []
                keys = ['id', 'credit_idea_id', 'scenario', 'last_price', 'dividends', 'rebate', 'hedge', 'deal_value',
                        'spread', 'gross_pct', 'annual_pct', 'days_to_close', 'dollars_to_make', 'dollars_to_lose',
                        'implied_prob', 'estimated_closing_date']
                scenario_count = 1
                for scenario in credit_idea_scenarios:
                    temp_dict = {key: getattr(scenario, key) for key in keys}
                    if arb_tradegroup.lower() != 'other':
                        if temp_dict['scenario'] and temp_dict['scenario'].lower() == 'earlier date':
                            temp_dict['estimated_closing_date'] = flat_file_closing_date - timedelta(days=31)
                        elif temp_dict['scenario'] and temp_dict['scenario'].lower() == 'base date':
                            temp_dict['estimated_closing_date'] = flat_file_closing_date
                        elif temp_dict['scenario'] and temp_dict['scenario'].lower() == 'worst date':
                            temp_dict['estimated_closing_date'] = flat_file_closing_date + timedelta(days=31)
                    if isinstance(temp_dict['estimated_closing_date'], (date, datetime)):
                        temp_dict['estimated_closing_date'] = temp_dict['estimated_closing_date'].strftime(
                            '%Y-%m-%d') if \
                            temp_dict['estimated_closing_date'] else temp_dict['estimated_closing_date']
                    temp_dict['days_to_close'] = calculate_number_of_days(temp_dict['estimated_closing_date'])
                    temp_dict['database_id'] = temp_dict.get('id')
                    temp_dict['last_price'] = round(target_live_price, 2)
                    temp_dict['dividends'] = convert_to_str_decimal(target_dividends - (acq_dividends * share_terms), 2)
                    try:
                        temp_implied_prob = round(
                            convert_to_float_else_zero(1 - (gross_spread / topping_break_spread)) * 100, 2)
                    except ZeroDivisionError:
                        temp_implied_prob = 0.00
                    temp_dict['implied_prob'] = temp_implied_prob
                    temp_dollars_to_lose = (gross_spread - topping_break_spread) * size_in_shares * fx_local_to_base
                    temp_dict['dollars_to_lose'] = round(convert_to_float_else_zero(temp_dollars_to_lose), 0)
                    temp_dict['DT_RowId'] = 'scenario_row_' + str(scenario_count)
                    scenario_data.append(temp_dict)
                    scenario_count += 1

                scenario_wo_hedge_data = []
                scenario_w_hedge_data = []
                keys = ['id', 'credit_idea_id', 'is_upside', 'is_downside', 'scenario', 'is_deal_closed',
                        'bond_last_price', 'bond_redemption', 'bond_redemption_type', 'bond_carry_earned',
                        'bond_rebate',
                        'bond_hedge', 'bond_deal_value', 'bond_spread', 'returns_gross_pct', 'returns_annual_pct',
                        'returns_estimated_closing_date', 'returns_days_to_close', 'profits_principal', 'profits_carry',
                        'profits_rebate', 'profits_hedge', 'profits_total', 'profits_day_of_break']
                scenario_count = 1
                for scenario in scenario_without_hedge:
                    temp_dict = {key: getattr(scenario, key) for key in keys}
                    if arb_tradegroup.lower() != 'other':
                        temp_dict['returns_estimated_closing_date'] = flat_file_closing_date
                    if isinstance(temp_dict['returns_estimated_closing_date'], (date, datetime)):
                        temp_dict['returns_estimated_closing_date'] = temp_dict[
                            'returns_estimated_closing_date'].strftime('%Y-%m-%d') if \
                            temp_dict['returns_estimated_closing_date'] else temp_dict['returns_estimated_closing_date']
                    temp_dict['returns_days_to_close'] = calculate_number_of_days(
                        temp_dict['returns_estimated_closing_date'])
                    temp_dict['database_id'] = temp_dict.get('id')
                    temp_dict['bond_redemption'] = convert_to_str_decimal(temp_dict['bond_redemption'], 3)
                    temp_dict['DT_RowId'] = 'scenario_wo_hedge_row_' + str(scenario_count)
                    scenario_wo_hedge_data.append(temp_dict)
                    scenario_count += 1

                scenario_count = 1
                for scenario in scenario_with_hedge:
                    temp_dict = {key: getattr(scenario, key) for key in keys}
                    if arb_tradegroup.lower() != 'other':
                        temp_dict['returns_estimated_closing_date'] = flat_file_closing_date
                    if isinstance(temp_dict['returns_estimated_closing_date'], (date, datetime)):
                        temp_dict['returns_estimated_closing_date'] = temp_dict[
                            'returns_estimated_closing_date'].strftime('%Y-%m-%d') if \
                            temp_dict['returns_estimated_closing_date'] else temp_dict['returns_estimated_closing_date']
                    temp_dict['returns_days_to_close'] = calculate_number_of_days(
                        temp_dict['returns_estimated_closing_date'])
                    temp_dict['database_id'] = temp_dict.get('id')
                    temp_dict['bond_redemption'] = convert_to_str_decimal(temp_dict['bond_redemption'], 3)
                    temp_dict['DT_RowId'] = 'scenario_w_hedge_row_' + str(scenario_count)
                    scenario_w_hedge_data.append(temp_dict)
                    scenario_count += 1

                scenario_comments_data = []
                scenario_count = 1
                for credit_idea_creditcomment in credit_idea_creditcomments:
                    temp_dict = {'database_id': credit_idea_creditcomment.id,
                                 'scenario': credit_idea_creditcomment.scenario,
                                 'comments': credit_idea_creditcomment.comments}
                    temp_dict['DT_RowId'] = 'scenario_comment_row_' + str(scenario_count)
                    scenario_comments_data.append(temp_dict)
                    scenario_count += 1

            except Exception:
                downsides_data, spread_data, rebate_data, sizing_data = [], [], [], []
                scenario_data, passive_data, bond_price_data, bond_information_data = [], [], [], []
                passive_phase_arb_data, estimated_liquidity_data, potential_outcomes_data, hedging_data = [], [], [], []
                scenario_wo_hedge_data, scenario_w_hedge_data, scenario_comments_data = [], [], []
            context.update({
                'credit_idea_id': credit_idea_id,
                'arb_tradegroup': arb_tradegroup.upper(),
                'other_tradegroup': other_tradegroup.upper() if other_tradegroup else '',
                'bbg_security_name': bbg_security_name.upper(),
                'downsides_data': json.dumps(downsides_data),
                'spread_data': json.dumps(spread_data),
                'fx_local_to_base': fx_local_to_base,
                'ccy': ccy.upper() if ccy else 'USD',
                'rebate_data': json.dumps(rebate_data),
                'sizing_data': json.dumps(sizing_data),
                'scenario_data': json.dumps(scenario_data),
                'passive_data': json.dumps(passive_data),
                'passive_phase_arb_data': json.dumps(passive_phase_arb_data),
                'bond_price_data': json.dumps(bond_price_data),
                'bond_information_data': json.dumps(bond_information_data),
                'estimated_liquidity_data': json.dumps(estimated_liquidity_data),
                'potential_outcomes_data': json.dumps(potential_outcomes_data),
                'hedging_data': json.dumps(hedging_data),
                'scenario_wo_hedge_data': json.dumps(scenario_wo_hedge_data),
                'scenario_w_hedge_data': json.dumps(scenario_w_hedge_data),
                'scenario_comments_data': json.dumps(scenario_comments_data),
            })
        return context


def save_credit_idea_data(context, master_data, credit_idea_id):
    response = 'failed'
    for key in master_data:
        if key.lower() == 'equity_scenario_data':
            context_scenario_data = json.loads(context['scenario_data'])
            scenario_data = master_data.get('equity_scenario_data')
            context_scenraio_data_ids = [i.get('id') for i in context_scenario_data]
            scenario_data_ids = [i.get('database_id') for i in scenario_data]
            credit_idea_scenarios = CreditIdeaScenario.objects.filter(credit_idea_id=credit_idea_id)
            for data in scenario_data:
                try:
                    credit_idea_scenario = credit_idea_scenarios.get(id=data.get('database_id'),
                                                                     credit_idea_id=credit_idea_id)
                except CreditIdeaScenario.DoesNotExist:
                    credit_idea_scenario = CreditIdeaScenario(credit_idea_id=credit_idea_id)
                credit_idea_scenario.scenario = data.get('scenario')
                credit_idea_scenario.last_price = data.get('last_price') or 0.00
                credit_idea_scenario.dividends = data.get('dividends') or 0.00
                credit_idea_scenario.rebate = data.get('rebate') or 0.00
                credit_idea_scenario.hedge = data.get('hedge') or 0.00
                credit_idea_scenario.deal_value = data.get('deal_value') or 0.00
                credit_idea_scenario.spread = data.get('spread') or 0.00
                credit_idea_scenario.gross_pct = data.get('gross_pct') or 0.00
                credit_idea_scenario.annual_pct = data.get('annual_pct') or 0.00
                credit_idea_scenario.dollars_to_make = data.get('dollars_to_make') or 0.00
                credit_idea_scenario.dollars_to_lose = data.get('dollars_to_lose') or 0.00
                credit_idea_scenario.implied_prob = data.get('implied_prob') or 0.00
                credit_idea_scenario.estimated_closing_date = data.get('exp_close')
                credit_idea_scenario.days_to_close = calculate_number_of_days(data.get('exp_close'))
                credit_idea_scenario.save()
            deleted_ids = set(context_scenraio_data_ids).difference(scenario_data_ids)
            if deleted_ids:
                for deleted_id in deleted_ids:
                    try:
                        CreditIdeaScenario.objects.get(id=deleted_id).delete()
                    except CreditIdeaScenario.DoesNotExist:
                        pass
        elif key.lower() == 'equity_details':
            try:
                credit_idea_details_object = CreditIdeaDetails.objects.get(credit_idea_id=credit_idea_id)
                credit_idea_details = master_data.get('equity_details')
                credit_idea_details_object.topping_big_upside = credit_idea_details.get('upside_value_upside')
                credit_idea_details_object.base_case_downside = credit_idea_details.get('upside_value_base_downside')
                credit_idea_details_object.outlier_downside = credit_idea_details.get('upside_value_outlier_downside')
                credit_idea_details_object.target_ticker = credit_idea_details.get('deal_terms_target_ticker')
                credit_idea_details_object.acq_ticker = credit_idea_details.get('deal_terms_acq_ticker')
                credit_idea_details_object.cash_consideration = credit_idea_details.get('deal_terms_value_cash')
                credit_idea_details_object.share_consideration = credit_idea_details.get('deal_terms_value_share')
                credit_idea_details_object.deal_value = credit_idea_details.get('deal_terms_value_deal_value')
                credit_idea_details_object.target_dividend = credit_idea_details.get('deal_terms_value_target_dividend')
                credit_idea_details_object.acq_dividend = credit_idea_details.get('deal_terms_value_acq_dividend')
                credit_idea_details_object.fund_assets = credit_idea_details.get('sizing_val_fund_assets')
                credit_idea_details_object.float_so = credit_idea_details.get('sizing_val_float_so')
                credit_idea_details_object.acq_pb_rate = credit_idea_details.get('rebate_acq_val_pb_rate')
                credit_idea_details_object.target_pb_rate = credit_idea_details.get('rebate_target_val_pb_rate')
                credit_idea_details_object.nav_pct_impact = credit_idea_details.get('passive_value_nav_impact')
                credit_idea_details_object.save()
            except CreditIdeaDetails.DoesNotExist:
                return 'failed'

        elif key.lower() == 'credit_details':
            try:
                credit_details_obj = CreditIdeaCreditDetails.objects.get(credit_idea_id=credit_idea_id)
                credit_details = master_data.get('credit_details')
                credit_details_obj.bond_ticker = credit_details.get('bond_information_bond_ticker')
                credit_details_obj.face_value_of_bonds = credit_details.get('passive_phase_arb_face_value_of_bonds')
                credit_details_obj.bbg_security_name = credit_details.get('bond_information_bbg_security_name')
                credit_details_obj.bbg_interest_rate = credit_details.get('bond_information_bbg_interest_rate')
                credit_details_obj.bbg_issue_size = credit_details.get('bond_information_bbg_issue_size')
                credit_details_obj.bond_est_purchase_price = credit_details.get('bond_price_est_purchase_price')
                credit_details_obj.bbg_bid_price = credit_details.get('bond_price_bbg_bid_price')
                credit_details_obj.bbg_ask_price = credit_details.get('bond_price_bbg_ask_price')
                credit_details_obj.bbg_last_price = credit_details.get('bond_price_bbg_last_price')
                credit_details_obj.base_break_price = credit_details.get('potential_outcomes_value_base_break_price')
                credit_details_obj.conservative_break_price = credit_details.get(
                    'potential_outcomes_value_conservative_break_price')
                credit_details_obj.call_price = credit_details.get('potential_outcomes_value_call_price')
                credit_details_obj.make_whole_price = credit_details.get('potential_outcomes_value_make_whole_price')
                credit_details_obj.equity_claw_percent = credit_details.get('potential_outcomes_equity_claw_value')
                credit_details_obj.equity_claw_value = credit_details.get('potential_outcomes_value_equity_claw_value')
                credit_details_obj.blend = credit_details.get('potential_outcomes_value_blend')
                credit_details_obj.change_of_control = credit_details.get('potential_outcomes_value_change_of_control')
                credit_details_obj.acq_credit = credit_details.get('potential_outcomes_value_acq_credit')
                credit_details_obj.other_acq_credit = credit_details.get('potential_outcomes_value_other_acq_credit')
                credit_details_obj.proposed_ratio = credit_details.get('hedging_proposed_ratio')
                credit_details_obj.break_spread = credit_details.get('hedging_break_spread')
                credit_details_obj.bbg_est_daily_vol = credit_details.get('estimated_liquidity_bbg_est_daily_vol')
                credit_details_obj.bbg_actual_thirty_day = credit_details.get(
                    'estimated_liquidity_bbg_actual_thirty_day')
                credit_team_view = credit_details.get('estimated_liquidity_credit_team_view') or 1
                credit_team_view = int(credit_team_view) if credit_team_view else 1
                credit_details_obj.credit_team_view = credit_team_view
                credit_details_obj.save()
            except CreditIdeaCreditDetails.DoesNotExist:
                return 'failed'
        elif key.lower() == 'credit_scenario_data':
            context_scenario_data = json.loads(context['scenario_w_hedge_data']) + json.loads(
                context['scenario_wo_hedge_data'])
            scenario_data = master_data.get('credit_scenario_data')
            context_scenraio_data_ids = [i.get('id') for i in context_scenario_data]
            scenario_data_ids = [i.get('database_id') for i in scenario_data]
            scenario_obj = CreditIdeaCreditScenario.objects.filter(credit_idea_id=credit_idea_id)
            for data in scenario_data:
                try:
                    credit_idea_scenario = scenario_obj.get(id=data.get('database_id'), credit_idea_id=credit_idea_id)
                except CreditIdeaCreditScenario.DoesNotExist:
                    credit_idea_scenario = CreditIdeaCreditScenario(credit_idea_id=credit_idea_id)
                credit_idea_scenario.is_upside = data.get('is_upside')
                credit_idea_scenario.is_downside = data.get('is_downside')
                credit_idea_scenario.scenario = data.get('scenario')
                credit_idea_scenario.is_deal_closed = data.get('is_deal_closed')
                credit_idea_scenario.is_hedge = data.get('is_hedge')
                credit_idea_scenario.bond_last_price = data.get('bond_last_price') or 0.00
                credit_idea_scenario.bond_redemption_type = data.get('bond_redemption_type')
                credit_idea_scenario.bond_redemption = data.get('bond_redemption') or 0.00
                credit_idea_scenario.bond_carry_earned = data.get('bond_carry_earned') or 0.00
                credit_idea_scenario.bond_rebate = data.get('bond_rebate') or 0.00
                credit_idea_scenario.bond_hedge = data.get('bond_hedge') or 0.00
                credit_idea_scenario.bond_deal_value = data.get('bond_deal_value') or 0.00
                credit_idea_scenario.bond_spread = data.get('bond_spread') or 0.00
                credit_idea_scenario.returns_gross_pct = data.get('returns_gross_pct') or 0.00
                credit_idea_scenario.returns_annual_pct = data.get('returns_annual_pct') or 0.00
                credit_idea_scenario.returns_estimated_closing_date = data.get('returns_estimated_closing_date')
                credit_idea_scenario.returns_days_to_close = calculate_number_of_days(
                    data.get('returns_estimated_closing_date'))
                credit_idea_scenario.profits_principal = data.get('profits_principal') or 0.00
                credit_idea_scenario.profits_carry = data.get('profits_carry') or 0.00
                credit_idea_scenario.profits_rebate = data.get('profits_rebate') or 0.00
                credit_idea_scenario.profits_hedge = data.get('profits_hedge') or 0.00
                credit_idea_scenario.profits_total = data.get('profits_total') or 0.00
                credit_idea_scenario.profits_day_of_break = data.get('profits_day_of_break') or 0.00
                credit_idea_scenario.save()
            deleted_ids = set(context_scenraio_data_ids).difference(scenario_data_ids)
            if deleted_ids:
                for deleted_id in deleted_ids:
                    try:
                        CreditIdeaCreditScenario.objects.get(id=deleted_id).delete()
                    except CreditIdeaCreditScenario.DoesNotExist:
                        pass
        elif key.lower() == 'credit_comments_data':
            context_comments_data = json.loads(context['scenario_comments_data'])
            credit_comments_data = master_data.get('credit_comments_data')
            context_comments_data_ids = [i.get('database_id') for i in context_comments_data]
            comments_data_ids = [i.get('database_id') for i in credit_comments_data]
            comments_obj = CreditIdeaCreditScenarioComments.objects.filter(credit_idea_id=credit_idea_id)
            for data in credit_comments_data:
                try:
                    credit_idea_comment = comments_obj.get(id=data.get('database_id'), credit_idea_id=credit_idea_id)
                except CreditIdeaCreditScenarioComments.DoesNotExist:
                    credit_idea_comment = CreditIdeaCreditScenarioComments(credit_idea_id=credit_idea_id)
                credit_idea_comment.scenario = data.get('scenario')
                credit_idea_comment.comments = data.get('comment')
                credit_idea_comment.save()
            deleted_ids = set(context_comments_data_ids).difference(comments_data_ids)
            if deleted_ids:
                for deleted_id in deleted_ids:
                    try:
                        CreditIdeaCreditScenarioComments.objects.get(id=deleted_id).delete()
                    except CreditIdeaCreditScenarioComments.DoesNotExist:
                        pass
    response = 'success'
    return response


def show_all_credit_deals(request):
    """
    View to retrieve all credit deals
    :param request: Request object (GET)
    :return: render with context dictionary
    """
    try:
        deals_dict = {}

        # get max date in CreditStaticScreen
        max_date = CreditStaticScreen.objects.latest('date_updated').date_updated

        # get all credit deals as of max date into dataframe
        deals_df = pd.DataFrame.from_records(
            list(CreditStaticScreen.objects.filter(date_updated=max_date).values()))

        # clean up dataframe
        deals_df = replace_boolean_fields(deals_df)
        deals_df = round_decimal_fields(deals_df, 2)
        deals_df = deals_df.round(2)
        deals_df = deals_df.dropna(how='all')
        deals_df = deals_df.fillna('')

        # get available funds
        funds_list = [x for x in sorted(deals_df['fund'].unique().tolist()) if x]
        # move TACO to the front of the list
        funds_list.insert(0, funds_list.pop(funds_list.index('TACO')))

        index_list = ['ACFIX US EQUITY', 'G0O1 Index']

        fund_level_tag = 'AGGREGATE_STATS'
        funds_df = deals_df[deals_df['tradegroup'] == fund_level_tag]
        index_df = deals_df[deals_df['tradegroup'].isin(index_list)]
        deals_df = deals_df[~deals_df['tradegroup'].isin(funds_list + index_list + [fund_level_tag])]
        deals_df = deals_df[deals_df['tradegroup'] != '']

        for fund in funds_list:
            deals_dict[fund] = deals_df[deals_df['fund'] == fund].to_dict('records')

    except Exception:
        funds_df = pd.DataFrame()
        index_df = pd.DataFrame()
        import traceback
        logging.error(traceback.format_exc())
    def prepare_dataframe(temp_df, rename_columns):
        # dynamically select fields from the table and transpose to display
        if temp_df.empty:
            return temp_df
        temp_df = temp_df[list(rename_columns.keys())]
        temp_df.rename(columns=rename_columns, inplace=True)
        temp_df = temp_df.T
        temp_df.reset_index(inplace=True)
        return temp_df


    rename_columns = {'fund': '', 'ytm': 'YTM', 'ytc': 'YTC', 'ytw': 'YTW', 'yte': 'YTE', 'dtm': 'DTM',
                      'eff_duration': 'Eff. Duration', 'dte': 'DTE', 'dv01': 'dv01', 'cr01': 'cr01'}
    # move funds_df row with 'TACO' to the front of the list
    funds_df = funds_df[funds_df['fund'] == 'TACO'].append(funds_df[funds_df['fund'] != 'TACO'])

    funds_df = prepare_dataframe(funds_df, rename_columns)

    rename_columns = {'tradegroup': '', 'last_close_trr_5d': '5D', 'last_close_trr_mtd': 'MTD',
                      'last_close_trr_ytd': 'YTD', 'last_close_trr_1yr': '1YR', 'last_close_trr_3yr': '3YR',
                      'last_close_trr_5yr': '5YR'}
    #call_schedule_yeild1

    index_df = prepare_dataframe(index_df, rename_columns)

    return render(request, 'credit_database.html', {'credit_deals_dict': deals_dict,
                                                    'funds_df': funds_df.to_dict(orient='records'),
                                                    'index_df': index_df.to_dict(orient='records'),
                                                    'fund_list': funds_list})


def show_credit_deal(request):
    isin = request.GET['ISIN']
    deal_details = CreditStaticScreen.objects.filter(isin=isin).order_by('-date_updated').values()
    if not deal_details.exists():
        return render(request, 'show_credit_deal.html', {'tradegroup_name': '', 'deal': None})
    deals_df = pd.DataFrame.from_records(deal_details)

    deals_df['present_value_add1'] = deals_df['present_value'] * 1.01
    deals_df['present_value_sub1'] = deals_df['present_value'] * 0.99

    # clean up dataframe
    deals_df = replace_boolean_fields(deals_df)
    deals_df = round_decimal_fields(deals_df, 2)
    deals_df = deals_df.fillna('N/A')

    deals_df = deals_df.iloc[0]

    return render(request, 'show_credit_deal.html', {'tradegroup_name': deals_df.tradegroup, 'deal': deals_df})
