import datetime
import logging
import time
import pandas as pd
import re

from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.urls import reverse
from django.views.generic.edit import FormView
from django_slack import slack_message

from bbgclient import bbgclient
from mna_deal.forms import CreateMaDealsForm, EditMaDealsForm
from risk.ma_regression_utils import mna_bulk_downside_forecast
from risk.mna_deal_bloomberg_utils import get_data_from_bloombery_using_action_id
from risk.models import MA_Deals, MaDealsActionIdDetails, MaDownsidePeerSource
from risk.tasks import calc_additional_fields_and_save, calculate_spread_targetp, calculate_net_div_impact, \
    calculate_deal_value
from risk.views import save_spread_index_edit, recalc_unaffected_price
from risk_reporting.deal_downside.downside_calculations import create_or_update_equal_weighted_model, \
    create_regression_downside_from_peer_source, generate_bloomberg_peer_source, regenerate_regression_downside
from risk_reporting.models import FormulaeBasedDownsides, DealDownside, LinearRegressionDownside
from slack_utils import get_channel_name
import math

logger = logging.getLogger(__name__)

class CreateMaDealsView(FormView):
    """Views for Creating a new M & A Deal"""
    template_name = 'create_mna_deal.html'
    form_class = CreateMaDealsForm
    fields = '__all__'

    def get_success_url(self):
        """Redirect the User back to the referal page"""
        http_referer = self.request.GET.get('referer')
        if http_referer == 'mna_idea_database':
            ## timer to get frontend to show latest deal - removed bc the views function waits
            return reverse('risk:mna_idea_database')
        if http_referer == 'formula_based_downsides':
            return reverse('risk:mna_idea_database')
        return '#'


    def form_valid(self, form):
        """Create the objects in respective models if the form is valid"""
        data = form.cleaned_data
        action_id = self.request.POST.get('action_id')
        deal_name = data.get('deal_name')
        deal_name = deal_name.upper() if deal_name else deal_name
        analyst = data.get('analyst')
        # announced_date = ''
        target_ticker = data.get('target_ticker')
        target_ticker = target_ticker.upper() if target_ticker else target_ticker
        acquirer_ticker = data.get('acquirer_ticker')
        acquirer_ticker = acquirer_ticker.upper() if acquirer_ticker else acquirer_ticker
        deal_cash_terms = data.get('deal_cash_terms')
        deal_share_terms = data.get('deal_share_terms')
        deal_value = data.get('deal_value')
        expected_close_date = data.get('expected_close_date')
        target_dividends = data.get('target_dividends')
        acquirer_dividends = data.get('acquirer_dividends')
        short_rebate = data.get('short_rebate')
        fx_carry_percent = data.get('fx_carry_percent')
        stub_cvr_value = data.get('stub_cvr_value')
        acquirer_upside = data.get('acquirer_upside')
        loss_tolerance_percentage_of_limit = data.get('loss_tolerance_percentage_of_limit')
        status = 'ACTIVE'
        created = datetime.datetime.now().date()
        risk_limit = -1 #data.get('risk_limit')
        origination_date = data.get('origination_date')
        position_in_acquirer = data.get('position_in_acquirer')
        is_cross_border_deal = data.get('is_cross_border_deal')
        deal_currency = data.get('deal_currency')
        unaffected_price = data.get('unaffected_price')
        unaffected_price_30 = data.get('unaffected_price_30')
        unaffected_price_90 = data.get('unaffected_price_90')
        unaffected_date = data.get('unaffected_date')
        outlier = data.get('outlier')
        base_case = data.get('base_case')
        unaffected_downsides = data.get('unaffected_downsides')
        origination_date = datetime.datetime.strptime(origination_date, '%m/%d/%Y')
        latest_object = FormulaeBasedDownsides.objects.filter(id__isnull=False).latest('id')
        max_id = latest_object.id if latest_object and latest_object.id else -1
        insert_id = max_id + 1

        # calcs for new mna deals for AID:
        data['action_id'] = action_id
        new_deal_response = calc_additional_fields_and_save(data)
        if not MaDealsActionIdDetails.objects.filter(action_id=action_id).exists():
            # DO NOT USE DF to save, its outside of django mvc structure and will fault
            # df.to_sql(name='risk_madealsactioniddetails', con=settings.SQLALCHEMY_CONNECTION, if_exists='append',index=False, schema=settings.CURRENT_DATABASE)
            MaDealsActionIdDetails.objects.create(**new_deal_response)

        # Save to MA Deal Model
        ma_deals_df = pd.DataFrame.from_records(MA_Deals.objects.all().values('deal_name'))
        if ma_deals_df[ma_deals_df['deal_name'].str.contains(deal_name, flags=re.IGNORECASE)].empty:
            deal_object = MA_Deals(action_id=action_id, deal_name=deal_name,
                                   target_ticker=target_ticker, acquirer_ticker=acquirer_ticker,
                                   deal_cash_terms=deal_cash_terms, deal_share_terms=deal_share_terms,
                                   deal_value=deal_value, expected_closing_date=expected_close_date,
                                   target_dividends=target_dividends, acquirer_dividends=acquirer_dividends,
                                   short_rebate=short_rebate, fx_carry_percent=fx_carry_percent,
                                   unaffected_date=unaffected_date,
                                   stub_cvr_value=stub_cvr_value, acquirer_upside=acquirer_upside, status=status,
                                   created=created, last_modified=datetime.datetime.now().date(), is_complete='No',
                                   loss_tolerance_percentage_of_limit=loss_tolerance_percentage_of_limit,
                                   is_cross_border_deal=is_cross_border_deal, deal_currency=deal_currency,
                                   unaffected_price=unaffected_price, unaffected_price_30=unaffected_price_30,
                                   unaffected_price_90=unaffected_price_90)
            deal_object.save()
            # try:
            #     result = get_data_from_bloombery_using_action_id([action_id])
            #     save_bloomberg_data_to_table(result, [deal_object])
            # except IntegrityError as e:
            #     slack_message('generic.slack', {'message': 'Duplicate Action ID While creating M&A Deal. Action ID: ' + str(action_id)},
            #                   channel=get_channel_name('new-mna-deals'), token=settings.SLACK_TOKEN,
            #                   name='ESS_IDEA_DB_ERROR_INSPECTOR')
            # except Exception as e:
            #     slack_message('generic.slack', {'message': 'Error: Action ID while creating M&A Deal. Action ID: ' + str(action_id)},
            #                   channel=get_channel_name('new-mna-deals'), token=settings.SLACK_TOKEN,
            #                   name='ESS_IDEA_DB_ERROR_INSPECTOR')
            try:
                #geerate default bloomberg regression model, and select the first one as a
                created_downsides = create_regression_downside_from_peer_source(
                    generate_bloomberg_peer_source(deal_object))
                if created_downsides:
                    bloomberg_regression_downside = created_downsides[0]
                    bloomberg_regression_downside.content_object.is_selected = True
                    bloomberg_regression_downside.content_object.save()

            except Exception as e:
                slack_message('generic.slack',
                              {'message': f'Error: Failed to create regression model for deal id {action_id}'},
                              channel=get_channel_name('new-mna-deals'), token=settings.SLACK_TOKEN,
                              name='ESS_IDEA_DB_ERROR_INSPECTOR')

            slack_message('new_mna_deal_notify.slack',
                          {'message': 'New M & A Deal Added', 'deal_name': deal_name, 'action_id': action_id,
                           'analyst': analyst, 'target_ticker': target_ticker, 'acquirer_ticker': acquirer_ticker,
                           'deal_cash_terms': deal_cash_terms, 'deal_share_terms': deal_share_terms,
                           'deal_value': deal_value, 'target_dividends': target_dividends,
                           'acquirer_dividends': acquirer_dividends, 'short_rebate': short_rebate,
                           'fx_carry_percent': fx_carry_percent, 'stub_cvr_value': stub_cvr_value,
                           'acquirer_upside': acquirer_upside, 'position_in_acquirer': position_in_acquirer,
                           'loss_tolerance_percentage_of_limit': loss_tolerance_percentage_of_limit,
                           'risk_limit': risk_limit, 'is_cross_border_deal': is_cross_border_deal,
                           'deal_currency': deal_currency, 'expected_closing_date': expected_close_date,
                           'origination_date': origination_date},
                          channel=get_channel_name('new-mna-deals'), token=settings.SLACK_TOKEN,
                          name='ESS_IDEA_DB_ERROR_INSPECTOR')

        formulae_df = pd.DataFrame.from_records(FormulaeBasedDownsides.objects.all().values('TradeGroup'))
        if formulae_df[formulae_df['TradeGroup'].str.contains(deal_name, flags=re.IGNORECASE)].empty:
            # Calculate Last Price and Save to FormulaeBasedDownsides Model for Target
            try:
                api_host = bbgclient.get_next_available_host()
                target_last_price = float(bbgclient.get_secid2field([target_ticker], 'tickers', ['CRNCY_ADJ_PX_LAST'],
                                                                    req_type='refdata', api_host=api_host)[
                                              target_ticker]
                                          ['CRNCY_ADJ_PX_LAST'][0]) if deal_share_terms > 0 else 0
            except Exception as error:
                target_last_price = None
            target_object = FormulaeBasedDownsides(id=insert_id, TradeGroup=deal_name, Underlying=target_ticker,
                                                   TargetAcquirer='Target', Analyst=analyst, RiskLimit=risk_limit,
                                                   OriginationDate=origination_date, DealValue=deal_value,
                                                   LastPrice=target_last_price, deal_currency=deal_currency,
                                                   unaffected_downsides=unaffected_downsides,
                                                   is_cross_border_deal=is_cross_border_deal,
                                                   outlier=unaffected_downsides,
                                                   base_case=unaffected_downsides,
                                                   BaseCaseDownsideType='Unaffected Downside',
                                                   OutlierDownsideType='Unaffected Downside')
            target_object.save()
            create_or_update_equal_weighted_model(deal_name, target_ticker)

            # If Position in Acquirer is Yes, then calculate last price and save to FormulaeBasedDownsides Model
            # for Acquirer
            if position_in_acquirer.lower() == 'yes':
                try:
                    api_host = bbgclient.get_next_available_host()
                    acquirer_last_price = float(
                        bbgclient.get_secid2field([target_ticker], 'tickers', ['CRNCY_ADJ_PX_LAST'],
                                                  req_type='refdata', api_host=api_host)[target_ticker]
                        ['CRNCY_ADJ_PX_LAST'][0]) if deal_share_terms > 0 else 0
                except Exception as error:
                    acquirer_last_price = None
                acquirer_object = FormulaeBasedDownsides(id=insert_id + 1, TradeGroup=deal_name,
                                                         Underlying=acquirer_ticker, TargetAcquirer='Acquirer',
                                                         Analyst=analyst, RiskLimit=risk_limit,
                                                         OriginationDate=origination_date, DealValue=deal_value,
                                                         LastPrice=acquirer_last_price, deal_currency=deal_currency,
                                                         outlier=unaffected_downsides,
                                                         base_case=unaffected_downsides,
                                                         is_cross_border_deal=is_cross_border_deal,
                                                         BaseCaseDownsideType='Unaffected Downside',
                                                         OutlierDownsideType='Unaffected Downside')
                acquirer_object.save()
            slack_message('new_mna_deal_notify.slack',
                          {'message': 'New Deal Added in Formulae Downside', 'deal_name': deal_name,
                           'action_id': action_id, 'analyst': analyst, 'target_ticker': target_ticker,
                           'acquirer_ticker': acquirer_ticker, 'deal_cash_terms': deal_cash_terms,
                           'deal_share_terms': deal_share_terms, 'deal_value': deal_value,
                           'target_dividends': target_dividends, 'acquirer_dividends': acquirer_dividends,
                           'short_rebate': short_rebate, 'fx_carry_percent': fx_carry_percent,
                           'stub_cvr_value': stub_cvr_value, 'acquirer_upside': acquirer_upside,
                           'loss_tolerance_percentage_of_limit': loss_tolerance_percentage_of_limit,
                           'risk_limit': risk_limit,
                           'is_cross_border_deal': is_cross_border_deal, 'deal_currency': deal_currency,
                           'expected_closing_date': expected_close_date, 'origination_date': origination_date},
                          channel=get_channel_name('new-mna-deals'), token=settings.SLACK_TOKEN,
                          name='ESS_IDEA_DB_ERROR_INSPECTOR')
        return super(CreateMaDealsView, self).form_valid(form)


class EditMaDealsView(FormView):
    """ Method to save the approvals by Deal ID as a JSON string. Use create or update """
    template_name = 'edit_mna_deal.html'
    form_class = EditMaDealsForm
    fields = '__all__'

    # to grab data from the url or html elements
    def get_context_data(self, **kwargs):
        # call parent class
        context = super().get_context_data(**kwargs)
        deal_id = self.request.GET['deal_id']
        deal_id = int(deal_id)
        context["deal_id"] = deal_id
        return context

    # this will run during setup when the form fields initiate
    def get_initial(self):
        # call parent class. init is a dict
        initial = super(EditMaDealsView, self).get_initial()
        deal_id = self.request.GET['deal_id']
        if deal_id:
            deal_object = MA_Deals.objects.get(id=int(deal_id))
            raw_action_id = deal_object.action_id
            aid_obj = MaDealsActionIdDetails.objects.filter(action_id=raw_action_id).values()
            deal_obj = MA_Deals.objects.filter(action_id=raw_action_id).values()
            list_aid_obj = list(aid_obj)
            list_aid_obj[0]['deal_name'] = deal_obj[0]['deal_name']
            # initial.update(vars(obj))
            list_keys = list_aid_obj[0].keys()
            # print(list_keys)
            for k in list_keys:
                initial[str(k)] = list_aid_obj[0][str(k)]
        return initial

    def rerun_ticker_edit(self, raw_action_id, diff_fields, slack_string, form, MaDealsActionIdDetails):
        ''' if only ticker changes, run new_deal, checks, slack and save to AID'''
        target_ticker, acquirer_ticker = diff_fields.get('target_ticker', None), diff_fields.get('acquirer_ticker',
                                                                                                 None)
        action_id_dict = {}
        action_id_dict['action_id'] = raw_action_id
        result = get_data_from_bloombery_using_action_id([raw_action_id + ' Action'])
        data = result.get(raw_action_id + ' Action')
        response_new_deal = calculate_spread_targetp(action_id_dict, data, target_ticker, acquirer_ticker)
        # error check
        if response_new_deal.get('error', False) == True:
            slack_string += 'ticker: ' + 'Error on the ticker you submitted '
            slack_message('generic.slack',
                          {'message': "Edit action id -> " + str(raw_action_id) + ' DATA: ' + slack_string},
                          channel=get_channel_name('new-mna-deals'), token=settings.SLACK_TOKEN, name='TEST AGENT')
            return
        new_deal_dict = calc_additional_fields_and_save(response_new_deal)
        new_deal_dict['manual_frontend_edit'] = 1
        df = pd.DataFrame([new_deal_dict])
        MaDealsActionIdDetails.objects.get(action_id=raw_action_id).delete()
        df.to_sql(name='risk_madealsactioniddetails', con=settings.SQLALCHEMY_CONNECTION,
                  if_exists='append', index=False, schema=settings.CURRENT_DATABASE)
        if 'acquirer_ticker' in diff_fields.keys():
            # deal_object.acquirer_ticker = diff_fields['acquirer_ticker']
            slack_string += 'acquirer_ticker ' + diff_fields['acquirer_ticker']
        if 'target_ticker' in diff_fields.keys():
            # deal_object.target_ticker = diff_fields['target_ticker']
            slack_string += 'target_ticker ' + diff_fields['target_ticker']
        slack_message('generic.slack',
                      {'message': "Edit action id -> " + str(raw_action_id) + ' DATA: ' + slack_string},
                      channel=get_channel_name('new-mna-deals'), token=settings.SLACK_TOKEN, name='TEST AGENT')
        return

    def form_valid(self, form):
        form_dict = self.request.POST.copy()
        diff_fields = {}
        calc_fields = ['special_dividend', 'cash_value', 'net_div_impact', 'tgt_div', 'acq_div', 'acq_div_amt', 'tgt_div_amt',
                       'acq_total_divs', 'tgt_total_divs', 'stock_terms', 'cash_terms', 'acq_div_days', 'deal_value']
        noncalc_fields = ['net_div_impacts','action_id','termination_date','drop_dead_date','acquirer_name', 'acquirer_ticker','target_name']
        skip_entry_list = ['csrfmiddlewaretoken', 'analyst', 'peer_index', 'spread_index', 'list']
        skip_format_list = ['deal_name']
        slack_string = ''
        if form_dict:
            deal_id = self.request.GET['deal_id']
            if deal_id:
                deal_object = MA_Deals.objects.get(id=int(deal_id))
                raw_action_id = deal_object.action_id
                # get obj allows for saving, filter/values allows for retrieving data.
                original_aid_obj = MaDealsActionIdDetails.objects.filter(action_id=raw_action_id).values()
                original_aid_obj_get = MaDealsActionIdDetails.objects.get(action_id=raw_action_id)
                original_aid_list = list(original_aid_obj)[0]
                original_aid_list_copy = original_aid_list.copy()
                # set to match frontend none
                for k in original_aid_list_copy.keys():
                    if original_aid_list_copy[k] == None:
                        original_aid_list_copy[k] = ''
                if 'csrfmiddlewaretoken' in form_dict.keys():
                    form_dict['csrfmiddlewaretoken'] = ''
                # list of changed fields. compare fields.
                for key in form_dict:
                    if key not in skip_entry_list:
                        if key not in skip_format_list:
                            if key not in ['unaffected_date', 'announced_date']:
                                original_aid_list_copy[key] = str(original_aid_list_copy[key])
                            else:
                                original_aid_list_copy[key] = original_aid_list_copy[key].strftime('%Y-%m-%d')
                        if form_dict.get(key, '') != original_aid_list_copy.get(key, ''):
                            diff_fields[key] = form_dict.get(key, None)
                    else:
                        if key in ['peer_index', 'spread_index']:
                            if form_dict.get(key, '') != original_aid_list_copy.get(key, ''):
                                diff_fields[key] = form_dict.get(key, None)
                # exit if no changes
                if len(diff_fields) < 1:
                    return super(EditMaDealsView, self).form_valid(form)
                # rerun the entire deal and return
                if 'acquirer_ticker' in diff_fields.keys() or 'target_ticker' in diff_fields.keys():
                    self.rerun_ticker_edit(raw_action_id, diff_fields, slack_string, form, MaDealsActionIdDetails)
                    return super(EditMaDealsView, self).form_valid(form)
                # if spread,cix index changes
                spread_index = diff_fields.get('spread_index',None)
                peer_index = diff_fields.get('peer_index',None)
                if spread_index:
                    spread_index_response = save_spread_index_edit(spread_index, deal_object)
                    diff_fields['spread_index'] = spread_index_response
                if peer_index:
                    peer_index_response = save_spread_index_edit(peer_index, deal_object)
                    diff_fields['peer_index'] = peer_index_response
                # check if in AID, then check if recalc fields
                if any(x in diff_fields.keys() for x in calc_fields):
                    slack_string = recalculate_mna_deal(diff_fields, original_aid_list, original_aid_obj_get, slack_string)
                # static values
                else:
                    for k in diff_fields.keys():
                        if k == 'net_div_impacts':
                            original_aid_obj_get.net_div_impacts = diff_fields[k]
                        if k == 'action_id':
                            string_action_id = diff_fields[k]
                            original_aid_obj_get.action_id = string_action_id.strip()
                            deal_object.action_id = string_action_id.strip()
                        if k == 'termination_date':
                            original_aid_obj_get.termination_date = diff_fields[k]
                        if k == 'drop_dead_date':
                            original_aid_obj_get.drop_dead_date = diff_fields[k]
                        if k == 'acquirer_name':
                            original_aid_obj_get.acquirer_name = diff_fields[k].strip()
                        if k == 'target_name':
                            original_aid_obj_get.target_name = diff_fields[k].strip()
                # independent re-calculations
                if 'deal_name' in diff_fields.keys():
                    deal_object.deal_name = diff_fields['deal_name']
                    deal_object.save()
                if 'action_id' in diff_fields.keys():
                    deal_object.action_id = diff_fields['action_id']
                    original_aid_obj_get.action_id = diff_fields['action_id']
                    deal_object.save()
                if 'unaffected_date' in diff_fields.keys():
                    in_request = diff_fields.copy()  # copy and input target tick + unaff date
                    in_request["target_ticker"] = original_aid_obj_get.target_ticker + ' EQUITY'
                    response = recalc_unaffected_price(in_request,1) # returns a float w/ 1
                    diff_fields['unaffected_prices_response:30,90,nominal'] = str(response['unaffected_price_30']) +' '+str(response['unaffected_price_90']) +' '+str(response['unaffected_price'])
                    diff_fields['unaffected_downsides'] = response['unaffected_downsides']
                    # original_aid_obj_get.unaffected_price_30 = response['unaffected_price_30']
                    original_aid_obj_get.unaffected_90d_vwap = response['unaffected_price_90']
                    original_aid_obj_get.unaffected_price = response['unaffected_price']
                    original_aid_obj_get.unaffected_downside = response['unaffected_downsides']
                    original_aid_obj_get.unaffected_date = datetime.datetime.strptime(diff_fields['unaffected_date'],'%Y-%m-%d')
                    deal_object.unaffected_date = datetime.datetime.strptime(diff_fields['unaffected_date'],'%Y-%m-%d').date()
                    deal_object.save()
                    # calculate regression model on unaffected_date change
                    try:
                        regenerate_regression_downside(deal_object)
                    except Exception as e:
                        import traceback
                        logger.error(traceback.format_exc())
                # save
                original_aid_obj_get.manual_frontend_edit = 1
                original_aid_obj_get.save()
                # unpack to avoid carriage debree strings "&#"
                for k,v in diff_fields.items():
                    slack_string += str(k)+':' +(str(v) if len(str(v))<100 else str(v)[0:100]) + ' , '

        slack_message('generic.slack',
                      {'message': "Edit action id -> "+ str(raw_action_id) +' DATA: '+ slack_string},
                      channel=get_channel_name('new-mna-deals'),
                      token=settings.SLACK_TOKEN,
                      name='TEST AGENT')

        return super(EditMaDealsView, self).form_valid(form)

    def get_success_url(self):
        """Redirect the User back to the deal page"""
        deal_id = self.request.GET['deal_id']
        time.sleep(0.5)
        return reverse('risk:show_mna_idea') + "?mna_idea_id=" + str(deal_id)

    def form_invalid(self, form):
        return super(EditMaDealsView, self).form_invalid(form)


def recalculate_mna_deal(frontend_changes, original_aid_list, aid_obj, slack_string):
    ''' shortcut calcs for mna deals. Omits  calc_additional_fields_and_save() and saves entries. '''
    # combine changes and format
    frontend_change_keys = frontend_changes.keys()
    for k in original_aid_list.keys():
        if k in frontend_change_keys:
            if k in ['cash_value','net_div_impact','special_dividend','stock_ratio','acq_div_days',
                     'acq_div','acq_div_freq', 'acq_div_amt', 'tgt_total_divs','acquirer_live_price','exchange_rate']:
                try:
                    original_aid_list[k] = float(frontend_changes[k])
                except:
                    original_aid_list[k] = frontend_changes[k]
            else:
                original_aid_list[k] = frontend_changes[k]
    # vars
    payment_type, cash_value, Net_Div_Impact, Special_Dividend = [original_aid_list['payment_type'],
                                                                  float(original_aid_list['cash_value']) if original_aid_list['cash_value'] else 0,
                                                                  float(original_aid_list['net_div_impact']),
                                                                  float(original_aid_list['special_dividend'])]
    Stock_Terms, Acq_Div_Days, Acq_Div, Acq_Div_Freq, Acq_Div_Amt = [original_aid_list['stock_terms'],
                                                                     float(original_aid_list['acq_div_days']),
                                                                     float(original_aid_list['acq_div']),
                                                                     original_aid_list['acq_div_freq'],
                                                                     float(original_aid_list['acq_div_amt'])]
    wic_close_est, Tgt_Total_Divs, Acquirer_Live_Price, exchangeRate = [original_aid_list['wiccloseest'],
                                                                    float(original_aid_list['tgt_total_divs'] if original_aid_list['tgt_total_divs'] else None),
                                                                    float(original_aid_list['acquirer_live_price'] if original_aid_list['acquirer_live_price'] else None),
                                                                    float(original_aid_list['exchange_rate']) if original_aid_list['exchange_rate'] else 0]
    Acq_Total_Divs, Deal_Value_Calc, Acquirer_Downside, Acquirer_Breakspread, Stock_Value = [float(original_aid_list['acq_total_divs']) if original_aid_list['acq_total_divs'] else None,
                                                                                            float(original_aid_list['deal_value']),
                                                                                            float(original_aid_list['acquirer_downside']) if original_aid_list['acquirer_downside'] else None,
                                                                                            float(original_aid_list['acq_breakspread']) if original_aid_list['acq_breakspread'] else None,
                                                                                            float(original_aid_list['stock_value']) ]
    cash_terms = original_aid_list['cash_terms']
    Stock_Ratio = float(Stock_Terms.split(' Aqr')[0]) if Stock_Terms else 0
    numb_div = {'Monthly': 30, 'Quarter': 90, 'Semi-Anl': 180, 'Annual': 365}
    # cash
    if 'cash_terms' in frontend_change_keys or 'stock_terms' in frontend_change_keys:
        if 'cash_terms' in frontend_change_keys:
            if cash_terms:
                cash_terms_split = float(cash_terms.split('/')[0]) if '/' in cash_terms else 0
                cash_value = cash_terms_split
            else:
                cash_terms = '0'
                cash_value = '0'
        if 'stock_terms' in frontend_change_keys:
            if 'Aqr' in Stock_Terms:
                stock_terms_split = float(Stock_Terms.split(' Aqr')[0]) if Stock_Terms else 0
                Stock_Ratio = stock_terms_split
            else:
                cash_value_special = float(Stock_Terms.split('/Tgt')[0].replace('USD ', '')) if Stock_Terms else 0
                Stock_Ratio = 0
                cash_value = max(cash_value_special, cash_value)
    # stock terms
    if 'stock_terms' in frontend_change_keys or 'cash_terms' in frontend_change_keys or 'acq_div_days' in frontend_change_keys:
        if Stock_Ratio == 0:
            Acq_Div = None
        elif Acq_Div_Days and Acq_Div_Freq:
            # can be timedelta or float
            try:
                Acq_Div = math.ceil((Acq_Div_Days / numb_div.get(Acq_Div_Freq, 0)).days)
            except:
                Acq_Div = math.ceil(Acq_Div_Days / numb_div.get(Acq_Div_Freq, 0))
        if Acq_Div_Amt == None:
            Acq_Div_Amt = 0
        if Acq_Div and Acq_Div_Amt:
            Acq_Total_Divs = round(Acq_Div * Acq_Div_Amt, 4)
        else:
            Acq_Total_Divs = 0
        if Stock_Ratio != None and Acquirer_Live_Price:
            Stock_Value = Stock_Ratio * Acquirer_Live_Price if not exchangeRate else Stock_Ratio * (Acquirer_Live_Price * float(exchangeRate))
        else:
            Stock_Value = 0
        #
        if Stock_Ratio == 0:
            Acquirer_Breakspread = 0
        elif Acquirer_Downside != None and Acquirer_Live_Price != None:
            Acquirer_Breakspread = round((Acquirer_Downside - Acquirer_Live_Price) * Stock_Ratio, 4)
        #
        aid_obj.acquirer_downside = Acquirer_Downside
        aid_obj.acq_breakspread = Acquirer_Breakspread
        aid_obj.stock_value = Stock_Value
    # wiccloseest
    if 'wiccloseest' in frontend_change_keys:
        try:
            days_to_close = (wic_close_est - datetime.date.today()).days
        except:
            days_to_close = None
        aid_obj.days_to_close = days_to_close
        aid_obj.wiccloseest = wic_close_est
    # acq tgt, div amt
    if 'tgt_div' in frontend_change_keys or 'acq_div' in frontend_change_keys or 'acq_div_amt' in frontend_change_keys or 'tgt_div_amt' in frontend_change_keys or 'acq_total_divs' in frontend_change_keys or 'tgt_total_divs' in frontend_change_keys or 'stock_terms' in frontend_change_keys:
        Net_Div_Impact,Acq_Total_Divs,Acq_Div_Amt = calculate_net_div_impact(Acq_Div_Amt,Acq_Div,Tgt_Total_Divs,Stock_Ratio)
    # deal value calcs
    if 'special_dividend' in frontend_change_keys or 'stock_terms' in frontend_change_keys or 'cash_value' in frontend_change_keys or 'cash_terms' in frontend_change_keys or 'net_div_impact' in frontend_change_keys:
        Deal_Value_Calc = calculate_deal_value(payment_type,cash_value,Stock_Value,Net_Div_Impact,Special_Dividend)
        slack_string += ' Deal_value ' + str(Deal_Value_Calc) + ' '
    # save
    aid_obj.cash_value = cash_value
    aid_obj.cash_terms = cash_terms
    aid_obj.net_div_impact = Net_Div_Impact
    aid_obj.special_dividend = Special_Dividend
    aid_obj.stock_terms = Stock_Terms
    aid_obj.acq_div_days = Acq_Div_Days
    aid_obj.acq_div = Acq_Div
    aid_obj.acq_total_divs = Acq_Total_Divs
    aid_obj.acq_div_amt = Acq_Div_Amt
    aid_obj.tgt_total_divs = Tgt_Total_Divs
    aid_obj.deal_value = Deal_Value_Calc
    aid_obj.save()

    return slack_string


