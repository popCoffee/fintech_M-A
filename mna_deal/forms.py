from datetime import date

from django import forms
from django.conf import settings
from django_slack import slack_message

from risk.models import MA_Deals
from risk_reporting.models import FormulaeBasedDownsides
from slack_utils import get_channel_name


BADGE_SUCCESS_CLASS = 'badge badge-default badge-success'
BADGE_DARK_CLASS = 'badge badge-default badge-dark'
DATE_PICKER_CLASS = 'form-control'
CUSTOM_SELECT_CLASS = 'custom-select form-control input-lg'
FORM_CONTROL_CLASS = 'form-control input-lg'
MAX_ACTUAL_DATE = date.today().strftime('%Y-%m-%d')

POSITION_ACQUIRER_CHOICES = [
    ('no', 'No'),
    ('yes', 'Yes')
]

DEAL_CURRENCY_CHOICES = [
    ('AUD', 'AUD'),
    ('CAD', 'CAD'),
    ('CHF', 'CHF'),
    ('EUR', 'EUR'),
    ('GBP', 'GBP'),
    ('HKD', 'HKD'),
    ('JPY', 'JPY'),
    ('KRW', 'KRW'),
    ('USD', 'USD'),
    ('PLN', 'PLN'),
    ('NOK', 'NOK'),
    ('SEK', 'SEK'),
    ('ZAR', 'ZAR'),
]

class CreateMaDealsForm(forms.Form):
    """
    Form for adding new M&A Deal
    """
    deal_name = forms.CharField(required=True, label="Deal Name", max_length=100,
                                widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                              'label_class': BADGE_DARK_CLASS, 'id': 'deal_name'
                                                              }))  #'placeholder': 'IBM - RHT'
    unaffected_date = forms.CharField(required=True, label="Unaffected Date  (YYYY-mm-dd)",
                                       widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                     'label_class': BADGE_DARK_CLASS,
                                                                     'id': 'unaffected_date'}))
    analyst = forms.CharField(required=False, label="Analyst", max_length=50,
                              widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                            'label_class': BADGE_SUCCESS_CLASS, 'id': 'analyst',
                                                            'placeholder': '', 'hide': 'true'}))
    target_ticker = forms.CharField(required=True, label="target_ticker", max_length=50,
                                    widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                  'label_class': BADGE_SUCCESS_CLASS, 'id': 'target_ticker',
                                                                  'placeholder': 'AAPL US EQUITY'}))
    target_name = forms.CharField(required=True, label="Target Name", max_length=100,
                                   widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                 'label_class': BADGE_SUCCESS_CLASS,
                                                                 'id': 'target_name'}))
    acquirer_ticker = forms.CharField(required=False, label="Acquirer Ticker", max_length=50,
                                      widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                    'label_class': BADGE_SUCCESS_CLASS,
                                                                    'id': 'acquirer_ticker', 'placeholder': 'GOOGL US EQUITY'}))
    acquirer_name = forms.CharField(required=True, label="acquirer name", max_length=100,
                                     widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                   'label_class': BADGE_SUCCESS_CLASS,
                                                                   'id': 'acquirer_name'}))
    deal_cash_terms = forms.CharField(required=False, label="Deal Cash Terms", max_length=50,
                                       widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                     'label_class': BADGE_SUCCESS_CLASS,
                                                                     'id': 'deal_cash_terms', 'placeholder': '0.00'}))
    deal_share_terms = forms.CharField(required=False, label="Deal Share Terms", max_length=50,
                                        widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                      'label_class': BADGE_SUCCESS_CLASS,
                                                                      'id': 'deal_share_terms', 'placeholder': '0.00'}))
    deal_value = forms.CharField(required=True, label="Deal Value", max_length=50,
                                  widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                'label_class': BADGE_SUCCESS_CLASS,
                                                                'id': 'deal_value', 'placeholder': '000.00'}))
    deal_currency = forms.CharField(required=True, label="Deal Currency", max_length=50,
                                     widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                   'label_class': BADGE_SUCCESS_CLASS,
                                                                   'id': 'deal_currency'}))
    short_rebate = forms.CharField(required=False, label="Short Rebate", max_length=50,
                                    widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                  'label_class': BADGE_SUCCESS_CLASS,
                                                                  'id': 'short_rebate','hide': 'true', 'placeholder': '0.00'}))

    stub_cvr_value = forms.CharField(required=False, label="Stub / CVR Value", max_length=50,
                                      widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                    'label_class': BADGE_SUCCESS_CLASS,'hide': 'true',
                                                                    'id': 'stub_cvr_value', 'placeholder': '0.00'}))
    is_cross_border_deal = forms.BooleanField(required=False, label="Cross-Border Deal",
                                              widget=forms.CheckboxInput(attrs={'class': 'form-check-input',
                                                                                'label_class': BADGE_SUCCESS_CLASS,
                                                                                'id': 'is_cross_border_deal', 'hide': 'true',
                                                                                'onChange': 'crossBorderDealChange()'}))

    ## new fields
    announced_date = forms.CharField(required=False, label="Announced Date", max_length=50,
                                  widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                'label_class': BADGE_SUCCESS_CLASS,
                                                                'id': 'announced_date','placeholder': '',}))
    origination_date = forms.CharField(required=True, label="Origination Date", max_length=50,
                                      widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                    'label_class': BADGE_SUCCESS_CLASS,
                                                                    'id': 'origination_date'}))
    target_price = forms.CharField(required=True, label="Target Price", max_length=50,
                                  widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                'label_class': BADGE_SUCCESS_CLASS,
                                                                'id': 'target_price'}))
    gross_spread = forms.CharField(required=True, label="Gross Spread", max_length=50,
                                    widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                  'label_class': BADGE_SUCCESS_CLASS,
                                                                  'id': 'gross_spread'}))
    spread_premium = forms.CharField(required=True, label="Spread Premium %", max_length=50,
                                    widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                  'label_class': BADGE_SUCCESS_CLASS,
                                                                  'id': 'spread_premium'}))
    # date_unaffected = forms.FloatField(required=False, label="date_unaffected",
    #                                   widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
    #                                                                 'label_class': BADGE_SUCCESS_CLASS,
    #                                                                 'id': 'date_unaffected'}))
    unaffected_price = forms.CharField(required=False, label="Unaffected Price", max_length=50,
                                      widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                    'label_class': BADGE_SUCCESS_CLASS,
                                                                    'id': 'unaffected_price'}))
    unaffected_price_30 = forms.CharField(required=False, label="Unaffected Price 30", max_length=100,
                                      widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                    'label_class': BADGE_SUCCESS_CLASS,
                                                                    'id': 'unaffected_price_30'}))
    unaffected_price_90 = forms.CharField(required=False, label="Unaffected Price 90", max_length=100,
                                           widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                         'label_class': BADGE_SUCCESS_CLASS,
                                                                         'id': 'unaffected_price_90'}))
    position_in_acquirer = forms.CharField(label="Position in Acquirer", max_length=50,
                                           widget=forms.Select(choices=POSITION_ACQUIRER_CHOICES,
                                                               attrs={'class': CUSTOM_SELECT_CLASS,
                                                                      'label_class': BADGE_SUCCESS_CLASS,
                                                                      'id': 'position_in_acquirer', 'initial': ''}))
    payment_type = forms.CharField(required=False, label="payment_type", max_length=50,
                                    widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                  'label_class': BADGE_SUCCESS_CLASS,
                                                                  'id': 'payment_type', 'hide': 'true'}))
    Announced_Premium = forms.CharField(required=False, label="Announced_Premium", max_length=50,
                                    widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                  'label_class': BADGE_SUCCESS_CLASS,
                                                                  'id': 'Announced_Premium', 'hide': 'true'}))
    stock_terms_str = forms.CharField(required=False, label="stock_terms_str", max_length=100,
                                    widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                  'label_class': BADGE_SUCCESS_CLASS,
                                                                  'id': 'stock_terms_str', 'hide': 'true'}))
    Cash_Value = forms.CharField(required=False, label="Cash_Value", max_length=50,
                                    widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                  'label_class': BADGE_SUCCESS_CLASS,
                                                                  'id': 'Cash_Value', 'hide': 'true'}))
    exchange_rate = forms.CharField(required=False, label="exchange_rate", max_length=50,
                                  widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                'label_class': BADGE_SUCCESS_CLASS,
                                                                'id': 'exchange_rate', 'hide': 'true'}))
    outlier = forms.CharField(required=False, label="outlier", max_length=50,
                                 widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                               'label_class': BADGE_SUCCESS_CLASS,
                                                               'id': 'outlier', 'hide': 'true'}))
    base_case = forms.CharField(required=False, label="base_case", max_length=50,
                                    widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                  'label_class': BADGE_SUCCESS_CLASS,
                                                                  'id': 'base_case', 'hide': 'true'}))
    target_currency = forms.CharField(required=False, label="target_currency", max_length=50,
                                widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                              'label_class': BADGE_SUCCESS_CLASS,
                                                              'id': 'target_currency', 'hide': 'true'}))
    unaffected_downsides = forms.CharField(required=False, label="unaffected_downsides", max_length=50,
                                      widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                    'label_class': BADGE_SUCCESS_CLASS,
                                                                    'id': 'unaffected_downsides', 'hide': 'true'}))
    cash_terms = forms.CharField(required=False, label="Cash Terms", max_length=50,
                                 widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                               'label_class': BADGE_SUCCESS_CLASS,
                                                               'id': 'cash_terms', 'hide': 'true'}))

    Stock_Value = forms.CharField(required=False, label="Stock Value", max_length=50,
                                  widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                'label_class': BADGE_SUCCESS_CLASS,
                                                                'id': 'Stock_Value', 'hide': 'true'}))

    Stock_Ratio = forms.CharField(required=False, label="Stock Ratio", max_length=50,
                                  widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                'label_class': BADGE_SUCCESS_CLASS,
                                                                'id': 'Stock_Ratio', 'hide': 'true'}))

    wic_close_est = forms.CharField(required=False, label="Wic Close Est", max_length=50,
                                    widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                  'label_class': BADGE_SUCCESS_CLASS,
                                                                  'id': 'wic_close_est', 'hide': 'true'}))

    Tgt_Div_Next_Ex = forms.CharField(required=False, label="Tgt Div Next Ex", max_length=50,
                                      widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                    'label_class': BADGE_SUCCESS_CLASS,
                                                                    'id': 'Tgt_Div_Next_Ex', 'hide': 'true'}))

    Tgt_Div_Days = forms.CharField(required=False, label="Tgt Div Days", max_length=50,
                                   widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                 'label_class': BADGE_SUCCESS_CLASS,
                                                                 'id': 'Tgt_Div_Days', 'hide': 'true'}))

    Tgt_Div_Freq = forms.CharField(required=False, label="Tgt Div Freq", max_length=50,
                                   widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                 'label_class': BADGE_SUCCESS_CLASS,
                                                                 'id': 'Tgt_Div_Freq', 'hide': 'true'}))

    Tgt_Div = forms.CharField(required=False, label="Tgt Div", max_length=50,
                              widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                            'label_class': BADGE_SUCCESS_CLASS,
                                                            'id': 'Tgt_Div', 'hide': 'true'}))

    Tgt_Div_Amt = forms.CharField(required=False, label="Tgt Div Amt", max_length=50,
                                  widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                'label_class': BADGE_SUCCESS_CLASS,
                                                                'id': 'Tgt_Div_Amt', 'hide': 'true'}))

    Tgt_Total_Divs = forms.CharField(required=False, label="Tgt Total Divs", max_length=50,
                                     widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                   'label_class': BADGE_SUCCESS_CLASS,
                                                                   'id': 'Tgt_Total_Divs', 'hide': 'true'}))

    Acq_Div_Next_Ex = forms.CharField(required=False, label="Acq Div Next Ex", max_length=50,
                                      widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                    'label_class': BADGE_SUCCESS_CLASS,
                                                                    'id': 'Acq_Div_Next_Ex', 'hide': 'true'}))

    Acq_Div_Days = forms.CharField(required=False, label="Acq Div Days", max_length=50,
                                   widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                 'label_class': BADGE_SUCCESS_CLASS,
                                                                 'id': 'Acq_Div_Days', 'hide': 'true'}))

    Acq_Div_Freq = forms.CharField(required=False, label="Acq Div Freq", max_length=50,
                                   widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                 'label_class': BADGE_SUCCESS_CLASS,
                                                                 'id': 'Acq_Div_Freq', 'hide': 'true'}))

    Acq_Div = forms.CharField(required=False, label="Acq Div", max_length=50,
                              widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                            'label_class': BADGE_SUCCESS_CLASS,
                                                            'id': 'Acq_Div', 'hide': 'true'}))

    Acq_Div_Amt = forms.CharField(required=False, label="Acq Div Amt", max_length=50,
                                  widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                'label_class': BADGE_SUCCESS_CLASS,
                                                                'id': 'Acq_Div_Amt', 'hide': 'true'}))

    Acq_Total_Divs = forms.CharField(required=False, label="Acq Total Divs", max_length=50,
                                     widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                   'label_class': BADGE_SUCCESS_CLASS,
                                                                   'id': 'Acq_Total_Divs', 'hide': 'true'}))

    Net_Div_Impact = forms.CharField(required=False, label="Net Div Impact", max_length=50,
                                     widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                   'label_class': BADGE_SUCCESS_CLASS,
                                                                   'id': 'Net_Div_Impact', 'hide': 'true'}))

    days_to_close = forms.CharField(required=False, label="Days to Close", max_length=50,
                                    widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                  'label_class': BADGE_SUCCESS_CLASS,
                                                                  'id': 'days_to_close', 'hide': 'true'}))
    acquirer_price = forms.CharField(required=False, label="acquirer price", max_length=50,
                                    widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                  'label_class': BADGE_SUCCESS_CLASS,
                                                                  'id': 'acquirer_price', 'hide': 'true'}))

    rebate_carry = forms.CharField(required=False, label="Rebate Carry", max_length=50,
                                  widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                'label_class': BADGE_SUCCESS_CLASS,
                                                                'id': 'rebate_carry', 'hide': 'true'}))
    risk_free_rate = forms.CharField(required=False, label="risk_free_rate", max_length=50,
                                   widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                 'label_class': BADGE_SUCCESS_CLASS,
                                                                 'id': 'risk_free_rate', 'hide': 'True'}))
    short_rebate = forms.CharField(required=False, label="short_rebate", max_length=50,
                                   widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
                                                                 'label_class': BADGE_SUCCESS_CLASS,
                                                                 'id': 'short_rebate', 'hide': 'true'}))


    def clean(self):
        """
        Set the following values to 0.0 if the User did not give any input
        """
        cleaned_data = super().clean()
        if not cleaned_data.get('deal_cash_terms'):
            cleaned_data['deal_cash_terms'] = 0.0
        if not cleaned_data.get('deal_share_terms'):
            cleaned_data['deal_share_terms'] = 0.0
        if not cleaned_data.get('target_dividends'):
            cleaned_data['target_dividends'] = 0.0
        if not cleaned_data.get('acquirer_dividends'):
            cleaned_data['acquirer_dividends'] = 0.0
        if not cleaned_data.get('short_rebate'):
            cleaned_data['short_rebate'] = 0.0
        if not cleaned_data.get('fx_carry_percent'):
            cleaned_data['fx_carry_percent'] = 0.0
        if not cleaned_data.get('stub_cvr_value'):
            cleaned_data['stub_cvr_value'] = 0.0
        if not cleaned_data.get('acquirer_upside'):
            cleaned_data['acquirer_upside'] = 0.0
        if not cleaned_data.get('loss_tolerance_percentage_of_limit'):
            cleaned_data['loss_tolerance_percentage_of_limit'] = 0.0
        target_ticker = cleaned_data.get('target_ticker')
        if target_ticker:
            if 'equity' not in str(target_ticker).lower():
                cleaned_data['target_ticker'] = target_ticker.upper() + ' EQUITY'
        acquirer_ticker = cleaned_data.get('acquirer_ticker')
        if acquirer_ticker:
            if 'equity' not in str(acquirer_ticker).lower():
                cleaned_data['acquirer_ticker'] = acquirer_ticker.upper() + ' EQUITY'
        is_cross_border_deal = cleaned_data.get('is_cross_border_deal')
        if not is_cross_border_deal:
            cleaned_data['deal_currency'] = 'USD'
        else:
            cleaned_data['deal_currency'] = cleaned_data.get('deal_currency').strip()
        return cleaned_data


    def is_valid(self):
        """
        Validate the fields and display error message if not valid
        """
        valid = super(CreateMaDealsForm, self).is_valid()
        # valid = True
        if not valid:
            return valid
        cleaned_data = self.cleaned_data
        deal_name = cleaned_data.get('deal_name', "")

        # if cleaned_data.get('position_in_acquirer').lower() == 'yes' and not cleaned_data.get('acquirer_ticker'):
        if not cleaned_data.get('acquirer_ticker'):
            valid = False
            self._errors['acquirer_ticker'] = 'Acquirer Ticker is required since Position Acquirer is marked as Yes'
        if not cleaned_data.get('deal_cash_terms') and not cleaned_data.get('deal_share_terms'):
            valid = False
            self._errors['deal_cash_terms'] = 'Either Deal Cash Terms or Deal Share Terms should be present'
            self._errors['deal_share_terms'] = 'Either Deal Cash Terms or Deal Share Terms should be present'
        deal_name = cleaned_data.get('deal_name')
        slack_dict = {'deal_name': deal_name, 'action_id': cleaned_data.get('action_id'),
                      'target_ticker': cleaned_data.get('target_ticker'), 'acquirer_ticker': cleaned_data.get('acquirer_ticker'),
                      'deal_cash_terms': cleaned_data.get('deal_cash_terms'), 'deal_share_terms': cleaned_data.get('deal_share_terms'),
                      'deal_value': cleaned_data.get('deal_value'), 'target_dividends': cleaned_data.get('target_dividends'),
                      'acquirer_dividends': cleaned_data.get('acquirer_dividends'), 'short_rebate': cleaned_data.get('short_rebate'),
                      'fx_carry_percent': cleaned_data.get('fx_carry_percent'), 'stub_cvr_value': cleaned_data.get('stub_cvr_value'),
                      'acquirer_upside': cleaned_data.get('acquirer_upside'), 'risk_limit': cleaned_data.get('risk_limit'),
                      'loss_tolerance_percentage_of_limit': cleaned_data.get('loss_tolerance_percentage_of_limit'),
                      'position_in_acquirer': cleaned_data.get('position_in_acquirer')}
        if MA_Deals.objects.filter(deal_name=deal_name).exists() and \
                FormulaeBasedDownsides.objects.filter(TradeGroup=deal_name).exists():
            # valid = False # comment to allow for editing of data in DB
            self._errors['deal_name'] = '{deal_name} is already present in the M&A Deal Database and Downside Formulae'.format(deal_name=deal_name)
            slack_dict['message'] = 'ERROR! Deal adready present in MA Deals and Downside Formulae'
            slack_message('new_mna_deal_notify.slack', slack_dict, channel=get_channel_name('new-mna-deals'),
                          token=settings.SLACK_TOKEN, name='ESS_IDEA_DB_ERROR_INSPECTOR')

        return valid


class EditMaDealsForm(forms.Form):
    """
    Form for editting new M&A Deal
    """
    action_id = forms.CharField(required=True, label="Action Id", max_length=100,
                                widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                              'label_class': BADGE_DARK_CLASS, 'id': 'action_id_form'
                                                              }))
    deal_name = forms.CharField(required=True, label="Deal Name", max_length=100,
                                widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                              'label_class': BADGE_DARK_CLASS, 'id': 'deal_name'
                                                              }))
    target_name = forms.CharField(required=False, label="Target Name", max_length=100,
                                   widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                 'label_class': BADGE_DARK_CLASS,
                                                                 'id': 'target_name'}))
    acquirer_name = forms.CharField(required=False, label="Acquirer Name", max_length=100,
                                     widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                   'label_class': BADGE_DARK_CLASS,
                                                                   'id': 'acquirer_name'}))
    target_ticker = forms.CharField(required=True, label="Target Ticker", max_length=50,
                                    widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                  'label_class': BADGE_DARK_CLASS, 'id': 'target_ticker'}))
    acquirer_ticker = forms.CharField(required=False, label="Acquirer Ticker", max_length=50,
                                      widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                    'label_class': BADGE_DARK_CLASS,
                                                                    'id': 'acquirer_ticker'}))
    # unaffected_price = forms.CharField(required=False, label="Unaffected Price", max_length=50,
    #                                   widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
    #                                                                 'label_class': BADGE_DARK_CLASS,
    #                                                                 'id': 'unaffected_price'}))
    # cash_value = forms.CharField(required=False, label="Cash Value", max_length=50,
    #                                 widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
    #                                                               'label_class': BADGE_DARK_CLASS,
    #                                                               'id': 'cash_value'}))
    # stock_value = forms.CharField(required=False, label="Stock Value (decimal)", max_length=50,
    #                               widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
    #                                                             'label_class': BADGE_DARK_CLASS,
    #                                                             'id': 'stock_value'}))
    cash_terms = forms.CharField(required=False, label="Cash Terms  (11.50/sh.)", max_length=50,
                                 widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                               'label_class': BADGE_DARK_CLASS,
                                                               'id': 'cash_terms'}))
    stock_terms = forms.CharField(required=False, label="Stock Terms (1.4500 Aqr sh./Tgt sh.)", max_length=50,
                                  widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                'label_class': BADGE_DARK_CLASS,
                                                                'id': 'stock_terms'}))
    deal_value = forms.CharField(required=False, label="Deal Value (no edit)", max_length=50, disabled=True,
                                 widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                               'label_class': BADGE_DARK_CLASS,
                                                               'id': 'deal_value'}))
    unaffected_date = forms.CharField(required=False, label="Unaffected Date (YYYY-mm-dd)",
                                      widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                    'label_class': BADGE_DARK_CLASS,
                                                                    'id': 'unaffected_date'}))
    announced_date = forms.CharField(required=False, label="Announced Date", max_length=50,
                                  widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                'label_class': BADGE_DARK_CLASS,
                                                                'id': 'announced_date','placeholder': '',}))
    termination_date = forms.CharField(required=False, label="Termination Date", max_length=50,
                                       widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                     'label_class': BADGE_DARK_CLASS,
                                                                     'id': 'termination_date'}))
    wiccloseest = forms.CharField(required=False, label="Wic Close Estimate Date", max_length=50,
                                       widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                     'label_class': BADGE_DARK_CLASS,
                                                                     'id': 'wiccloseest'}))
    drop_dead_date = forms.CharField(required=False, label="Drop Dead Date", max_length=50,
                                  widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                'label_class': BADGE_DARK_CLASS,
                                                                'id': 'drop_dead_date'}))
    # target_live_price = forms.CharField(required=False, label="Target Price", max_length=50,
    #                               widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
    #                                                             'label_class': BADGE_DARK_CLASS,
    #                                                             'id': 'target_live_price'}))
    # announced_premium = forms.CharField(required=False, label="Announced_Premium", max_length=50,
    #                                 widget=forms.TextInput(attrs={'class': CUSTOM_SELECT_CLASS,
    #                                                               'label_class': BADGE_DARK_CLASS,
    #                                                               'id': 'announced_premium', 'hide': 'true'}))
    special_dividend = forms.CharField(required=False, label="Special Dividend", max_length=50,
                                       widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                     'label_class': BADGE_DARK_CLASS,
                                                                     'id': 'special_dividend'}))
    tgt_div = forms.CharField(required=False, label="# of Target Dividends", max_length=50,
                              widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                            'label_class': BADGE_DARK_CLASS,
                                                            'id': 'tgt_div'}))
    tgt_div_amt = forms.CharField(required=False, label="Target Dividend Amount", max_length=50,
                                  widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                'label_class': BADGE_DARK_CLASS,
                                                                'id': 'tgt_div_amt'}))
    acq_div = forms.CharField(required=False, label="# of Acquirer Dividends", max_length=50,
                              widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                            'label_class': BADGE_DARK_CLASS,
                                                            'id': 'acq_div'}))
    acq_div_amt = forms.CharField(required=False, label="Acquirer Dividend Amount", max_length=50,
                                  widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                'label_class': BADGE_DARK_CLASS,
                                                                'id': 'acq_div_amt'}))
    acq_total_divs = forms.CharField(required=False, label="Expected Acquirer Dividends  (no edit)", max_length=50, disabled=True,
                                     widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                   'label_class': BADGE_DARK_CLASS,
                                                                   'id': 'acq_total_divs'}))
    tgt_total_divs = forms.CharField(required=False, label="Expected Target Dividends  (no edit)", max_length=50, disabled=True,
                                     widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                   'label_class': BADGE_DARK_CLASS,
                                                                   'id': 'tgt_total_divs'}))
    net_div_impact = forms.CharField(required=False, label="Net Div Impact  (no edit)", max_length=50, disabled=True,
                                     widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                   'label_class': BADGE_DARK_CLASS,
                                                                   'id': 'net_div_impact'}))
    manual_frontend_edit = forms.CharField(required=False, label="manual_edit", max_length=50,
                                     widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                                   'label_class': BADGE_DARK_CLASS,
                                                                   'id': 'manual_edit', 'hide':'true', 'initial':1 }))
    peer_index = forms.CharField(required=False, label="Peer Index", max_length=50,
                                widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                              'label_class': BADGE_DARK_CLASS,
                                                              'id': 'peer_index','placeholder':"e.g, ARBCCOMP INDEX"}))
    spread_index = forms.CharField(required=False, label="Spread Index", max_length=50,
                                widget=forms.TextInput(attrs={'class': FORM_CONTROL_CLASS,
                                                              'label_class': BADGE_DARK_CLASS,
                                                              'id': 'spread_index','placeholder':"e.g, ARNCPXY INDEX"}))






