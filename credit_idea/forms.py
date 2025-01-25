import pandas as pd

from django import forms
from django.db import connection

from credit_idea.models import CreditIdea, CreditIdeaComments


BAGDE_INFO_CLASS = 'badge badge-default badge-info'
BADGE_SUCCESS_CLASS = 'badge badge-default badge-success'
CUSTOM_SELECT_CLASS = 'custom-select form-control input-lg'
DATE_PICKER_CLASS = 'form-control'
FORM_CONTROL_CLASS = 'form-control input-lg'


def get_arb_tradegroup_list():
    query = 'SELECT DISTINCT TradeGroup FROM wic.daily_flat_file_db where Fund = "ARB" and Sleeve = "Merger Arbitrage";'
    deal_names_df = pd.read_sql_query(query, con=connection)
    deal_names = set()
    if not deal_names_df.empty:
        deal_names = [i.upper() for i in list(deal_names_df['TradeGroup'])]
        deal_names = set(deal_names)
    tg_list = CreditIdea.objects.values_list('arb_tradegroup', flat=True)
    tg_list = [i.upper() for i in tg_list]
    tg_list = set(tg_list)
    deal_names = deal_names.union(tg_list)
    deal_names = sorted(deal_names)
    deal_names = [('', '')] + [(i.upper(), i.upper()) for i in deal_names if i]
    deal_names += [('OTHER', 'OTHER')]
    return deal_names

class CreditIdeaForm(forms.ModelForm):
    """
    Form for Credit Idea Database
    """
    def __init__(self, *args, **kwargs):
        super(CreditIdeaForm, self).__init__(*args, **kwargs)
        arb_tradegroup_choices = get_arb_tradegroup_list()
        self.fields['arb_tradegroup'].choices = arb_tradegroup_choices
        self.fields['arb_tradegroup'].widget.choices = arb_tradegroup_choices

    class Meta:
        model = CreditIdea
        fields = ['analyst', 'arb_tradegroup', 'other_tradegroup', 'deal_bucket', 'deal_strategy_type', 'catalyst',
                  'catalyst_tier', 'target_sec_cusip', 'coupon', 'hedge_sec_cusip', 'estimated_closing_date',
                  'upside_price', 'downside_price', 'deal_category']
        labels = {
            "analyst": "Analyst",
            "arb_tradegroup": "ARB TradeGroup",
            "other_tradegroup": "Other TradeGroup",
            "deal_bucket": "Deal Bucket",
            "deal_strategy_type": "Deal Strategy Type",
            "catalyst": "Catalyst",
            "catalyst_tier": "Catalyst Tier",
            "target_sec_cusip": "Target Sec CUSIP",
            "coupon": "Coupon",
            "hedge_sec_cusip": "Hedge Sec CUSIP",
            "estimated_closing_date": "Estimated Closing Date",
            "upside_price": "Upside Price",
            "downside_price": "Downside Price",
            "deal_category": "Deal Category"
        }
        widgets = {
            'id': forms.HiddenInput(),
            'analyst': forms.TextInput(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                              'id': 'analyst', 'required': True}),
            'arb_tradegroup': forms.Select(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                  'id': 'arb_tradegroup', 'required': True,
                                                  'onChange': 'display_other_tradegroup()'},
                                           choices=get_arb_tradegroup_list()),
            'other_tradegroup': forms.TextInput(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                       'id': 'other_tradegroup', 'required': False}),
            'deal_bucket': forms.Select(attrs={'class': CUSTOM_SELECT_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                               'id': 'deal_bucket', 'required': True}),
            "deal_strategy_type": forms.Select(attrs={'class': CUSTOM_SELECT_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                      'id': 'deal_strategy_type', 'required': True}),
            "catalyst": forms.Select(attrs={'class': CUSTOM_SELECT_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                            'id': 'catalyst', 'required': True}),
            "catalyst_tier": forms.Select(attrs={'class': CUSTOM_SELECT_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                 'id': 'catalyst_tier', 'required': True}),
            "target_sec_cusip": forms.TextInput(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                       'id': 'target_sec_cusip', 'required': False}),
            "coupon": forms.TextInput(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                             'id': 'coupon', 'required': False}),
            "hedge_sec_cusip": forms.TextInput(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                      'id': 'hedge_sec_cusip', 'required': False}),
            "estimated_closing_date": forms.DateInput(attrs={'type': 'date', 'class': DATE_PICKER_CLASS,
                                                             'label_class': BAGDE_INFO_CLASS,
                                                             'id': 'estimated_closing_date', 'required': False}),
            "upside_price": forms.TextInput(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                   'id': 'upside_price', 'required': False}),
            "downside_price": forms.TextInput(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                     'id': 'downside_price', 'required': False}),
            "deal_category": forms.Select(attrs={'class': CUSTOM_SELECT_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                 'id': 'deal_category', 'required': False}),
        }


class CreditIdeaCommentsForm(forms.ModelForm):
    """
    Form for Credit Idea Comments Database
    """
    class Meta:
        model = CreditIdeaComments
        fields = ['summary_comments', 'press_release_comments', 'proxy_statement_comments',
                  'change_of_control_comments', 'restricted_payments_comments', 'liens_indebtedness_comments',
                  'other_comments']
        labels = {
            "summary_comments": "Summary",
            "press_release_comments": "Press Release / Presentation / Merger Agreement",
            "proxy_statement_comments": "Proxy Statement",
            "change_of_control_comments": "Change of Control",
            "restricted_payments_comments": "Restricted Payments",
            "liens_indebtedness_comments": "Limitation on Liens / Indebtedness",
            "other_comments": "Other Covenants / Miscellaneous"
        }
        widgets = {
            "summary_comments": forms.Textarea(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                      'id': 'summary_comments', 'rows': 8, 'cols': 40, 'required': False}),
            "press_release_comments": forms.Textarea(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                            'id': 'press_release_comments', 'rows': 8, 'cols': 40,
                                                            'required': False}),
            "proxy_statement_comments": forms.Textarea(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                              'id': 'proxy_statement_comments', 'rows': 8, 'cols': 40,
                                                              'required': False}),
            "change_of_control_comments": forms.Textarea(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                                'id': 'change_of_control_comments', 'rows': 8, 'cols': 40,
                                                                'required': False}),
            "restricted_payments_comments": forms.Textarea(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                                  'id': 'restricted_payments_comments', 'rows': 8, 'cols': 40,
                                                                  'required': False}),
            "liens_indebtedness_comments": forms.Textarea(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                                 'id': 'liens_indebtedness_comments', 'rows': 8, 'cols': 40,
                                                                 'required': False}),
            "other_comments": forms.Textarea(attrs={'class': FORM_CONTROL_CLASS, 'label_class': BADGE_SUCCESS_CLASS,
                                                    'id': 'other_comments', 'rows': 8, 'cols': 40, 'required': False}),
        }
