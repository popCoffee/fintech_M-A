import pandas as pd

from django import forms

from pnl_attribution.dfutils import convert_file_to_df
FORM_CONTROL_CLASS = 'form-control input-lg'
BADGE_SUCCESS_CLASS = 'badge badge-default badge-success'
BADGE_DARK_CLASS = 'badge badge-default badge-dark'
DATE_PICKER_CLASS = 'form-control'

FUND_OPTIONS = (
    ('ARBETF', 'ARBETF'),
)


class ETFTrackingErrorInputForm(forms.Form):
    """ Form for ETF Tracking Error """
    fund = forms.CharField(required=True, label='Fund', widget=forms.Select(
        choices=FUND_OPTIONS, attrs={'label_class': BADGE_SUCCESS_CLASS, 'id': 'fund_choice',
                                     'class': 'form-control custom-select'}))

    start_date = forms.DateField(required=True, label="Start Date",
                                 widget=forms.DateInput(attrs={'type': 'date', 'class': DATE_PICKER_CLASS,
                                                               'label_class': BADGE_DARK_CLASS, 'id': 'start_date'}))
    end_date = forms.DateField(required=True, label="End Date",
                               widget=forms.DateInput(attrs={'type': 'date', 'class': DATE_PICKER_CLASS,
                                                             'label_class': BADGE_DARK_CLASS, 'id': 'end_date',
                                                             }))

    def is_valid(self):
        """
        Validate the fields and display error message if not valid
        """
        valid = super(ETFTrackingErrorInputForm, self).is_valid()

        return valid

class ETFDailyTrackingInputForm(forms.Form):
    """ Form for ETF Tracking Error """
    fund = forms.CharField(required=True, label='Fund', widget=forms.Select(
        choices=FUND_OPTIONS, attrs={'label_class': BADGE_SUCCESS_CLASS, 'id': 'fund_choice',
                                     'class': 'form-control custom-select'}))

    start_date = forms.DateField(required=True, label="Start Date",
                                 widget=forms.DateInput(attrs={'type': 'date', 'class': DATE_PICKER_CLASS,
                                                               'label_class': BADGE_DARK_CLASS, 'id': 'start_date'}))

    def is_valid(self):
        """
        Validate the fields and display error message if not valid
        """
        valid = super(ETFTrackingErrorInputForm, self).is_valid()

        return valid

