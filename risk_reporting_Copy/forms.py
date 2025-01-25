import pandas as pd

from django import forms

from pnl_attribution.dfutils import convert_file_to_df


BADGE_SUCCESS_CLASS = 'badge badge-default badge-success'


class FormulaDownsideForm(forms.Form):
    """
    Form for Formula Downside Page
    """
    file = forms.FileField(required=True, label="BaseCase/Outlier File",
                           widget=forms.FileInput(attrs={'label_class': BADGE_SUCCESS_CLASS, 'id': 'downside_file',
                                                         'onChange': 'updateFileType()'}))

    def is_valid(self):
        """
        Validate the fields and display error message if not valid
        """
        valid = super(FormulaDownsideForm, self).is_valid()
        if not valid:
            return valid
        cleaned_data = self.cleaned_data
        uploaded_file = cleaned_data.get('file', "")
        file_df = pd.DataFrame()
        actual_file_columns = ['Underlying', 'TradeGroup', 'Outlier', 'Base Case']
        if uploaded_file:
            file_error_message = 'File does not have correct columns. Please make sure the columns are labeled ' \
                                 'correctly. The file should have the following columns only in any order: '
            file_df = convert_file_to_df(uploaded_file.file)
            if 'Unnamed: 0' in file_df.columns.values:
                del file_df['Unnamed: 0']
            file_columns = sorted(file_df.columns.values.tolist(), reverse=True)
            if file_columns != actual_file_columns:
                valid = False
                self._errors['file'] = file_error_message + ", ".join(col for col in actual_file_columns)
            file_df = file_df.dropna().drop_duplicates().reset_index(drop=True)
            if file_df.empty:
                valid = False
                self._errors['file'] = 'The uploaded file does not have Base case and Outlier values'
        if valid:
            file_df['TradeGroup'] = file_df['TradeGroup'].str.upper()
            file_df['Underlying'] = file_df['Underlying'].str.upper()
            file_df['Base Case'] = file_df['Base Case'].astype(float)
            file_df['Outlier'] = file_df['Outlier'].astype(float)
            self.cleaned_data['file_df'] = file_df
        return valid
