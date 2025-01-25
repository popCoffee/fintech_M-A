from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

from dateutil.parser import parse

import holiday_utils


def append_equity_to_ticker(ticker):
    if ticker:
        ticker = ticker.upper()
        ticker = ticker + ' EQUITY' if 'equity' not in ticker.lower() else ticker.upper()
    return ticker if ticker else ''


def calculate_number_of_days(target_date):
    """
    Returns number of days from the given date till today
    """
    if target_date:
        if isinstance(target_date, (str)):
            target_date = parse(target_date, fuzzy=False)
        if isinstance(target_date, (datetime)):
            return (target_date.date() - holiday_utils.get_todays_date()).days
        elif isinstance(target_date, (date)):
            return (target_date - holiday_utils.get_todays_date()).days
    return 0


def convert_to_float_else_zero(value):
    if value:
        try:
            return float(value)
        except ValueError:
            return 0.00
        except Exception:
            return 0.00
    else:
        return 0.00


def convert_to_str_decimal(value, decimal=2):
    try:
        expression = '{:.' + str(decimal) + 'f}'
        if value:
            value = float(value)
            return expression.format(value)
    except ValueError:
        return expression.format(0)
    return expression.format(0)

def round_decimal_fields(df, digits=2):
    def round_decimal(val):
        if isinstance(val, Decimal):
            return val.quantize(Decimal('1.' + '0' * digits), rounding=ROUND_HALF_UP)
        return val

    return df.applymap(round_decimal)

def replace_boolean_fields(df, yes='Yes', no='No'):
    def convert(val):
        if isinstance(val, bool):
            return yes if val else no
        return val
    return df.applymap(convert)
