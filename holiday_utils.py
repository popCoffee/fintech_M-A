import datetime
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, USMartinLutherKingJr, USPresidentsDay,
                                    USMemorialDay, USLaborDay, USThanksgivingDay, GoodFriday, nearest_workday,
                                    weekend_to_monday)
from pandas.tseries.offsets import CustomBusinessDay

# First trade day off for Juneteenth is in 2022
Juneteenth = Holiday("Juneteenth", start_date=datetime.datetime(2021, 6, 20), month=6, day=19,
                     observance=nearest_workday)


class USNyseHolidayCalendar(AbstractHolidayCalendar):
    """
    Define a custom calendar for all the trading holidays
    NYSE Holiday Calendar based on dates specified by:
    https://www.nyse.com/markets/hours-calendars
    """

    rules = [
        Holiday("New Years Day", month=1, day=1, observance=weekend_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Juneteenth,
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


cal = USNyseHolidayCalendar()
holidays = cal.holidays(start='2016-01-01').to_pydatetime()


def is_market_closed(now=None):
    if now == None:
        now = get_todays_datetime()
    if isinstance(now, datetime.date): return is_market_holiday(now)
    if now.weekday() in [5, 6]: return True  # weekend
    if now in holidays: return True  # holiday
    if now > datetime.datetime(now.year, now.month, now.day, 16, 0): return True  # after hours
    if now < datetime.datetime(now.year, now.month, now.day, 9, 30): return True  # before hours
    return False


def is_market_holiday(now=None):
    if now == None:
        now = get_todays_datetime()
    # clean up date before comparing it with the holidays
    if isinstance(now, datetime.date):
        now = datetime.datetime.combine(now, datetime.datetime.min.time())
    else:
        now = datetime.datetime.combine(now.date(), datetime.datetime.min.time())
    if now.weekday() in [5, 6]: return True  # weekend
    if now in holidays: return True  # holiday
    return False


def get_last_trade_date(input_date=None):
    """
    Function for getting the nearest pervious trade date of the input date based on the NYSE calendar:
    https://www.nyse.com/markets/hours-calendars
    Parameters
    ----------
    input_date : Datetime object: target date to get the last trade date

    Returns Datetime.date: object of the closest pervious trading date to the input date
    -------

    """
    if input_date:
        today = input_date
    else:
        today = get_todays_date()
    calendar_rule = USNyseHolidayCalendar()
    US_BUSINESS_DAY = CustomBusinessDay(calendar=calendar_rule)
    last_trade_date = today - 1 * US_BUSINESS_DAY  # Get last trade day
    return last_trade_date

def get_todays_date() -> datetime.date:
    return get_todays_datetime().date()

def get_todays_datetime() -> datetime.datetime:
    d = datetime.datetime.now()
    return d
