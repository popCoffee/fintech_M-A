import numpy as np
import pandas as pd
from django.conf import settings


def get_rec_summaries(merged_df, for_day=None):
    """ Show summaries for Long, Short and Gross weights """
    forwards_df = merged_df[merged_df['sectype'] == 'FXFWD']
    merged_df = merged_df[merged_df['sectype'] != 'FXFWD']
    index_long = np.round(merged_df[merged_df['index'] > 0]['index'].sum(), decimals=2)
    index_short = np.round(merged_df[merged_df['index'] < 0]['index'].sum(), decimals=2)
    index_weights_forwards = np.round((forwards_df[forwards_df['index']<0]['index'].sum() - forwards_df[forwards_df['index']>0]['index'].sum()), decimals=2)
    index_gross = np.round(abs(index_long) + abs(index_short) + abs(index_weights_forwards), decimals=2)

    weight_tracked_long = np.round(merged_df[merged_df['weight_tracked'] > 0]['weight_tracked'].sum(), decimals=2)
    weight_tracked_short = np.round(merged_df[merged_df['weight_tracked'] < 0]['weight_tracked'].sum(), decimals=2)
    weight_tracked_forwards = np.round((forwards_df[forwards_df['index']<0]['weight_tracked'].sum() - forwards_df[forwards_df['index']>0]['weight_tracked'].sum()), decimals=2)
    weight_tracked_gross = np.round(abs(weight_tracked_long) + abs(weight_tracked_short) + abs(weight_tracked_forwards), decimals=2)

    pct_tracked_long = np.round(1e2 * (weight_tracked_long / index_long), decimals=2)
    pct_tracked_short = np.round(1e2 * (weight_tracked_short / index_short), decimals=2)
    pct_tracked_forwards = np.round(1e2 * (weight_tracked_forwards / index_weights_forwards), decimals=2)
    pct_tracked_gross = np.round(1e2 * (weight_tracked_gross / index_gross), decimals=2)

    additional_etf_exposure_long = np.round(merged_df[merged_df['additional_etf_exposure'] > 0]['additional_etf_exposure'].sum(), decimals=2)
    additional_etf_exposure_short = np.round(merged_df[merged_df['additional_etf_exposure'] < 0]['additional_etf_exposure'].sum(), decimals=2)
    additional_etf_exposure_forwards = 0
    additional_etf_exposure_gross = np.round(abs(additional_etf_exposure_long) + abs(additional_etf_exposure_short),
                                             decimals=2)

    date = for_day
    df_data = [[date, index_long, index_short, index_weights_forwards, index_gross, weight_tracked_long, weight_tracked_short,
                weight_tracked_forwards, weight_tracked_gross, pct_tracked_long, pct_tracked_short, pct_tracked_forwards,
                pct_tracked_gross, additional_etf_exposure_long, additional_etf_exposure_short,
                additional_etf_exposure_forwards, additional_etf_exposure_gross]]
    summary_df = pd.DataFrame(columns=['date', 'index_weights_long', 'index_weights_short', 'index_weights_forwards',
                                       'index_weights_gross', 'weight_tracked_long', 'weight_tracked_short',
                                       'weight_tracked_forwards', 'weight_tracked_gross', 'pct_tracked_long',
                                       'pct_tracked_short', 'pct_tracked_forwards', 'pct_tracked_gross',
                                       'additional_etf_exposure_long', 'additional_etf_exposure_short',
                                       'additional_etf_exposure_forwards', 'additional_etf_exposure_gross'],
                              data=df_data)
    return summary_df


def get_queries(for_type,  for_day=None):
    eze_cash_ticker = '%%@CASH%%'
    if for_type.lower() == 'eze':
        query = "SELECT Flat_file_as_of as `date`,sectype, sedol, Ticker as eze_ticker, " \
                "       TradeGroup as deal, PctOfSleeveCurrent as eze " \
                "FROM " \
                "wic.daily_flat_file_db " \
                "WHERE " \
                f"Flat_file_as_of = '{for_day}' AND fund = 'ARBETF' AND sedol NOT LIKE 'BDFDQP1' and ticker not like '{eze_cash_ticker}'" \
                f" UNION  " \
                " SELECT Flat_file_as_of AS `date`, sectype, REPLACE(SUBSTRING_INDEX(Ticker, ' ', 1), 'USD', '')  AS `sedol`, " \
                "Ticker, 'FORWARD' AS deal, 1e2*SUM(amount)*Price/aum AS eze  "\
                f" FROM wic.daily_flat_file_db WHERE Flat_file_as_of = '{for_day}' and fund like 'ARBETF' "\
                "  AND sectype LIKE 'FXFWD' "\
                "  GROUP BY ticker, price, aum;"

    elif for_type.lower() == 'state_street':
        query = "SELECT sedol, weight AS basket " \
                "FROM " + settings.ETF_DB_NAME+".statestreet_wicpcf  " \
                "WHERE " \
                f"date_updated = '{for_day}' AND fund = 'ARBETF'"

    elif for_type.lower() == 'solactive':
        query = "SELECT security_sedol AS sedol, 1e2*percentage_weighting AS _index " \
                "FROM " \
                "Wic_Index.opening_holdings_wimarb " \
                "WHERE " \
                f"Date = '{for_day}' AND security_sedol NOT LIKE 'BDFDQP1'"\
                "UNION " \
                "SELECT currency AS sedol, -1e2*closing_hedged_weight AS _index " \
                "FROM " \
                "Wic_Index.index_currency_info_opening " \
                "WHERE " \
                f"Date='{for_day}'"

    elif for_type.lower() == 'notes':
        query = "SELECT DISTINCT deal, eze_ticker, notes " \
                "FROM " \
                + settings.CURRENT_DATABASE+".etf_etfrecrecords " \
                "WHERE " \
                f"date = (SELECT MAX(date) from " + settings.CURRENT_DATABASE + ".etf_etfrecrecords)"
    elif for_type.lower() == 'index_ticker_lookup':
        query = "SELECT security_sedol AS sedol,security_ticker as ticker " \
                "FROM " \
                "Wic_Index.opening_holdings_wimarb " \
                "WHERE " \
                f"Date = '{for_day}' AND security_sedol NOT LIKE 'BDFDQP1'"
    else:
        query = 'N/A'

    return query
