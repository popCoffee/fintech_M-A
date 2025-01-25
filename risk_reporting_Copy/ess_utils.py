from django.conf import settings
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import bbgclient

def get_queries_for_ess_cix_downsides(type=None):
    """ Retrieves the data for CIX based downsides """

    query = None
    if type == 'ess':
        query = f"SELECT DISTINCT alpha_ticker, cix_index, pt_down_cix FROM {settings.CURRENT_DATABASE}.risk_ess_idea " \
                f"WHERE " \
                f"is_archived = 0 AND pt_down_cix IS NOT NULL and tradegroup is not NULL"

    if type == 'eze':
        # Get the weights for tradegroups from flat file in Eze
        query = "SELECT Fund, TradeGroup, Ticker, AlphaHedge, RiskLimit, CurrentMktVal_Pct, DealDownside, SecType,Type, "\
                "LongShort, (amount*factor) as QTY, CurrentMktVal, FXCurrentLocalToBase as FxFactor, aum, "\
                "PutCall, Strike as StrikePrice, DealUpside FROM "\
                "wic.daily_flat_file_db WHERE Flat_file_as_of = "\
                "(SELECT MAX(Flat_file_as_of) FROM wic.daily_flat_file_db) "\
                "AND AlphaHedge IN ('Alpha', 'Alpha Hedge') AND amount<>0 AND sleeve LIKE 'EQUITY SPECIAL SITUATIONS' "\
                "AND Sleeve = 'Equity Special Situations'"


    return query


def get_ess_nav_impacts():
    engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" +
                           settings.WICFUNDS_DATABASE_PASSWORD + "@" + settings.WICFUNDS_DATABASE_HOST + "/" +
                           settings.WICFUNDS_DATABASE_NAME)
    con = engine.connect()
    sum_df = pd.DataFrame()
    details_df = pd.DataFrame()
    try:
        ess_drawdown_df = pd.read_sql_query(get_queries_for_ess_cix_downsides('eze'), con=con)
        ess_idea_df = pd.read_sql_query(get_queries_for_ess_cix_downsides('ess'), con=con)
        ess_idea_df.columns = ['Ticker', 'CIX Index', 'PT Down CIX']

        ess_drawdown_df[
            ['RiskLimit', 'CurrentMktVal_Pct', 'DealDownside', 'QTY', 'aum', 'StrikePrice', 'DealUpside']].astype(
            float)
        # if Risk Limit is 0 or NULL assume 30 basis point Risk limit
        ess_drawdown_df.loc[ess_drawdown_df.RiskLimit == 0, "RiskLimit"] = 0.30
        # Set downside to 0 for WTUS Type
        ess_drawdown_df.loc[ess_drawdown_df.Type == 'WTUS', "DealDownside"] = 0
        ess_drawdown_df['RiskLimit'] = ess_drawdown_df['RiskLimit'].apply(lambda x: -x if x > 0 else x)

        def calculate_break_pl(row):
            if row['AlphaHedge'] == 'Alpha':
                if row['SecType'] == 'EQ' and row['LongShort'] == 'Short':
                    return (row['DealUpside'] * row['QTY']) - (row['CurrentMktVal'] / row['FxFactor'])

                return (row['DealDownside'] * row['QTY']) - (row['CurrentMktVal'] / row['FxFactor'])
            # Todo Add logic for Options where LongShort = Short

            if row['PutCall'] == 'CALL':
                if row['StrikePrice'] <= row['DealDownside']:
                    x = (row['DealDownside'] - row['StrikePrice']) * (row['QTY'])
                else:
                    x = 0
            elif row['PutCall'] == 'PUT':
                if row['StrikePrice'] >= row['DealDownside']:
                    x = (row['StrikePrice'] - row['DealDownside']) * (row['QTY'])
                else:
                    x = 0
            return -row['CurrentMktVal'] + x
        ess_drawdown_df['Break PL'] = ess_drawdown_df.apply(calculate_break_pl, axis=1)
        ess_drawdown_df['NAV Risk'] = 1e2 * ess_drawdown_df['Break PL'] / ess_drawdown_df['aum']
        ess_drawdown_df_equity = ess_drawdown_df[ess_drawdown_df['SecType'] == 'EQ']

        ess_drawdown_df_equity = ess_drawdown_df_equity[
            ['Fund', 'TradeGroup', 'Ticker', 'Type', 'DealDownside', 'CurrentMktVal_Pct', 'NAV Risk']]
        ess_df_nav_risk = ess_drawdown_df_equity
        ess_df_nav_risk['Ticker'] = ess_df_nav_risk['Ticker'].apply(
            lambda x: x + " EQUITY" if 'equity' not in x.lower() else x)

        merged_df = pd.merge(ess_df_nav_risk, ess_idea_df, how='left', on='Ticker')
        cix_index = merged_df['CIX Index'].dropna().unique()
        alpha_tickers = merged_df['Ticker'].dropna().unique()
        ticker_list = list(cix_index) + list(alpha_tickers)
        live_price_df = pd.DataFrame.from_dict(
            bbgclient.bbgclient.get_secid2field(ticker_list, 'tickers', ['PX_LAST'], req_type='refdata'),
            orient='index').reset_index()
        live_price_df['PX_LAST'] = live_price_df['PX_LAST'].apply(lambda x: float(x[0]) if x[0] else None)
        live_price_df.columns = ['Ticker', 'Alpha PX Last']

        # Merge for the alpha price
        merged_df = pd.merge(merged_df, live_price_df, how='left', on='Ticker')
        live_price_df.columns = ['CIX Index', 'CIX PX Last']
        merged_df = pd.merge(merged_df, live_price_df, how='left', on='CIX Index')

        merged_df['CIX Move'] = merged_df['PT Down CIX'] - merged_df['CIX PX Last']
        merged_df['Alpha CIX Based Downside'] = merged_df['Alpha PX Last'] + merged_df['CIX Move']
        merged_df['Alpha Move'] = 1e2 * (merged_df['Alpha CIX Based Downside'] - merged_df['Alpha PX Last']) / \
                                  merged_df['Alpha PX Last']
        merged_df['CIX NAV Risk'] = merged_df['CurrentMktVal_Pct'] * merged_df['Alpha Move'] / 100  # in %
        details_df = merged_df.copy()

        merged_df = merged_df[['Fund', 'TradeGroup', 'NAV Risk', 'CIX NAV Risk']]
        merged_df.columns = ['fund', 'tradegroup', 'nav_risk', 'cix_nav_risk']
        merged_df = merged_df.drop_duplicates()
        details_df_2 = details_df[details_df['Type'] != 'WTUS']
        extra_columns_df = details_df_2[
            ['TradeGroup','Ticker',  'Alpha PX Last', 'DealDownside', 'Alpha CIX Based Downside']].drop_duplicates().copy()
        extra_columns_df.columns = ['tradegroup','alpha_ticker', 'alpha_last_price', 'fundamental_downside', 'cix_implied_downside']
        if 'WED' not in merged_df['fund'].unique():
            merged_df = merged_df.append({'fund': 'WED'}, ignore_index=True)
        sum_df = pd.pivot_table(merged_df, index=['tradegroup'], columns='fund', aggfunc=np.sum, fill_value='',
                                dropna=False)
        sum_df.columns = ["_".join((i, j)) for i, j in sum_df.columns]
        sum_df.replace(0, np.NaN, inplace=True)
        sum_df.reset_index(inplace=True)
        sum_df = pd.merge(sum_df, extra_columns_df, how='left', on='tradegroup')

    except Exception as e:
        print(e)
    finally:
        con.close()
        return sum_df, details_df

