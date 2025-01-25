import datetime
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sqlalchemy import create_engine
from celery import shared_task
from celery_progress.backend import ProgressRecorder

import dbutils
import holiday_utils
from risk.models import MA_Deals
from slack_utils import get_channel_name
import bbgclient
from django.conf import settings
from django_slack import slack_message
from .models import CustomUserInputs, ETFLivePnL, ETFMonitors, MarketOnClose, DailyIndexPnL
from .etf_queries import get_pnl_tab_queries, get_pnl_queries
from .utils import get_queries, get_rec_summaries


DB_USER = settings.ETF_DB_USER
DB_PASSWORD = settings.ETF_DB_PASSWORD
INDEX_DB_HOST = settings.ETF_DB_HOST
DB_NAME = settings.ETF_DB_NAME
WIC_DB_HOST = settings.WICFUNDS_DATABASE_HOST
WIC_DB_NAME = settings.WICFUNDS_DATABASE_NAME

TRADAR_DB_NAME = settings.TRADAR_DB
TRADAR_USER = settings.TRADAR_USERNAME
TRADAR_PASSWORD = settings.TRADAR_PASSWORD
TRADAR_DB_HOST = settings.TRADAR_HOST

index_db_engine = create_engine("mysql://" + DB_USER + ":" + DB_PASSWORD + "@" + \
                                INDEX_DB_HOST + "/" + DB_NAME)

tradar_db_engine = create_engine("mssql+pymssql://" + TRADAR_USER + ":" + TRADAR_PASSWORD + "@" + \
                                 TRADAR_DB_HOST + "/" + TRADAR_DB_NAME)

wic_db_engine = create_engine("mysql://" + DB_USER + ":" + DB_PASSWORD + "@" + WIC_DB_HOST + "/" + WIC_DB_NAME)


@shared_task(bind=True)
def get_etf_tracking_error(self, start_date=None, end_date=None, start_date_tradar=None, end_date_tradar=None):
    """ Get tracking Error for ETF and Index"""
    # Database Connection Parameters
    progress_recorder = ProgressRecorder(self)

    data = dict()
    # Retrieve the data from Pandas sql query function
    tradar_connection = tradar_db_engine.connect()
    index_connection = index_db_engine.connect()

    try:
        tradar_df = pd.read_sql_query(get_pnl_tab_queries('tradar', start_date_tradar=start_date_tradar,
                                                          end_date_tradar=end_date_tradar), con=tradar_connection)
        opening_index = pd.read_sql_query(get_pnl_tab_queries('opening_holdings_wimarb', start_date=start_date,
                                                              end_date=end_date),
                                          con=index_connection)
        closing_index = pd.read_sql_query(get_pnl_tab_queries('closing_holdings_wimarb', start_date=start_date,
                                                              end_date=end_date),
                                          con=index_connection)
        opening_index_ccy = pd.read_sql_query(get_pnl_tab_queries('ccy_opening', start_date=start_date,
                                                                  end_date=end_date),
                                              con=index_connection)
        closing_index_ccy = pd.read_sql_query(get_pnl_tab_queries('ccy_closing', start_date=start_date,
                                                                  end_date=end_date),
                                              con=index_connection)
        # end_date_dcaf = datetime.datetime.strptime(end_date, '%Y%M%d').strftime('%Y-%M-%d')
        index_unhedged = pd.read_sql_query(get_pnl_tab_queries('index_profile_opening', start_date=start_date,
                                                              end_date=end_date),
                                          con=index_connection) # uses end_date only
        # dcaf = pd.read_sql_query(get_pnl_tab_queries('solactive_dcaf', start_date=start_date,
        #                                                       end_date=end_date),
        #                                   con=index_connection) # uses end_date only
        dcaf_div = pd.read_sql_query(get_pnl_tab_queries('dcaf_div', start_date=start_date,
                                                     end_date=end_date),
                                 con=index_connection)  # uses end_date only

        closing_index['NextBusinessDate'] = closing_index['Date'].apply(lambda x: (x + BDay(1)).date())
        # progress_recorder.set_progress(10, 100)
    except Exception as e:
        print(e)
        data['error'] = str(e)
    finally:
        print('Closing all Database connections....')
        tradar_connection.close()
        index_connection.close()
    try:
        try:
            index_unhedged_val= float(index_unhedged.values[0][0])
        except:
            index_unhedged_val = 0

        opening_index['percentage_weighting'] = opening_index['percentage_weighting'].apply(
            lambda x: np.round(float(x) * 100, decimals=2))
        opening_index['price'] = opening_index['closing_price'] * opening_index['closing_fx']
        closing_index['price'] = closing_index['closing_price'] * closing_index['closing_fx']
        closing_index['percentage_weighting'] = closing_index['percentage_weighting'].apply(
            lambda x: np.round(float(x) * 100, decimals=2))
        closing_index_ccy = process_ccy_files(opening_index_ccy, closing_index_ccy)
        closing_index = process_holdings_files_old(opening_index, closing_index, dcaf_div, index_unhedged_val)
        tracker_df = pd.concat([closing_index[['Date', 'security_sedol', 'security_ticker', 'currency', 'pnl','Dividend Pnl']].groupby(
            ['Date', 'security_sedol', 'security_ticker', 'currency']).sum().reset_index(),
                                closing_index_ccy[['Date', 'currency', 'fwd_pnl']].groupby(
                                    ['Date', 'currency']).sum().reset_index().rename(columns={'fwd_pnl': 'pnl'})],
                               sort=False)
        tracker_df['security_ticker'] = tracker_df.apply(
            lambda x: x['currency'] if pd.isna(x['security_ticker']) else x['security_ticker'], axis=1)
        tracker_df['security_sedol'] = tracker_df.apply(
            lambda x: x['currency'] if pd.isna(x['security_sedol']) else x['security_sedol'], axis=1)

        progress_recorder.set_progress(25, 100)

        tradar_df['sedol'] = tradar_df.apply(lambda x: x['ticker'] if pd.isna(x['sedol']) else x['sedol'], axis=1)
        trdr_df = tradar_df[['Date', 'tradegroup', 'sedol', 'SecType', 'Pct_pnl']].groupby(['Date', 'tradegroup',
                                                                                            'sedol', 'SecType']). \
            sum().reset_index().rename(columns={'Pct_pnl': 'tradar_pnl'})
        # Convert Date to string for merging
        tracker_df['Date'] = tracker_df['Date'].astype(str)
        trdr_df['Date'] = trdr_df['Date'].astype(str)

        # Take out currencies frm Tradar
        currencies_df = trdr_df[trdr_df['SecType'] == 'Currencies']
        # Rename Currencies TradeGroups to SEDOLs
        currencies_df['tradegroup'] = currencies_df.apply(lambda x: x['sedol'], axis=1)
        # Remove Currencies from Tradar
        trdr_df = trdr_df[trdr_df['SecType'] != 'Currencies']
        # Group all by date
        del currencies_df['SecType']
        del trdr_df['SecType']
        currencies_df = currencies_df.groupby(['Date', 'tradegroup', 'sedol']).sum().reset_index()

        progress_recorder.set_progress(50, 100)

        trdr_df = pd.concat([trdr_df, currencies_df])
        tracker_df = pd.merge(tracker_df, trdr_df, how='outer', left_on=['Date', 'security_sedol'],
                              right_on=['Date', 'sedol'])
        tracker_df['pnl'] = tracker_df['pnl'].fillna(0)
        tracker_df['tradar_pnl'] = tracker_df['tradar_pnl'].fillna(0)

        tracker_df['tracking_error'] = tracker_df['tradar_pnl'] - tracker_df['pnl']
        tracker_df['tracking_error'] = tracker_df['tracking_error'].round(2)
        tracker_df['Date'] = tracker_df['Date'].astype(str)
        summed_up_df = tracker_df[['Date', 'pnl', 'tradar_pnl', 'tracking_error']].groupby('Date').sum().reset_index()
        summed_up_df['Cumulative_pnl_index'] = summed_up_df['pnl'].cumsum()
        summed_up_df['Cumulative_pnl_tradar'] = summed_up_df['tradar_pnl'].cumsum()
        summed_up_df.drop(columns=['pnl', 'tradar_pnl', 'tracking_error'], inplace=True)
        data['daily_tracking_error'] = summed_up_df.to_json(orient='records')

        progress_recorder.set_progress(65, 100)

        rename_cols = {'currency': 'CCY', 'pnl': 'P&L(Index) %', 'security_sedol': 'Sec SEDOL',
                       'security_ticker': 'Ticker', 'sedol': 'SEDOL',
                       'tradar_pnl': 'P&L(tradar) %', 'tradegroup': 'TradeGroup',
                       'tracking_error': 'TE %'}
        tracker_df.rename(columns=rename_cols, inplace=True)
        top_level_cols = ['TradeGroup', 'P&L(Index) %', 'P&L(tradar) %', 'TE %']
        top_level_df = tracker_df[top_level_cols].fillna('N/A').groupby('TradeGroup').sum().reset_index()
        # Create a Drilldown dialog with Json results
        cols2groupby = ['TradeGroup', 'Ticker', 'Sec SEDOL', 'SEDOL', 'CCY']

        # Fill Tracker DF Null TradeGroups with N/A
        tracker_df.fillna({'TradeGroup': 'N/A', 'Sec SEDOL': 'N/A', 'SEDOL': 'N/A'}, inplace=True)
        progress_recorder.set_progress(90, 100)

        def tradegroup_drilldown_dialog(y):
            tg_drilldown_df = tracker_df[tracker_df['TradeGroup'] == y].groupby(cols2groupby).sum().reset_index()
            # creates additi0nal col to display target and acquirer fields
            return {} if tg_drilldown_df.empty else tg_drilldown_df.fillna('N/A').to_json(orient='records')

        top_level_df['Drilldown'] = top_level_df['TradeGroup'].apply(lambda x: tradegroup_drilldown_dialog(x))
        data['detailed_tracking_error'] = top_level_df.to_json(orient='records')
    except Exception as e:
        print(e)
        data['error'] = str(e)
        # todo post to slack or error return

    return data


def process_ccy_files(opening_ccy, closing_ccy):
    """ Adds the FWD P&L to Closing index by referencing values from opening index and adds the fwd P&L
        Adds the following columns: Start Weight, Start Spot, End Spot, Spot Change, Start Interpolated Fwd,
        End Interpolated Fwd, Fwd change, Fwd P&L
    """

    def add_cols(col_name, row):
        """ Reads the col_name from ticker and returns the appropriate value """
        values_df = opening_ccy[((opening_ccy['currency'] == row['currency']) & (opening_ccy['Date'] == row['Date']))]
        if not values_df.empty:
            return values_df.iloc[0][col_name]
        else:
            return np.NaN

    closing_ccy['start_weight'] = closing_ccy.apply(lambda x: add_cols('closing_hedged_weight', x), axis=1)
    closing_ccy['start_spot'] = closing_ccy.apply(lambda x: add_cols('closing_spot', x), axis=1)
    closing_ccy['start_interpolated_fwd'] = closing_ccy.apply(lambda x: add_cols('interpolated_forward', x), axis=1)

    closing_ccy['spot_change'] = -(closing_ccy['closing_spot'] - closing_ccy['start_spot']) / closing_ccy[
        'start_spot']
    closing_ccy['fwd_change'] = -(closing_ccy['interpolated_forward'] - closing_ccy['start_interpolated_fwd']) / \
                                closing_ccy['start_interpolated_fwd']

    # Calculate the P&L. apply negative for appropriate pnl values
    closing_ccy['fwd_pnl'] = 1e2 * closing_ccy['fwd_change'].astype(float) * closing_ccy['start_weight'].astype(
        float)

    return closing_ccy


def process_holdings_files_old(opening_, closing_, dcaf, index_unhedged):
    """ archived: Adds the P&L for holdings (Index) """

    def add_cols(col_name, row):
        """ Reads the col_name from ticker and returns the appropriate value """
        values_df = opening_[
            ((opening_['security_sedol'] == row['security_sedol']) & (opening_['Date'] == row['Date']))]
        if not values_df.empty:
            return values_df.iloc[0][col_name]
        else:
            return np.NaN

    # A Dataframe to track if dividend payment is made
    dividends_df = opening_[['Date', 'security_sedol', 'closing_price', 'percentage_weighting', 'closing_fx']]
    dividends_df.rename(columns={'closing_price': 'Dvds_Price', 'percentage_weighting': 'dvds_pct_weight',
                                 'closing_fx': 'dvds_closing_fx'}, inplace=True)
    # Adjust Price by Fx
    dividends_df['Dvds_Price'] = dividends_df['Dvds_Price'] * dividends_df['dvds_closing_fx']
    closing_['sod_weight'] = closing_.apply(lambda x: add_cols('percentage_weighting', x), axis=1)
    closing_['start_price'] = closing_.apply(lambda x: add_cols('price', x), axis=1)
    closing_['price_change'] = (closing_['price'] - closing_['start_price']) / closing_['start_price']

    # Merge with dividends based on NextBusinessDate
    closing_ = pd.merge(closing_, dividends_df, left_on=['security_sedol', 'NextBusinessDate'],
                        right_on=['security_sedol', 'Date'], how='left')  # Adds the Dvds_Price column

    # If there is a difference between Dvds_Price and closing_price, then a dividend has been paid
    closing_['Dividend Rate'] = closing_['Dvds_Price'] - closing_['price']
    closing_['Dividend Pnl'] = (closing_['Dividend Rate'] / closing_['price']) * closing_['dvds_pct_weight'] * \
                               closing_['dvds_closing_fx']
    closing_['Dividend Pnl'] = (dcaf['dividend_amount'].fillna(0)*dcaf['post_event_fraction_shares'].fillna(0)) / index_unhedged

    closing_['price_pnl'] = closing_['price_change'] * closing_['sod_weight']
    closing_['pnl'] = closing_['price_pnl'] + closing_['Dividend Pnl']
    closing_.drop(columns=['Dividend Rate', 'NextBusinessDate', 'Dvds_Price', 'dvds_pct_weight',
                           'dvds_closing_fx', 'Date_y'], inplace=True) # 'Dividend Pnl'
    closing_.rename(columns={'security_sedol_x': 'security_sedol', 'Date_x': 'Date'}, inplace=True)

    return closing_

def process_holdings_files(opening_, closing_, dcaf_div, index_unhedged, currency_list):
    """ Adds the P&L for holdings (Index) & include dividends pnl """

    def add_cols(col_name, row):
        """ Reads the col_name from ticker and returns the appropriate value. """
        values_df = opening_[
            ((opening_['security_sedol'] == row['security_sedol']) & (opening_['Date'] == row['Date']))]
        if not values_df.empty:
            return values_df.iloc[0][col_name]
        else:
            return np.NaN

    # A Dataframe to track if dividend payment is made
    dividends_df = opening_[['Date', 'security_sedol', 'percentage_weighting','closing_price' , 'closing_fx']] # 'closing_price'  # removed due to duplicate when merging?
    dividends_df.rename(columns={'closing_price': 'Dvds_Price', 'percentage_weighting': 'dvds_pct_weight',
                                 'closing_fx': 'dvds_closing_fx'}, inplace=True)
    # Adjust Price by Fx
    dividends_df['Dvds_Price'] = dividends_df['Dvds_Price'] * dividends_df['dvds_closing_fx']
    closing_['sod_weight'] = closing_.apply(lambda x: add_cols('percentage_weighting', x), axis=1)
    closing_['start_price'] = closing_.apply(lambda x: add_cols('price', x), axis=1)

    # merge on sedol/Ex_date to incorporate dividend_amount data for dividend pnl. causing duplicates
    closing_ = pd.merge(closing_, dcaf_div[['security_sedol', 'event_effective_date','current_fraction_shares','dividend_amount','adjusted_open_price']],
                        left_on=['security_sedol', 'Date'],right_on=['security_sedol', 'event_effective_date'], how='left')

    # Merge opening & closing via dividends based on NextBusinessDate
    closing_ = pd.merge(closing_, dividends_df, left_on=['security_sedol', 'NextBusinessDate'],
                        right_on=['security_sedol', 'Date'], how='left')  # Adds the Dvds_Price column
    # calc adjusted_open_price
    closing_['adjusted_open_price'] = closing_.apply(
        lambda row: row['opening_price'] if pd.isnull(row['adjusted_open_price']) else row['adjusted_open_price'] + row[
            'dividend_amount'] if row['dividend_amount'] is not None else row['adjusted_open_price'], axis=1)
    # calc price change w/ adjusted_open_price
    closing_['price_change'] = (closing_['price'] - closing_['adjusted_open_price']) / closing_['start_price']
    ### start_price = opening price
    ### adjusted_open_price = opening price w./ divs
    # If there is a difference between Dvds_Price and closing_price, then a dividend has been paid
    closing_['Dividend Rate'] = closing_['Dvds_Price'] - closing_['price']
    # closing_['Dividend Pnl'] = (closing_['Dividend Rate'] / closing_['price']) * closing_['dvds_pct_weight'] * \
    #                            closing_['dvds_closing_fx']
    closing_['Dividend Pnl'] = round( (closing_['dividend_amount'].fillna(0)*closing_['current_fraction_shares'].fillna(0)) / index_unhedged ,9)
    # overwrite with previous historical value if div_pnl calc is not zero.
    # closing_['price_change_bbg'] = closing_.apply(lambda x: (x['price']-x['prev_day_price'])/x['prev_day_price'] if abs(x['Dividend Pnl'])>0 else x['price_change'], axis=1)

    # (merged_df['price_change_pct'] * merged_df['percentage_weighting']) + (
    #             merged_df['fx_change_pct'] * merged_df['percentage_weighting'])
    closing_['price_pnl'] = closing_['price_change'] * closing_['sod_weight'] + (closing_['fx_change_pct'] * closing_['sod_weight'])
    closing_['pnl'] = closing_['price_pnl'] + closing_['Dividend Pnl']*100
    # new = closing_[['pnl','price_change','sod_weight','security_sedol']]
    closing_.drop(columns=['Dividend Rate', 'NextBusinessDate', 'Dvds_Price', 'dvds_pct_weight',
                           'dvds_closing_fx', 'Date_y'], inplace=True) # 'Dividend Pnl'
    closing_.rename(columns={'security_sedol_x': 'security_sedol', 'Date_x': 'Date'}, inplace=True)

    return closing_


def get_sectype_with_tickers(df):
    api_host = bbgclient.bbgclient.get_next_available_host()
    all_unique_tickers = list(df['security_ticker'].apply(lambda x: x.upper() ).unique())
    live_sectype_df = pd.DataFrame.from_dict(
        bbgclient.bbgclient.get_secid2field(all_unique_tickers, 'tickers', ['SECURITY_TYP'], req_type='refdata',
                                            api_host=api_host), orient='index').reset_index()
    live_sectype_df['SECURITY_TYP'] = live_sectype_df['SECURITY_TYP'].apply(lambda x: x[0])
    live_sectype_df['index'] = live_sectype_df['index'].str.replace(' EQUITY', ' Equity')
    return live_sectype_df

def get_prev_price(df):
    '''get prev day price and merge as column prev_day_price'''
    api_host = bbgclient.bbgclient.get_next_available_host()
    all_unique_tickers = list(df['security_ticker'].apply(lambda x: x.upper() if(np.all(pd.notnull(x))) else x   ).unique() )
    # if abs(sum(df.dividend_pnl.to_list())) > 0:
    today = max(df['Date'])
    prev_date = get_previous_business_day(today.strftime("%Y-%m-%d"))
    live_prev_price_df = pd.DataFrame.from_dict(bbgclient.bbgclient.get_secid2field(all_unique_tickers, "tickers",
                                                             ['PX_LAST'],
                                                             start_date=prev_date,
                                                             end_date=prev_date)).transpose().reset_index()
    # convert nones to [] to float
    # live_prev_price_df['PX_LAST'] = live_prev_price_df['PX_LAST'].fillna(value='')
    live_prev_price_df['PX_LAST'] = live_prev_price_df['PX_LAST'].apply(lambda x: float(x[0]) if (x and x[0]) else None)
    live_prev_price_df['index'] = live_prev_price_df['index'].str.replace(' EQUITY', ' Equity')
    df = pd.merge(df, live_prev_price_df, how='left', left_on='security_ticker', right_on='index')
    del df['index']
    df.rename(columns={ 'PX_LAST':'prev_day_price' },inplace=True)
    return df

def get_previous_business_day(today):
    # Convert the input date string to a datetime object
    today_date = datetime.datetime.strptime(today, "%Y-%m-%d")
    previous_business_day = None
    # (Monday to Friday)
    weekdays = [0, 1, 2, 3, 4]
    # Iterate backwards find the previous business day
    for i in range(1, 7):
        previous_date = today_date - datetime.timedelta(days=i)
        if previous_date.weekday() in weekdays:
            previous_business_day = previous_date
            break
    # Format string in "YYYY-MM-DD" format
    return previous_business_day.strftime('%Y%m%d')

def get_tradegroup_etf(df):
    all_unique_sedols = list(df['security_sedol'].apply(lambda x: x.upper()).unique())
    all_unique_tick = list(df['security_ticker'].apply(lambda x: x.upper()).unique())
    sedol_df = pd.DataFrame.from_records( list(  ETFLivePnL.objects.filter(sedol__in=all_unique_sedols).values('sedol','tradegroup').distinct() ))
    ticker_df = pd.DataFrame.from_records( list(  MA_Deals.objects.filter(target_ticker__in=all_unique_tick).values('target_ticker','deal_name').distinct() ))
    ticker_df['target_ticker'] = ticker_df['target_ticker'].str.replace(' EQUITY', ' Equity')
    return sedol_df, ticker_df

def get_previous_td(ss):
    ## frontfill
    previous_dip =  DailyIndexPnL.objects.filter(ticker=ss).order_by('-date')[2:3]
    if len(previous_dip):
        previous_dip = previous_dip.values('sedol','tradegroup')
    else:
        return None
    val = previous_dip[0] if previous_dip else {}
    return val.get('tradegroup',None)


@shared_task()
def upload_daily_etf_index_pnl(start_date=None, start_date_tradar=None):
    """ Store index p&l daily"""
    # skipping today's execution if it's a holiday
    if start_date != None:
        if holiday_utils.is_market_holiday(datetime.datetime.strptime(start_date, '%Y%m%d')):
            return
    if start_date == None and start_date_tradar == None:
        today = datetime.datetime.today().date()
        start_date = today.strftime('%Y%m%d')
        start_date_tradar = today.strftime('%Y%m%d')
    # Database Connection Parameters
    closing_columns = ['Date', 'security_sedol', 'security_ticker', 'currency', 'pnl','Dividend Pnl','closing_price','closing_fx','percentage_weighting','fraction_of_shares','opening_price'] #prev_day_price
    data = dict()
    # Retrieve the data from Pandas sql query function
    tradar_connection = tradar_db_engine.connect()
    index_connection = index_db_engine.connect()

    try:
        tradar_df = pd.read_sql_query(get_pnl_tab_queries('tradar', start_date_tradar=start_date_tradar,
                                                          end_date_tradar=start_date_tradar), con=tradar_connection)
        opening_index = pd.read_sql_query(get_pnl_tab_queries('opening_holdings_wimarb', start_date=start_date,
                                                              end_date=start_date),
                                          con=index_connection)
        closing_index = pd.read_sql_query(get_pnl_tab_queries('closing_holdings_wimarb', start_date=start_date,
                                                              end_date=start_date),
                                          con=index_connection)
        opening_index_ccy = pd.read_sql_query(get_pnl_tab_queries('ccy_opening', start_date=start_date,
                                                                  end_date=start_date),
                                              con=index_connection)
        closing_index_ccy = pd.read_sql_query(get_pnl_tab_queries('ccy_closing', start_date=start_date,
                                                                  end_date=start_date),
                                              con=index_connection)
        # end_date_dcaf = datetime.datetime.strptime(end_date, '%Y%M%d').strftime('%Y-%M-%d')
        index_unhedged = pd.read_sql_query(get_pnl_tab_queries('index_profile_opening', start_date=start_date,
                                                              end_date=start_date),
                                          con=index_connection) # uses end_date only
        # dcaf = pd.read_sql_query(get_pnl_tab_queries('solactive_dcaf', start_date=start_date,
        #                                                       end_date=start_date),
        #                                   con=index_connection) # uses end_date only
        dcaf_div = pd.read_sql_query(get_pnl_tab_queries('dcaf_div', start_date=start_date,
                                                         end_date=start_date),
                                     con=index_connection)  # uses end_date only

        closing_index['NextBusinessDate'] = closing_index['Date'].apply(lambda x: (x + BDay(1)).date())
    except Exception as e:
        print(e)
        return str(e)
    finally:
        print('Closing all Database connections....')
        tradar_connection.close()
        index_connection.close()
    try:
        index_unhedged_val= float(index_unhedged.values[0][0])
    except:
        index_unhedged_val = 0
    opening_index['percentage_weighting'] = opening_index['percentage_weighting'].apply(
        lambda x: np.round(float(x) * 100, decimals=3))

    closing_index['fx_change_pct'] = (closing_index['closing_fx'] - opening_index['closing_fx']) / opening_index['closing_fx']

    opening_index['price'] = opening_index['closing_price'] #* opening_index['closing_fx']
    ## temporary value for debugging
    closing_index['opening_price'] = closing_index['isin'].map(opening_index.set_index('isin')['price'])
    closing_index['price'] = closing_index['closing_price'] #* closing_index['closing_fx']
    closing_index['percentage_weighting'] = closing_index['percentage_weighting'].apply(
        lambda x: np.round(float(x) * 100, decimals=3))
    currency_list = closing_index_ccy['currency'].unique()

    # remove duplicate dcaf entries on different dated rows and preserve the most recent dated entries
    if len(dcaf_div)>0:
        dcaf_div['diff_days'] = dcaf_div.apply(lambda x: abs(x['dcaf_date'] - datetime.datetime.strptime(start_date, '%Y%m%d').date() ),axis=1 )
        dcaf_div = dcaf_div.sort_values('diff_days', ascending=False).drop_duplicates(['event_effective_date','security_name'],keep='last').reset_index(drop=True)

    closing_index_ccy = process_ccy_files(opening_index_ccy, closing_index_ccy)
    # closing_index = get_prev_price(closing_index)

    closing_index = process_holdings_files(opening_index, closing_index, dcaf_div, index_unhedged_val, currency_list)
    # combine closing df with currency df and rename fwd_pnl to pnl to match pnl col of closing
    tracker_df = pd.concat([closing_index[closing_columns].groupby(
        ['Date', 'security_sedol', 'security_ticker', 'currency']).sum().reset_index(),
                            closing_index_ccy[['Date', 'currency', 'fwd_pnl']].groupby(
                                ['Date', 'currency']).sum().reset_index().rename(columns={'fwd_pnl': 'pnl'})],
                           sort=False)
    # add closing_hedged_weight column to the main df.
    closing_index_ccy.closing_hedged_weight = closing_index_ccy.closing_hedged_weight.astype(float)
    # add currency as a ticker to the main df. col ticker = EUR , GBP
    tracker_df = tracker_df.join(closing_index_ccy[['currency','closing_hedged_weight']].set_index( 'currency'), on=['currency'])
    #

    tracker_df['security_ticker'] = tracker_df.apply(
        lambda x: x['currency'] if pd.isna(x['security_ticker']) else x['security_ticker'], axis=1)
    tracker_df['security_sedol'] = tracker_df.apply(
        lambda x: x['currency'] if pd.isna(x['security_sedol']) else x['security_sedol'], axis=1)

    tradar_df['sedol'] = tradar_df.apply(lambda x: x['ticker'] if pd.isna(x['sedol']) else x['sedol'], axis=1)
    trdr_df = tradar_df[['Date', 'tradegroup', 'sedol', 'SecType', 'Pct_pnl']].groupby(['Date', 'tradegroup',
                                                                                        'sedol', 'SecType']). \
        sum().reset_index().rename(columns={'Pct_pnl': 'tradar_pnl'})
    # Convert Date to string for merging
    tracker_df['Date'] = tracker_df['Date'].astype(str)
    # trdr_df['Date'] = trdr_df['Date'].astype(str)

    # pnl apply -1 for only currencies
    tracker_df['pnl'] = tracker_df.apply(lambda row: -row['pnl'] if row['security_ticker'] in list(currency_list) else row['pnl'], axis=1)

    # get and merge sectype data
    sectype_df = get_sectype_with_tickers(tracker_df)
    tracker_df = pd.merge(tracker_df, sectype_df, how='left', left_on='security_ticker',right_on='index')
    del tracker_df['index']

    sedol_df, tickers_df = get_tradegroup_etf(tracker_df)
    tracker_df = pd.merge(tracker_df, sedol_df, how='left', left_on='security_sedol', right_on='sedol')
    ## causes duplicates based on ticker symbol
    # tracker_df = pd.merge(tracker_df, tickers_df, how='left', left_on='security_ticker', right_on='target_ticker')
    # tracker_df['tradegroup'].fillna(tracker_df['deal_name'],inplace=True)
    # del tracker_df['deal_name']
    # del tracker_df['target_ticker']
    del tracker_df['sedol']


    ## tradar processes.
    # Take out currencies frm Tradar
    currencies_df = trdr_df[trdr_df['SecType'] == 'Currencies']
    # Rename Currencies TradeGroups to SEDOLs
    currencies_df['tradegroup'] = currencies_df.apply(lambda x: x['sedol'], axis=1)
    # Remove Currencies from Tradar
    trdr_df = trdr_df[trdr_df['SecType'] != 'Currencies']
    # Group all by date
    del currencies_df['SecType']
    del trdr_df['SecType']
    # currencies_df = currencies_df.groupby(['Date', 'tradegroup', 'sedol']).sum().reset_index()


    ## remove tradar merge due to complexity of tracking-errors and to show only index performance.
    # trdr_df = pd.concat([trdr_df, currencies_df])
    # tracker_df = pd.merge(tracker_df, trdr_df, how='outer', left_on=['Date', 'security_sedol'],
    #                       right_on=

    ## to fill in remaining missing tradegroups
    # trdr_df.rename(columns={'tradegroup':'tradegroupTR'} , inplace=True)
    # tracker_df = pd.merge(tracker_df, trdr_df[['tradegroupTR', 'sedol']], how='left', left_on='security_sedol', right_on='sedol')
    # tracker_df['tradegroup'].fillna(tracker_df['tradegroupTR'], inplace=True)
    # del tracker_df['tradegroupTR']
    # del tracker_df['sedol']
    # del tracker_df['prev_day_price']
    tracker_df.drop_duplicates(['tradegroup', 'security_sedol' ], inplace=True)


    ## fill in final missing tradegroups
    tracker_df['tradegroup'] = tracker_df.apply(
        lambda x: get_previous_td(x['security_ticker']) if pd.isna(x['tradegroup']) else x['tradegroup'], axis=1)


    tracker_df['pnl'] = tracker_df['pnl'].fillna(0)
    # tracker_df['tradar_pnl'] = tracker_df['tradar_pnl'].fillna(0)
    # saving data
    tracker_df.rename(columns={'Date':'date','Dividend Pnl':'dividend_pnl','security_sedol':'sedol','security_ticker':'ticker','currency':'cur','SECURITY_TYP': 'sectype'}, inplace=True)
    DailyIndexPnL.objects.filter(date= datetime.datetime.strptime(start_date, '%Y%m%d').date() ).delete()
    tracker_df.to_sql(name='etf_dailyindexpnl', con=settings.SQLALCHEMY_CONNECTION, if_exists='append',
              index=False, schema=settings.CURRENT_DATABASE)
    return
    # error calc
    # tracker_df['tracking_error'] = tracker_df['tradar_pnl'] - tracker_df['pnl']
    # tracker_df['tracking_error'] = tracker_df['tracking_error'].round(2)
    # tracker_df['date'] = tracker_df['date'].astype(str)
    # top_level_cols = ['tradegroup', 'pnl', 'tradar_pnl', 'tracking_error']
    # tracker_df = tracker_df[top_level_cols].fillna('N/A').groupby('tradegroup').sum().reset_index()
    # except:
    #     print('didnt store dailyIndexP&L perf')
    #     pass
    ### several issues: tradar merge causes some sedol and tradegroup missing values. tradar merge causes additional securites like swaps per tradegroup.



@shared_task
def get_etf_recs():

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    try:
        index_db_engine = create_engine("mysql://" + settings.ETF_DB_USER + ":" + settings.ETF_DB_PASSWORD
                                        + "@" + settings.ETF_DB_HOST + "/" + settings.ETF_DB_NAME)

        wic_db_engine = create_engine("mysql://" + settings.WICFUNDS_DATABASE_USER + ":" +
                                      settings.WICFUNDS_DATABASE_PASSWORD + "@" + settings.WICFUNDS_DATABASE_HOST +
                                      "/" + settings.WICFUNDS_DATABASE_NAME)

        for_day = datetime.datetime.now().date()
        index_day = for_day + BDay(1)
        for_day = for_day.strftime('%Y-%m-%d')
        index_day = index_day.strftime('%Y-%m-%d')
        wic_connection = wic_db_engine.connect()
        index_connection = index_db_engine.connect()

        # Retrieve the data from Pandas sql query function
        eze_df = pd.read_sql_query(get_queries('eze', for_day=for_day), con=wic_connection)
        state_street_df = pd.read_sql_query(get_queries('state_street', for_day=for_day), con=index_connection)
        # For the Index, reference the opening values
        solactive_df = pd.read_sql_query(get_queries('solactive', for_day=index_day), con=index_connection)
        index_ticker_lookup_df = pd.read_sql_query(get_queries('index_ticker_lookup', for_day=index_day), con=index_connection)

        # Rename _index to Index (SQL query doesn't support the keyword index hence used _index)
        solactive_df.rename(columns={'_index': 'index'}, inplace=True)
        # Clean the Statestreet SEDOLs

        # Exclue cash after 2021-09-16 fund conversion
        state_street_df = state_street_df[~state_street_df['sedol'].isna()]
        state_street_df['sedol'] = state_street_df['sedol'].apply(lambda x: x.replace("'", ""))
        merged_df = pd.merge(solactive_df, state_street_df, how='outer', on=['sedol'])
        merged_df = pd.merge(merged_df, eze_df, how='outer', on=['sedol'])
        merged_df = merged_df[['date', 'sectype', 'sedol', 'eze_ticker', 'deal', 'index', 'eze', 'basket']]
        merged_df['date'] = merged_df['date'].apply(pd.to_datetime)
        df_date = merged_df['date'].max()
        merged_df['date'] = merged_df['date'].fillna(df_date)

        def ticker_lookup_from_index(row):
            _df = index_ticker_lookup_df[index_ticker_lookup_df['sedol'] == row['sedol']]
            if not _df.empty:
                ticker = _df.iloc[0]['ticker']
            else:
                ticker = '?'

            return ticker

        merged_df['eze_ticker'] = merged_df.apply(lambda x: ticker_lookup_from_index(x) if pd.isna(x['eze_ticker']) else x['eze_ticker'], axis=1)
        merged_df[['sectype', 'eze_ticker', 'deal']] = merged_df[['sectype', 'eze_ticker', 'deal']].fillna('Unknown')
        merged_df[['index', 'eze', 'basket']] = merged_df[['index', 'eze', 'basket']].fillna(0).round(2)
        # Get the Weight differences
        merged_df['index_eze'] = merged_df['eze'] - merged_df['index']
        merged_df['basket_eze'] = merged_df['basket'] - merged_df['eze']
        merged_df['index_basket'] = merged_df['index'] - merged_df['basket']

        merged_df[['index_eze', 'basket_eze', 'index_basket']] = merged_df[['index_eze', 'basket_eze',
                                                                            'index_basket']].round(2)

        def get_weight_tracked(row):
            if row['index'] == 0:
                return_val = 0
            elif row['index'] > 0:
                return_val = row['index'] if row['eze'] > row['index'] else row['eze']
            else:
                return_val = row['index'] if row['eze'] < row['index'] else row['eze']
            return return_val

        merged_df['weight_tracked'] = merged_df.apply(lambda x: get_weight_tracked(x), axis=1)

        merged_df['pct_tracked'] = merged_df.apply(
            lambda x: 100 if np.divide(x['eze'] , x['index']) > 1 else np.round(1e2 * np.divide(x['eze'], x['index']), 2), axis=1)

        # Set Pct Tracked to 0 if index weight is 0 and eze has additional weight
        merged_df['pct_tracked'] = merged_df.apply(
            lambda x: 0 if (x['index'] == 0 and x['eze'] != 0) else x['pct_tracked'], axis=1)

        # Add additional etf exposure to the list

        def get_additional_etf_exposure(row):
            return_val = 0
            if row['index'] == 0:
                return_val = row['eze']
            elif row['index'] < 0:
                if row['eze'] < row['index']:
                    return_val = row['eze'] - row['index']
            elif row['index'] > 0:
                if row['eze'] > row['index']:
                    return_val = row['eze'] - row['index']
            return return_val

        merged_df['additional_etf_exposure'] = merged_df.apply(lambda x: get_additional_etf_exposure(x), axis=1)

        # Call a function to get the previous days notes and merge with merged_df to
        # preserve notes from the previous day.
        # Match on deal names
        aggregate_dict_comprehension = {k: 'first' if k in ['eze_ticker', 'index', 'basket', 'index_basket'] else sum for k in
                                        merged_df.columns if k not in ['date', 'deal', 'sedol']}
        # Group by SEDOL and Date for a consolidated view of equity and swap (no longer used)
        merged_df = merged_df.groupby(['date', 'sedol', 'deal']).agg(aggregate_dict_comprehension).reset_index()

        # connect from wic db instead of index db
        notes_df = pd.read_sql_query(get_queries('notes'), con=wic_connection).drop_duplicates().fillna("")

        if notes_df.empty:
            merged_df['notes'] = ''
        else:
            merged_df = pd.merge(merged_df, notes_df, how='left', on=['eze_ticker', 'deal']).drop_duplicates()

        summary_df = get_rec_summaries(merged_df, for_day=index_day)

        # Insert Merged Df and Summary df into the database

        # Delete everything from models for the same date to avoid duplicates
        rs = wic_connection.execute("DELETE FROM " + settings.CURRENT_DATABASE + ".etf_etfrecrecords "
                                                                                 "WHERE date = '" + index_day + "'")

        rs = wic_connection.execute("DELETE FROM " + settings.CURRENT_DATABASE + ".etf_etfrecsummary "
                                                                                 "WHERE date = '" + index_day + "'")

        # Replace inf values
        merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        summary_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Post the next business day date as ran on date
        del merged_df['date']
        merged_df['date'] = index_day

        merged_df.to_sql(name='etf_etfrecrecords', schema=settings.CURRENT_DATABASE, index=False,
                         if_exists='append', con=wic_connection)
        summary_df.to_sql(name='etf_etfrecsummary', schema=settings.CURRENT_DATABASE, index=False,
                          if_exists='append', con=wic_connection)

        slack_message('generic.slack',
                      {'message': f'*SUCCESS: ETF Recs stored for {for_day}'},
                      channel=get_channel_name('etf-task-reports'), token=settings.SLACK_TOKEN)
        dbutils.add_task_record()
    except Exception as e:
        import traceback
        traceback.print_exc()
        slack_message('generic.slack',
                      {'message': '*ERROR: Error in solving ETF recs. Please check logs: ' + str(e)},
                      channel=get_channel_name('etf-task-errors'), token=settings.SLACK_TOKEN)
        dbutils.add_task_record(status=e)
    finally:
        wic_connection.close()
        index_connection.close()


@shared_task(bind=True)
def update_etf_pnl_and_bskt(self, record_progress=False):
    """ Task to generate P&L and Market on Close list for the ETF. Scheduled for every 5 mins from market open to close
    """

    # skipping today's execution if it's a holiday
    if holiday_utils.is_market_holiday():
        dbutils.add_task_record('Skipping execution in observance of holiday.')
        return

    if record_progress:
        progress_recorder = ProgressRecorder(self)
        progress_recorder.set_progress(10, 100)

    try:
        index_db_engine = create_engine("mysql://" + DB_USER + ":" + DB_PASSWORD + "@" + \
                                        INDEX_DB_HOST + "/" + DB_NAME)
        wic_db_engine = create_engine("mysql://" + DB_USER + ":" + DB_PASSWORD + "@" + WIC_DB_HOST + "/" + WIC_DB_NAME)
        index_con = index_db_engine.connect()
        wic_con = wic_db_engine.connect()
        pcf_df = pd.read_sql_query(get_pnl_queries(reference='pcf'), con=index_con)
        # try to get PCF INAV file too if it exists for a rebalance day!
        pcf_inav_df = pd.read_sql_query(get_pnl_queries(reference='pcf_inav'), con=index_con)

        if not pcf_inav_df.empty:
            # Check if max date is same as pcf date
            if pcf_inav_df['date_updated'].max() == pcf_df['date_updated'].max():
                pcf_df = pcf_inav_df
                print('Rebalance file is detected. Using PCFINAV file as PCF file...')

        eze_df = pd.read_sql_query(get_pnl_queries(reference='eze'), con=wic_con)
        nav_df = pd.read_sql_query(get_pnl_queries(reference='nav'), con=index_con)
        basket_valuation_df = pd.read_sql_query(get_pnl_queries(reference='basket_valuation'), con=index_con)
        tg_performance_df = pd.read_sql_query(get_pnl_queries(reference='tg_performance',
                                                              for_date=pcf_df['date_updated'].max().strftime(
                                                                  '%Y-%m-%d')), con=wic_con)
        opening_index_holdings = pd.read_sql_query(get_pnl_queries(reference='opening_index_holdings',
                                                                   for_date=pcf_df['date_updated'].max().strftime(
                                                                       '%Y-%m-%d')), con=index_con)
        if record_progress:
            progress_recorder.set_progress(35, 100)
        # Cleanups
        pcf_df['ticker'] = pcf_df['ticker'].str.replace("'", "")
        pcf_df['sedol'] = pcf_df['sedol'].str.replace("'", "")
        pcf_df = pd.merge(pcf_df, eze_df, how='left', on='sedol')
        pcf_df['ticker'] = pcf_df.apply(lambda x: x['ticker'] + ' LN' if x['cur'] == 'GBP' else x['ticker'], axis=1)
        # Adjust tickers absent in Eze to avoid index errors when fetching live prices
        pcf_df['eze_ticker'] = pcf_df.apply(lambda x: x['ticker'] if pd.isna(x['eze_ticker']) else x['eze_ticker'],
                                            axis=1)
        api_host = settings.SAPI_HOST
        if nav_df.empty:
            raise Exception('NAV Dataframe Empty...')
        if pcf_df.empty:
            raise Exception('PCF Dataframe Empty...')
        if basket_valuation_df.empty:
            raise Exception('Basket Valuation Dataframe Empty...')
        if tg_performance_df.empty:
            raise Exception('TradeGroup performances not found...')
        # Add the % of NAV/CU to pcf df
        nav_cu = nav_df.iloc[0]['nav_cu']
        pcf_df['pct_nav_cu'] = pcf_df['base_mv'] / nav_cu

        # Get Live Prices for all sedols
        # Replace eze ticker for CASH (BIL US)
        pcf_df.loc[pcf_df['sedol'] == 'BDFDQP1', 'eze_ticker'] = 'BIL US'
        pcf_df.loc[pcf_df['sedol'] == '5805103', 'eze_ticker'] = 'DVT FP'
        pcf_df.loc[pcf_df['sedol'] == 'BMQ7FW5', 'eze_ticker'] = 'BKKT US'

        unique_tickers = pcf_df['eze_ticker'].dropna().unique()
        unique_tickers = [x + ' EQUITY' for x in unique_tickers]
        # Get the prices based on SEDOLs for SWAPs since tickers don't work to fetch prices for SWAPS
        pcf_df['eze_ticker'].fillna('', inplace=True)
        swap_sedols = pcf_df[pcf_df['eze_ticker'].str.contains('SWAP')]['sedol'].dropna().unique()
        swap_sedols = [x + ' EQUITY' for x in swap_sedols]

        live_price_df = pd.DataFrame.from_dict(
                        bbgclient.bbgclient.get_secid2field(unique_tickers + swap_sedols, 'tickers', ['PX_LAST', 'BID', 'ASK'],
                                                            req_type='refdata',
                                                            api_host=api_host), orient='index').reset_index()


        # Also get the Live FX dataframe
        unique_currencies = [x + 'USD BGN CURNCY' for x in pcf_df['cur'].unique() if x != 'USD']
        live_fx_df = pd.DataFrame.from_dict(
            bbgclient.bbgclient.get_secid2field(unique_currencies, 'tickers', ['PX_LAST'],
                                                req_type='refdata',
                                                api_host=api_host), orient='index').reset_index()

        live_fx_df['PX_LAST'] = live_fx_df['PX_LAST'].apply(lambda x: float(x[0]) if x[0] else None)
        live_fx_df.columns = ['cur', 'live_fx_price']
        live_fx_df['cur'] = live_fx_df['cur'].apply(lambda x: x.replace('USD BGN CURNCY', ''))
        # Add USD
        live_fx_df.loc[live_fx_df.index.max() + 1] = ['USD', 1.0]

        live_price_df['PX_LAST'] = live_price_df['PX_LAST'].apply(lambda x: float(x[0]) if x[0] else None)
        live_price_df['BID'] = live_price_df['BID'].apply(lambda x: float(x[0]) if x[0] else None)
        live_price_df['ASK'] = live_price_df['ASK'].apply(lambda x: float(x[0]) if x[0] else None)
        live_price_df.columns = ['eze_ticker', 'live_price', 'security_bid', 'security_ask']
        live_price_df['eze_ticker'] = live_price_df['eze_ticker'].apply(lambda x: x.replace(' EQUITY', ''))

        pcf_df = pd.merge(pcf_df, live_price_df, how='left', on='eze_ticker')
        # At this point, we won't have prices for SEDOLs. Get those

        def get_px_for_swaps(row, for_column):
            sedol = row['sedol']
            _sedol_px = live_price_df[live_price_df['eze_ticker'] == sedol]
            if _sedol_px.empty:
                return np.NaN
            else:
                return _sedol_px.iloc[0][for_column]

        pcf_df['live_price'] = pcf_df.apply(lambda x: get_px_for_swaps(x, 'live_price') if pd.isna(x['live_price'])
                                            else x['live_price'], axis=1)
        pcf_df['security_bid'] = pcf_df.apply(lambda x: get_px_for_swaps(x, 'security_bid') if pd.isna(x['security_bid'])
                                            else x['security_bid'], axis=1)
        pcf_df['security_ask'] = pcf_df.apply(lambda x: get_px_for_swaps(x, 'security_ask') if pd.isna(x['security_ask'])
                                            else x['security_ask'], axis=1)

        # Merge with FX Df for Live FX prices
        pcf_df = pd.merge(pcf_df, live_fx_df, how='left', on='cur')

        # Clean up for GBP
        pcf_df['live_price'] = pcf_df.apply(
            lambda x: x['live_price'] / 100 if x['cur'] == 'GBP' else x['live_price'], axis=1)

        pcf_df['security_bid'] = pcf_df.apply(
            lambda x: x['security_bid'] / 100 if x['cur'] == 'GBP' else x['security_bid'], axis=1)

        pcf_df['security_ask'] = pcf_df.apply(
            lambda x: x['security_ask'] / 100 if x['cur'] == 'GBP' else x['security_ask'], axis=1)

        pcf_df['bid_market_value'] = pcf_df.apply(
            lambda x: x['security_bid'] * x['live_fx_price'] * x['basket_unit_size'] if x['cil'] == 'N' else None,
            axis=1)

        pcf_df['ask_market_value'] = pcf_df.apply(
            lambda x: x['security_ask'] * x['live_fx_price'] * x['basket_unit_size'] if x['cil'] == 'N' else None,
            axis=1)

        pcf_df['usd_live_price'] = pcf_df['live_price'] * pcf_df['live_fx_price']
        pcf_df['cil_mv'] = pcf_df.apply(
            lambda x: x['basket_unit_size'] * x['usd_live_price'] if x['cil'] == 'Y' else None, axis=1)

        pcf_df['px_change'] = pcf_df.apply(lambda x: (x['live_price'] - x['local_price']) / x['local_price'],
                                           axis=1)
        pcf_df['stock_return'] = pcf_df['px_change'] * pcf_df['weight']
        # Create the Deal return dataframe
        deal_return_df = pcf_df[['tradegroup', 'stock_return']].groupby(
                                'tradegroup').sum().reset_index().rename(columns={'stock_return': 'deal_return'})
        # merge it with pcf for a deal return column
        pcf_df = pd.merge(pcf_df, deal_return_df, how='left', on='tradegroup')
        pcf_df = pd.merge(pcf_df, tg_performance_df, how='left', on='tradegroup')
        pcf_df['live_ytd_return'] = pcf_df['ytd_return'] + pcf_df['deal_return']

        etf_ticker = settings.ETF_TICKER
        index_ticker = settings.INDEX_TICKER

        bbg_mneumonics = ['PX_LAST', 'VOLUME', 'BID', 'ASK']

        bbg_df = pd.DataFrame.from_dict(bbgclient.bbgclient.get_secid2field([etf_ticker, index_ticker],
                                                                            'tickers', bbg_mneumonics,
                                                                            req_type='refdata', api_host=api_host),
                                        orient='index').reset_index()

        if record_progress:
            progress_recorder.set_progress(60, 100)
        # Clean up
        for col in bbg_mneumonics:
            bbg_df[col] = bbg_df[col].apply(lambda x: float(x[0]) if x[0] else x[0])

        # 1. NAV, AUM and Spread Monitor
        custom_inputs = CustomUserInputs.objects.all().get()
        nav_df['collateral_buffer'] = custom_inputs.collateral_buffer
        nav_df['total_aum'] = nav_df['fund_aum'] + nav_df['collateral_buffer']
        nav_df['units_outstanding'] = (nav_df['shrs_outstanding'] // 50000).astype(int)

        start_nav = nav_df.iloc[0]['nav']
        daily_pnl_pct = pcf_df['stock_return'].sum()
        daily_pnl_dollar = daily_pnl_pct * nav_df.iloc[0]['total_aum'] / 100
        fv_nav = start_nav * (1 + daily_pnl_pct/100)
        daily_pnl_price = fv_nav - start_nav
        arbiv = bbg_df[bbg_df['index'] == index_ticker].iloc[0]['PX_LAST']
        arb_last_price = bbg_df[bbg_df['index'] == etf_ticker].iloc[0]['PX_LAST']
        volume = int(bbg_df[bbg_df['index'] == etf_ticker].iloc[0]['VOLUME'])
        nav_df['live_aum'] = nav_df['total_aum'] + daily_pnl_dollar
        spread_df_data = [start_nav, daily_pnl_pct, daily_pnl_dollar, daily_pnl_price, fv_nav, arbiv, arb_last_price,
                          volume]
        spread_df_columns = ['start_nav', 'daily_pnl_pct', 'daily_pnl_dollar', 'daily_pnl_price', 'fv_nav', 'arbiv',
                             'arb_last_price', 'volume']
        spread_df = pd.DataFrame(columns=spread_df_columns, data=[spread_df_data])

        bid_basket_mvs = pcf_df['bid_market_value'].sum()
        ask_basket_mvs = pcf_df['ask_market_value'].sum()

        cash = pcf_df['cil_mv'].sum()
        tax = custom_inputs.tax
        fees = custom_inputs.fees

        bid_basket_cost = bid_basket_mvs + cash - fees
        ask_basket_cost = ask_basket_mvs + cash - fees

        fv_bid = bid_basket_cost / 50000
        fv_ask = ask_basket_cost / 50000

        spread_estimate_dollar = (ask_basket_cost - bid_basket_cost) / 50000
        spread_estimate_percent = 1e2 * spread_estimate_dollar / fv_bid

        spread_estimate_data = [['Basket MVs', np.round(bid_basket_mvs, decimals=2),
                                 np.round(ask_basket_mvs, decimals=2)],
                                ['Cash', np.round(cash, decimals=2), np.round(cash, decimals=2)],
                                ['Tax', np.round(tax, decimals=2), np.round(tax, decimals=2)],
                                ['Fees', np.round(fees,decimals=2), np.round(fees,decimals=2)],
                                ['Basket Cost', np.round(bid_basket_cost, decimals=2),
                                 np.round(ask_basket_cost, decimals=2)],
                                ['Fair Value', np.round(fv_bid, decimals=2), np.round(fv_ask, decimals=2)],
                                ['$ Spread', '$' + str(np.round(spread_estimate_dollar, decimals=2)), '-'],
                                ['% Spread', str(np.round(spread_estimate_percent, decimals=2)) +'%', '-']
                                ]

        spread_estimate_monitor = pd.DataFrame(columns=['key', 'BID', 'ASK'], data=spread_estimate_data)
        # 2. Bid Ask Monitor Dataframe
        bid = bbg_df[bbg_df['index'] == etf_ticker].iloc[0]['BID']
        ask = bbg_df[bbg_df['index'] == etf_ticker].iloc[0]['ASK']

        bid_price_fvnav = bid - fv_nav
        ask_price_fvnav = ask - fv_nav

        prem_dis_bid = 1e2 * (bid - fv_nav) / fv_nav
        prem_dis_ask = 1e2 * (ask - fv_nav) / fv_nav

        bid_ask_data = [['Market Price', np.round(bid, decimals=2), np.round(ask, decimals=2)],
                        ['Market Price vs FV NAV', '$' + str(np.round(bid_price_fvnav, decimals=2)),
                         '$' + str(np.round(ask_price_fvnav, decimals=2))],
                        ['Premium/Discount', str(np.round(prem_dis_bid, decimals=2))+'%',
                         str(np.round(prem_dis_ask, decimals=2)) + '%']]
        bid_ask_df = pd.DataFrame(columns=['key', 'BID', 'ASK'], data=bid_ask_data)

        # Spread DF

        bid_ask_spread = ask - bid
        bid_ask_spread_df = pd.DataFrame(columns=['key', 'Market', 'Fair_Value'], data=[['Spread', bid_ask_spread,
                                                                                         spread_estimate_dollar]])

        # Basket Valuation Dataframe

        basket_valuation_df['net'] = basket_valuation_df['basket_market_value'] + basket_valuation_df['cil'] + \
                                     basket_valuation_df['cash_component'] + basket_valuation_df['estimated_expense'] - \
                                     basket_valuation_df['estimated_dividends']

        basket_valuation_df['basket_market_value_pct'] = 1e2 * basket_valuation_df['basket_market_value'] / nav_cu
        basket_valuation_df['cil_pct'] = 1e2 * basket_valuation_df['cil'] / nav_cu
        basket_valuation_df['cash_component_pct'] = 1e2 * basket_valuation_df['cash_component'] / nav_cu
        basket_valuation_df['estimated_expense_pct'] = 1e2 * basket_valuation_df['estimated_expense'] / nav_cu
        basket_valuation_df['estimated_dividends_pct'] = 1e2 * basket_valuation_df['estimated_dividends'] / nav_cu
        basket_valuation_df['net_pct'] = 1e2 * basket_valuation_df['net'] / nav_cu
        basket_valuation_df['nav_cu_boolean'] = 1e2 * basket_valuation_df['net'] == nav_cu

        # Upload all monitors to the database...
        now = datetime.datetime.now()

        if record_progress:
            progress_recorder.set_progress(80, 100)

        # Process the ETF Creation unit activity monitor and the Market on Close file

        # ETF Creation Unit Activity Dataframe
        shares_outstanding = nav_df.iloc[0]['shrs_outstanding']
        units_outstanding = int(shares_outstanding / 50000)
        net_td_creations = custom_inputs.net_td_creations
        net_td_redemptions = custom_inputs.net_td_redemptions

        net_td_cus = net_td_creations + net_td_redemptions
        eod_units_out = units_outstanding + net_td_cus
        eod_shares_outstanding = shares_outstanding + (net_td_cus * 50000)

        etf_creation_unit_activity_data = [[shares_outstanding, units_outstanding, net_td_creations, net_td_redemptions,
                                            net_td_cus, eod_units_out, eod_shares_outstanding]]
        etf_creation_unit_activity_df = pd.DataFrame(columns=['shares_outstanding', 'units_outstanding',
                                                              'net_td_creations', 'net_td_redemptions', 'net_td_cus',
                                                              'eod_units_out', 'eod_shares_outstanding'],
                                                     data=etf_creation_unit_activity_data)

        # % of NAV/CU is used as the weight here
        bskt_df = pcf_df[['date_updated', 'eze_ticker', 'cil', 'cur', 'sedol', 'tradegroup', 'base_price',
                          'basket_unit_size', 'pct_nav_cu', 'eze_shares']].copy()

        # Merge with Index holdings to get index weight
        bskt_df = pd.merge(bskt_df, opening_index_holdings, how='left', on='sedol')
        bskt_df[['index_weight', 'base_price']] = bskt_df[['index_weight', 'base_price']].astype(float)

        bskt_df['sod_units'] = units_outstanding
        bskt_df['sod_calc_shares'] = nav_df.iloc[0]['total_aum'] * bskt_df['index_weight'] / bskt_df['base_price']
        bskt_df['eze_shares'] = bskt_df['eze_shares'].fillna(0).astype(int)

        bskt_df['wic_cleanup'] = bskt_df['sod_calc_shares'] - bskt_df['eze_shares']
        bskt_df['eod_units'] = eod_units_out
        bskt_df['eod_shares'] = bskt_df.apply(
            lambda x: (x['eze_shares'] + ((x['eod_units'] - x['sod_units']) * x['basket_unit_size'])), axis=1)

        bskt_df['ap_trades'] = bskt_df.apply(
            lambda x: 0 if x['cil'] == 'Y' else ((x['eod_units'] - x['sod_units']) * x['basket_unit_size']), axis=1)

        bskt_df['wic_trades'] = bskt_df.apply(
            lambda x: 0 if (pd.isna(x['eod_shares']) or x['pct_nav_cu'] > 0) else round(
                (x['eod_units'] - x['sod_units']) * x['basket_unit_size']), axis=1)

        wic_trades_df = bskt_df[((bskt_df['sedol'] != 'BDFDQP1') & (bskt_df['cil'] == 'Y'))][
            ['cil', 'sedol', 'cur', 'eze_ticker', 'tradegroup', 'eze_shares', 'wic_trades']].rename(
            columns={'wic_trades': 'trade_size'})

        def get_trade_direction(row):
            """ Inner function to determine the side of trade per ticker """
            side = 'N/A'
            if row['eze_shares'] > 0:
                if row['trade_size'] > 0:
                    side = 'BUY'
                else:
                    side = 'SELL'
            else:
                if row['trade_size'] > 0:
                    side = 'COVER'
                else:
                    side = 'SHORT'
            return side

        wic_trades_df['side'] = wic_trades_df.apply(lambda x: get_trade_direction(x), axis=1)
        wic_trades_df['tradedate'] = now
        # Assign Trader - ML for US names and FR for foreign names
        wic_trades_df['trader'] = wic_trades_df.apply(lambda x: 'ML' if x['cur'] == 'USD' else 'FR', axis=1)

        # Assign broker - MSWE for Swap and GTS for other
        wic_trades_df['broker'] = wic_trades_df.apply(lambda x: 'MSWE' if 'swap' in x['eze_ticker'].lower() else 'GTS',
                                                      axis=1)

        # Assign a static manager for now
        wic_trades_df['manager'] = 'MAP'

        # Convert the amount to absolute values for representation purposes
        wic_trades_df['trade_size'] = wic_trades_df['trade_size'].apply(lambda x: abs(x))

        wic_trades_df.rename(columns={'tradegroup': 'strategy1', 'trade_size': 'amount', 'eze_ticker': 'security',
                                      }, inplace=True)
        wic_trades_df.drop(columns=['cil', 'sedol', 'eze_shares', 'cur'], inplace=True)

        wic_trades_df['prt'] = 'ARBETF'

        monitors_data = [now, spread_df.to_json(orient='records'), bid_ask_df.to_json(orient='records'),
                         bid_ask_spread_df.to_json(orient='records'), nav_df.to_json(orient='records'),
                         basket_valuation_df.to_json(orient='records'),
                         etf_creation_unit_activity_df.to_json(orient='records'),
                         spread_estimate_monitor.to_json(orient='records')]

        monitors_df = pd.DataFrame(columns=['updated_on', 'spread_monitor', 'bid_ask_monitor',
                                            'bid_ask_spread_monitor', 'nav_monitor', 'basket_valuation_monitor',
                                            'unit_activity_monitor', 'spread_estimate_monitor'], data=[monitors_data])
        # Save to DB
        ETFLivePnL.objects.all().delete()
        ETFMonitors.objects.all().delete()
        MarketOnClose.objects.all().delete()

        if record_progress:
            progress_recorder.set_progress(94, 100)

        monitors_df.to_sql(name='etf_etfmonitors', schema=settings.CURRENT_DATABASE, index=False, if_exists='append',
                           con=wic_con)

        pcf_df.drop(columns=['date_updated'], inplace=True)
        pcf_df['updated_on'] = now

        pcf_df.to_sql(name='etf_etflivepnl', schema=settings.CURRENT_DATABASE, index=False, if_exists='append',
                      con=wic_con)
        wic_trades_df.to_sql(name='etf_marketonclose', schema=settings.CURRENT_DATABASE, index=False,
                             if_exists='append', con=wic_con)

    except Exception as e:
        slack_message('ESS_IDEA_DATABASE_ERRORS.slack',
                      {'message': 'Error while updating ETF live P&L & Market on Close list :',
                       'errors': str(e)},
                      channel=get_channel_name('portal-task-errors'), token=settings.SLACK_TOKEN)
    finally:
        index_con.close()
        wic_con.close()


