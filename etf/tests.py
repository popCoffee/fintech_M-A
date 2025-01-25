from django.test import TestCase

# Create your tests here.
def fix_pnls(start_date=None):
    #fix currency pnls. for function daily_etf_index_pnl and table DailyIndexPnL
    index_connection = index_db_engine.connect()
    closing_index_ccy = pd.read_sql_query(get_pnl_tab_queries('ccy_closing', start_date=start_date,
                                                              end_date=start_date),con=index_connection)
    currency_list = closing_index_ccy['currency'].unique()
    df = pd.DataFrame.from_records( DailyIndexPnL.objects.filter(date=datetime.datetime.strptime(start_date, '%Y%m%d').date()).values() )
    print(start_date)
    print(df[['pnl','ticker']].tail(4))
    df['pnl'] = df.apply(lambda row: -1*row['pnl'] if row['ticker'] in list(currency_list) else row['pnl'], axis=1)
    DailyIndexPnL.objects.filter(date=datetime.datetime.strptime(start_date, '%Y%m%d').date()).delete()
    df.to_sql(name='etf_dailyindexpnl', con=settings.SQLALCHEMY_CONNECTION, if_exists='append',
                      index=False, schema=settings.CURRENT_DATABASE)

def get_pnls(dt=None):
    '''get the pnls from daily index and sum for a day'''
    from etf.models import DailyIndexPnL
    import pandas as pd;import datetime
    if not dt:
        return
    dt=datetime.datetime.strptime(dt, '%Y%m%d').date()
    # previous_dip = DailyIndexPnL.objects.filter(date=dt).order_by('-date')
    df = pd.DataFrame.from_records( DailyIndexPnL.objects.filter(date=dt  ).values() )
    pnls_col = df['pnl'].sum()
    # PnL_Pct = closing_price - adjusted_open_price / opening_price * percentage_weighting + (c.closing_fx - o.closing_fx) / o.closing_fx) * percentage_weighting
    return pnls_col

def loop_pnls(month=0):
    # multiple pnl pull
    print('pulling daily pnls for multiple dates:')
    res=[]
    for day in range(1, 32):
        fd = f"{2024:04d}{month:02d}{day:02d}"
        # fde= "2024"+month+str(day)
        print(fd, ' date')
        try:
            pnl = get_pnls( fd)
            res.append(pnl)
        except Exception as e:
            print('passing on ', fd, str(e)[0:70])
    res_back = [res[i] for i in range(len(res)-1,-1,-1)]
    return res_back


def get_dups_pnl(dt=None):
    '''get all dailyindex  duplicates as a df'''
    from etf.models import DailyIndexPnL
    import pandas as pd;import datetime
    if dt:
        dt=datetime.datetime.strptime(dt, '%Y%m%d').date()
    # previous_dip = DailyIndexPnL.objects.filter(date=dt).order_by('-date')
    df = pd.DataFrame.from_records( DailyIndexPnL.objects.all().values() )
    count = df.groupby(['date','tradegroup','sedol'], as_index=False, dropna=False).size()
    dups = count[count['size']>1]
    return dups

def compare_data_on_dates(a,b,c,col='tradegroup',diffs=False):
    '''compare dfs on dates , a='2024-02-02', c='20240202'. return 3 dfs OR list difference  '''
    from etf.models import DailyIndexPnL
    import pandas as pd;
    from etf.etf_queries import get_pnl_tab_queries
    from etf.tasks import tradar_db_engine
    tradar_connection = tradar_db_engine.connect()
    import datetime
    # if a:
    #     dt = datetime.datetime.strptime(a, '%Y%m%d').date()
    # if b:
    #     dt = datetime.datetime.strptime(a, '%Y%m%d').date()
    df = pd.DataFrame.from_records(DailyIndexPnL.objects.filter(date=a  ).values())
    df2 = pd.DataFrame.from_records(DailyIndexPnL.objects.filter(date=b  ).values())
    tradar = pd.read_sql_query(get_pnl_tab_queries('tradar', start_date_tradar=c,
                                                      end_date_tradar=c), con=tradar_connection)
    if diffs:
        diff = [x for x in list(set(df[col])) if x not in list(df2[col])]
        # diff2= [x for x in list(set(df[col])) if x not in list(tradar[col])]
        return diff
    return df,df2[col],tradar


def compare_pnl_date(a=None,b=None):
    '''compare pnls on a single date with diffs'''
    print('running tests.py')
    data= dict()
    if not a and not b:
        a='2024-02-14'
        b='20240214'
    from etf.models import DailyIndexPnL
    import pandas as pd;
    from etf.etf_queries import get_pnl_tab_queries
    from etf.tasks import tradar_db_engine
    tradar_connection = tradar_db_engine.connect()

    df = pd.DataFrame.from_records(DailyIndexPnL.objects.filter(date=a).values())
    tradar = pd.read_sql_query(get_pnl_tab_queries('tradar', start_date_tradar=b,
                                                   end_date_tradar=b), con=tradar_connection)


    df = df[['pnl','ticker','tradegroup','sectype','sedol','dividend_pnl','closing_price']]
    tradar['sedol'] = tradar['sedol'].fillna(tradar['ticker'])
    tradar = tradar[['Date', 'ticker','sedol', 'SecType', 'Pct_pnl']].groupby(['Date', 'ticker','sedol',
                                                                                         'SecType']). \
        sum().reset_index()
    dfm = pd.merge(df, tradar, left_on=['sedol'],
                        right_on=['sedol'], how='outer')
    # import ipdb;ipdb.set_trace()

    # dfm['pnl'] = dfm['pnl'].fillna(0)
    dfm['diff'] = dfm['Pct_pnl'] - dfm['pnl']
    dfm['Pctdiff'] = dfm['diff'] / dfm['pnl']

    dfm=dfm.fillna('NA')

    data['detailed_tracking_error'] = dfm.to_json(orient='records')
    return data


def get_pnl_day_remove(a=None):
    '''remove one row at a time and get sum pnl'''
    print('running tests.py')
    data= []
    curr=['NOK','EUR','GBP']
    from etf.models import DailyIndexPnL
    import pandas as pd
    df = pd.DataFrame.from_records(DailyIndexPnL.objects.filter(date=a).values())
    visited=[]
    for t in df.ticker.to_list():
        if t not in visited:
            dft = df.drop(df[df.ticker == t].index)
            data.append( [round(sum(dft.pnl.to_list()),6),t]  )
            visited.append(t)
    return data