from django.conf import settings


def get_pnl_tab_queries(for_type, start_date=None, end_date=None, start_date_tradar=None, end_date_tradar=None):
    query = ""
    if for_type.lower() == 'eze':
        query = "SELECT Flat_file_as_of as `date`, sedol, Ticker as eze_ticker, " \
                "       TradeGroup as deal, PctOfSleeveCurrent as eze " \
                "FROM " \
                "wic.daily_flat_file_db " \
                "WHERE " \
                f"Flat_file_as_of >= '{start_date}' and Flat_file_as_of <= '{end_date}' AND " \
                f" fund = 'ARBETF' AND sedol NOT LIKE 'BDFDQP1'"

    elif for_type.lower() == 'state_street':
        query = "SELECT date_updated as date, description, ticker, cur, sedol, shares, local_price, local_mv, " \
                "       forex, base_price, base_mv, weight as basket, cil, est_dividend, new, " \
                "       share_change " \
                f"FROM {settings.ETF_DB_NAME}.statestreet_wicpcf " \
                "WHERE " \
                f"date_updated >= '{start_date}' and date_updated <= '{end_date}' "

    elif for_type.lower() == 'solactive':
        query = "SELECT Date as `date`, security_sedol AS sedol, 1e2*percentage_weighting AS _index " \
                "FROM " \
                "Wic_Index.closing_holdings_wimarb " \
                "WHERE " \
                f" Date >= '{start_date}' AND Date<='{end_date}' AND security_sedol NOT LIKE 'BDFDQP1'"

    elif for_type.lower() == 'nav':
        query = f"SELECT * FROM prod_etf_db.wicetf_api_dailyetfmetrics WHERE as_of>='{start_date}' " \
                f" AND as_of <='{end_date}'"

    elif for_type.lower() == 'tradar':
        query = "SELECT CAST(CONVERT(VARCHAR(8), PAF.timeKey) AS DATE) AS [Date], TF.fund, S.strategyName as tradegroup, " \
                "       SEC.ticker, SEC.sedol, PAF.pnlFC as pnl, FPF.valueFCEnd as NAV, STYPE.groupingName as SecType," \
                " 100*(PAF.pnlFC/FPF.valueFCEnd) as Pct_pnl " \
                " FROM " \
                " dbo.PerformanceAttributionFact PAF INNER JOIN dbo.TradeFund TF " \
                " ON PAF.fundKey = TF.fundId " \
                " INNER JOIN dbo.Strategy S ON S.stratId = PAF.stratKey " \
                " INNER JOIN dbo.Sec SEC ON SEC.secId = PAF.secIdKey " \
                " INNER JOIN dbo.FundPerformanceFact FPF ON TF.fundId = FPF.fundKey AND FPF.timeKey = PAF.timeKey " \
                " INNER JOIN dbo.SecType STYPE ON STYPE.sectype = SEC.sectype" \
                " WHERE " \
                f" TF.fund = 'ARBETF' AND PAF.pnlFC <> 0 AND PAF.timeKey >= {start_date_tradar} AND " \
                f" PAF.timekey <= {end_date_tradar} "

    elif for_type.lower() == 'closing_holdings_wimarb':
        query = f"SELECT * FROM Wic_Index.closing_holdings_wimarb WHERE Date >= '{start_date}' " \
                f" AND Date<='{end_date}'"
    elif for_type.lower() == 'opening_holdings_wimarb':
        query = f"SELECT * FROM Wic_Index.opening_holdings_wimarb WHERE Date >= '{start_date}'" \
                f" AND Date<='{end_date}'"

    elif for_type.lower() == 'ccy_closing':
        query = f"SELECT * FROM Wic_Index.index_currency_info WHERE Date>='{start_date}'" \
                f" AND Date<='{end_date}'"
    elif for_type.lower() == 'ccy_opening':
        query = f"SELECT * FROM Wic_Index.index_currency_info_opening WHERE Date>='{start_date}'" \
                f" AND Date<='{end_date}'"
    elif for_type.lower() == 'index_profile_opening':
        query = f"SELECT index_value_unhedged FROM Wic_Index.index_profile_opening WHERE Date = '{end_date}' "
    elif for_type.lower() == 'solactive_dcaf':
        query = f"SELECT * FROM prod_etf_db.solactive_dcaf WHERE dcaf_date = '{end_date}' "
    elif for_type.lower() == 'dcaf_div':
        query = f"SELECT * FROM prod_etf_db.solactive_dcaf WHERE event_effective_date = '{end_date}' "
    else:
        query = 'N/A'

    return query


def get_pnl_queries(reference=None, for_date=None):
    query = None
    if reference == 'pcf':
        query = "SELECT date_updated, ticker,cur,  " \
                "sedol, shares AS basket_unit_size, base_mv, local_price,base_price, weight, cil  " \
                f"FROM {settings.ETF_DB_NAME}.statestreet_wicpcf  " \
                f"WHERE date_updated = (SELECT MAX(date_updated) FROM {settings.ETF_DB_NAME}.statestreet_wicpcf)"

    elif reference == 'pcf_inav':
        query = "SELECT date_updated, ticker,cur,  " \
                "sedol, shares AS basket_unit_size, base_mv, local_price,base_price, weight, cil  " \
                f"FROM {settings.ETF_DB_NAME}.statestreet_wicpcfinav  " \
                f"WHERE date_updated = (SELECT MAX(date_updated) FROM {settings.ETF_DB_NAME}.statestreet_wicpcfinav)"

    elif reference == 'eze':
        query = "SELECT tradegroup, ticker as eze_ticker, sedol, amount as eze_shares FROM " \
                "wic.daily_flat_file_db WHERE " \
                "Flat_file_as_of = (SELECT MAX(flat_file_as_of) FROM wic.daily_flat_file_db) AND Fund like 'ARBETF'"

    elif reference == 'nav':
        query = "SELECT date_updated as trade_date, nav_per_shr as nav, nav_per_cu as nav_cu, " \
                "total_net_assets as fund_aum, shrs_outstanding " \
                f"FROM {settings.ETF_DB_NAME}.statestreet_wicnav " \
                f"WHERE date_updated = (SELECT MAX(date_updated) FROM {settings.ETF_DB_NAME}.statestreet_wicnav)"

    elif reference == 'basket_valuation':
        query = "SELECT trade_date, basket_market_value, actual_total_cash as cil, " \
                "actual_cash_component as cash_component, estimated_expense, estimated_dividends, " \
                "nav_per_creation_unit " \
                "FROM " \
                f"{settings.ETF_DB_NAME}.statestreet_wicpcfinfo " \
                f"WHERE trade_date = (SELECT MAX(trade_date) FROM {settings.ETF_DB_NAME}.statestreet_wicpcfinfo)"

    elif reference == 'tg_performance':
        query = "SELECT tradegroup, days_1_bps/100 as one_day_return, days_5_bps/100 as five_day_return, " \
                "ytd_bps/100 as ytd_return FROM " \
                f"{settings.CURRENT_DATABASE}.funds_snapshot_tradegroupperformancefundnavbps " \
                f"WHERE date = '{for_date}' " \
                "AND fund like 'ARBETF' "

    elif reference == 'opening_index_holdings':
        query = "SELECT security_sedol AS sedol, percentage_weighting AS index_weight " \
                f"FROM Wic_Index.opening_holdings_wimarb WHERE Date = '{for_date}'"

    return query
