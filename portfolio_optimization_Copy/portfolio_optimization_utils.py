# Calculates PL (Used for Gross RoR accounting for Hedges)
def calculate_pl_sec_impact(row):
    if row['SecType'] in ['EQ', 'EQSWAP'] and row['AlphaHedge'] == 'Hedge':
        return 0    # Assume 0 pnl impact on Equity Hedges. Cant say for sure where acquirer would trade

    # Hedges on Options
    #if row['SecType'] == 'EQSWAP':
    #    return (row['deal_value'] * row['QTY']) - (row['CurrentMktVal'])  # No Fx Adjustment for Swaps. Use Dollor Exp

    if row['SecType'] == 'EXCHOPT' and row['AlphaHedge'] == 'Hedge':
        return -row['CurrentMktVal'] + (row['SecurityPrice']*row['QTY'])

    if row['SecType'] != 'EXCHOPT':
        return (row['deal_value'] * row['QTY'] * row['FxFactor']) - (row['CurrentMktVal'])

    if row['PutCall'] == 'CALL':
        if row['StrikePrice'] <= row['deal_value']:
            x = (row['deal_value'] - row['StrikePrice']) * (row['QTY'])
        else:
            x = 0
    elif row['PutCall'] == 'PUT':
        if row['StrikePrice'] >= row['deal_value']:
            x = (row['StrikePrice'] - row['deal_value']) * (row['QTY'])
        else:
            x = 0
    return -row['CurrentMktVal'] + x


def calculate_arb_hedged_pl(row):
    if row['SecType'] in ['EQ', 'EQSWAP'] and row['AlphaHedge'] == 'Hedge':
        return 0    # Assume 0 pnl impact on Equity Hedges. Cant say for sure where acquirer would trade

    # Hedges on Options
    if row['SecType'] == 'EXCHOPT' and row['AlphaHedge'] == 'Hedge':
        return -row['CurrentMktVal'] + (row['SecurityPrice']*row['QTY'])

    if row['SecType'] != 'EXCHOPT':
        return (row['deal_upside'] * row['QTY'] * row['FxFactor']) - (row['CurrentMktVal'])

    if row['PutCall'] == 'CALL':
        if row['StrikePrice'] <= row['deal_upside']:
            x = (row['deal_upside'] - row['StrikePrice']) * (row['QTY'])
        else:
            x = 0
    elif row['PutCall'] == 'PUT':
        if row['StrikePrice'] >= row['deal_upside']:
            x = (row['StrikePrice'] - row['deal_upside']) * (row['QTY'])
        else:
            x = 0
    return -row['CurrentMktVal'] + x


def calculate_cr_hedge_pl(row):
    x = 0
    if row['sleeve'] == 'Credit Opportunities' and not row['target_ticker']:
        if row['SecType'] != 'EXCHOPT':
            return row['QTY'] * (row['upside'] - row['SecurityPrice']) * row['FxFactor']
        if row['SecType'] == 'EXCHOPT':
            if row['PutCall'] == 'CALL':
                if row['StrikePrice'] < row['upside']:
                    x = -1 * row['QTY'] * (row['StrikePrice'] - row['upside'])
                else:
                    x = 0
            elif row['PutCall'] == 'PUT':
                x = 0
                if row['StrikePrice'] > row['upside']:
                    x = row['QTY'] * (row['StrikePrice'] - row['upside'])
        return x - row['CurrentMktVal']
    return 0


def calculate_cr_break_pl(row):
    result = 0
    if row['sleeve'] in ['Credit Opportunities', 'Equity Special Situations'] and row['target_ticker']:
        if row['LongShort'] == 'Long':
            if row['SecType'] == 'EXCHOPT':
                result = -1 * row['net'] *(row['Strike'] - row['deal_downside'])
                result = result - abs(row['net']) * row['Price']
            else:
                result = row['net'] * (row['deal_downside'] - row['Price'])
        else:
            if row['SecType'] == 'EXCHOPT':
                result = -1 * row['net'] * (row['Strike'] - row['DealUpside'])
                result = result - abs(row['net']) * row['Price']
            else:
                result = row['net'] * (row['DealUpside'] - row['Price'])
    return result


def calculate_cr_hedge_ror(row):
    if not row['PnL'] or row['PnL'] == '0':
        return 0
    else:
        if row['AlphaHedge'] == 'Hedge':
            return 1e2 * row['pnl_impact'] / row['pct_of_sleeve_current']
        return 0


def calculate_gross_spread_ror(row):
    if row['AlphaHedge'] == 'Alpha':
        return 1e2 * row['arb_nav_pnl'] / row['delta_adj_pct_aum']
    return 0


def calculate_mstrat_weighted_ror(row):
    if row['catalyst'] == 'Hard' and row['catalyst_rating'] == '2':
        return 'Conv'
    else:
        return (row['curr_rwd_ror'] + row['curr_cwd_ror']) * 0.5
