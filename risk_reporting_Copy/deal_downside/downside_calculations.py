from datetime import datetime, date, timedelta, time
import decimal
import logging
import traceback

import numpy as np
from celery import shared_task
from django.contrib.contenttypes.models import ContentType
from django.db import transaction
from typing import List

from slack_sdk import WebClient

import bbgclient
import pandas as pd

import dbutils
from WicPortal_Django import settings
from risk.ma_regression_utils import bloomberg_peers
from risk.models import MaDealsActionIdDetails, MA_Deals, MaDownsidePeerSource
from risk_reporting.deal_downside.downside_context import DownsideCalculationContext
from risk_reporting.deal_downside.strategies.regression_strategy import LinearRegressionStrategy
from risk_reporting.models import EqualWeightedDownside, DealDownside, LinearRegressionDownside, AnalystDownside, \
    HistoricalDealDownside
from slack_utils import download_file_from_slack

logger = logging.getLogger(__name__)


def create_or_update_equal_weighted_model(deal_name: str, underlying: str, peer_source: MaDownsidePeerSource = None,
                                          weights_only=False) -> decimal.Decimal:
    """ Calculate the latest downside for the equal weighted model, and then stores the value in the DealDownside table
    and the EqualWeightedDownside table

    @param deal_name: str deal name deal_name in MA_Deals
    @param underlying: str underlying target/acquirer name
    @param peer_source: MaDownsidePeerSource optional peer source to use inplace of the all peers
    @param weights_only: bool toggle to return the weight or the weight adjusted downside
    @return: float downside value
    """

    downside_value: decimal.Decimal = decimal.Decimal(0)
    pct_weight: decimal.Decimal = decimal.Decimal(0)
    try:
        # find all peers
        deal_object = MA_Deals.objects.get(deal_name=deal_name)
        if peer_source:
            peers = [peer_source]
        else:
            peers = MaDownsidePeerSource.objects.filter(deal_id=deal_object)
        for peer in peers:
            with transaction.atomic():
                number_of_peers: int = len(peer.proxy_list.split(','))
                # see if there is an existing equal weighted downside object
                equal_weighted_downside = EqualWeightedDownside.objects.filter(peer_source=peer).first()
                if equal_weighted_downside is None:  # if there is no existing equal weighted downside object
                    equal_weighted_downside = EqualWeightedDownside(peer_source=peer)
                equal_weighted_downside.number_of_peers = number_of_peers
                equal_weighted_downside.save()

                pct_weight, unaffected_downside = generate_equal_weighted_downside(peer, deal_object.action_id)
                downside_value = pct_weight * unaffected_downside
                deal_downside, _ = DealDownside.objects.get_or_create(
                    deal=deal_object, underlying=underlying,
                    content_type=ContentType.objects.get_for_model(EqualWeightedDownside),
                    object_id=equal_weighted_downside.pk,
                    defaults={'downside': downside_value}
                )
                deal_downside.downside = round(pct_weight * unaffected_downside, 3)
                deal_downside.save()
    except Exception:
        traceback.print_exc()
    if weights_only:
        return pct_weight * 100
    else:
        return downside_value


def generate_equal_weighted_downside(peer: MaDownsidePeerSource, action_id: int):
    """given a peer source, return the equal weighted downside and the unaffected downside"""

    pct_weight = decimal.Decimal(0)
    unaffected_downside = decimal.Decimal(0)
    try:
        # get unaffected date/price from action_id
        action_id_details = MaDealsActionIdDetails.objects.get(action_id=action_id)
        unaffected_date = action_id_details.unaffected_date
        # toggle to return the weight adjusted downside or just the weight in percentage
        unaffected_downside = min(x for x in [action_id_details.unaffected_downside,
                                              action_id_details.unaffected_90d_vwap,
                                              action_id_details.unaffected_price] if x is not None)
        unaffected_downside = decimal.Decimal(action_id_details.unaffected_downside)
        tickers = peer.proxy_list.split(',')
        tickers = [ticker.strip() for ticker in tickers]
        pct_weight = get_peer_index_weights(tickers, unaffected_date) * decimal.Decimal(0.01)

    except Exception:
        logging.error(traceback.format_exc())
        traceback.print_exc()
    return pct_weight, unaffected_downside


def get_peer_index_weights(ticker_list, unaffected_date) -> decimal.Decimal:
    """ function to generate equal weights for a given list of peer tickers and a date
    @param ticker_list: list of str tickers to look up in bloomberg
    @param unaffected_date: datetime/str date to look up tickers on
    """

    def clean_up_bbg_result(input_df):
        input_df['PX_LAST'] = input_df['PX_LAST'].apply(lambda x: float(x[0])
        if isinstance(x, list) and x != [] and x[0] is not None else np.nan)
        input_df.dropna(inplace=True)
        input_df.rename(columns={'index': 'ticker'}, inplace=True)
        return input_df

    if isinstance(unaffected_date, date):
        unaffected_date = unaffected_date.strftime('%Y%m%d')
    if isinstance(unaffected_date, datetime):
        unaffected_date = unaffected_date.strftime('%Y%m%d')
    if isinstance(unaffected_date, str):
        unaffected_date = unaffected_date.replace('-', '')

    # get bbg LAST_PX of tickers on unaffected_date
    unaffected_df = pd.DataFrame.from_dict(bbgclient.bbgclient.get_secid2field(ticker_list, 'tickers', ['PX_LAST'],
                                                                               start_date=unaffected_date,
                                                                               end_date=unaffected_date,
                                                                               req_type='histdata'
                                                                               ), orient='index').reset_index()
    unaffected_df = clean_up_bbg_result(unaffected_df)
    unaffected_df['weight'] = 100.0 / unaffected_df['PX_LAST'] / unaffected_df.shape[0]
    del unaffected_df['PX_LAST']
    last_price_df = pd.DataFrame.from_dict(
        bbgclient.bbgclient.get_secid2field(ticker_list, 'tickers', ['PX_LAST'], req_type='refdata',
                                            ), orient='index').reset_index()
    last_price_df = clean_up_bbg_result(last_price_df)

    joint_df = unaffected_df.merge(last_price_df, on='ticker', how='left')
    joint_df.dropna(inplace=True)
    # reset index as its own column

    return decimal.Decimal(sum(joint_df['weight'] * joint_df['PX_LAST']))


def update_equal_weighted_downside(deal_downside: DealDownside) -> None:
    """update the equal weighted downside for a given deal downside object"""
    try:
        equal_weighted_downside = deal_downside.content_object
        pct_weight, unaffected_downside = generate_equal_weighted_downside(equal_weighted_downside.peer_source,
                                                                           int(deal_downside.deal.action_id))
        downside_value = pct_weight * unaffected_downside
        deal_downside.downside = downside_value
        deal_downside.save()
    except Exception as e:
        logger.error(f'Error updating equal weighted downside: {e}')


def update_regression_downside(deal_downside: DealDownside) -> None:
    """update the regression downside for a given deal downside object"""
    update_downside_from_model(deal_downside)


def update_deal_downsides():
    """
        main process for recalculating all deal downsides
    """
    handlers = {
        ContentType.objects.get_for_model(EqualWeightedDownside): update_equal_weighted_downside,
        ContentType.objects.get_for_model(LinearRegressionDownside): update_regression_downside,
    }
    # iterate through all live deal downsides
    deal_downsides = DealDownside.objects.filter(deal__archived=False)
    for deal_downside in deal_downsides:
        handler = handlers.get(deal_downside.content_type)
        if handler:
            handler(deal_downside)


def generate_bloomberg_peer_source(deal: MA_Deals) -> MaDownsidePeerSource:
    """ generate a default bloomberg peer source from a deal object"""
    list_of_peers = bloomberg_peers(deal.target_ticker)
    peer_source = MaDownsidePeerSource(deal_id=deal,
                                       proxy_name='Bloomberg',
                                       proxy_list=', '.join(list_of_peers))
    peer_source.save()
    return peer_source


def regenerate_regression_downside_from_peer_source(peer_source: MaDownsidePeerSource) -> None:
    """ delete and regenerate the regression downside for a given peer source"""
    delete_regression_downside(peer_source)
    create_regression_downside_from_peer_source(peer_source)


def regenerate_regression_downside(deal_object: MA_Deals) -> None:
    """ delete and regenerate all the regression downsides for a given deal object"""
    # delete existing regression model related to the deal
    regression_downsides = DealDownside.objects.filter(deal=deal_object,
                                                       content_type=ContentType.objects.
                                                       get_for_model(LinearRegressionDownside))
    for regression_downside in regression_downsides:
        regression_model = regression_downside.content_object
        regression_model.delete()
        regression_downside.delete()
    # find all peer sources for the deal
    peer_sources = MaDownsidePeerSource.objects.filter(deal_id=deal_object)
    for peer_source in peer_sources:
        # create regression model for each peer source
        create_regression_downside_from_peer_source(peer_source)


def delete_regression_downside(peer_source: MaDownsidePeerSource) -> None:
    with transaction.atomic():
        regression_downsides = LinearRegressionDownside.objects.filter(peer_source=peer_source)

        for regression_downside in regression_downsides:

            deal_downsides = regression_downside.downsides.all()
            for deal_downside in deal_downsides:
                deal_downside.delete()

            regression_downside.delete()


@shared_task(bind=True)
def create_regression_downside_from_peer_source(self, peer_source: MaDownsidePeerSource) -> List[DealDownside]:
    """ create regression downsides from a peer source, returns a list of created DealDownsides
    """
    created_downsides = []
    try:
        if peer_source.deal_id.archived:
            # deal is archived, skip
            return created_downsides
    except Exception:
        # deal does not exist, skip
        return created_downsides

    context = DownsideCalculationContext(LinearRegressionStrategy(LinearRegressionDownside.DAILY_PRICES))
    context.prepare_data(peer_source)
    latest_data = context.get_new_data()

    for year_duration in [1, 3, 5]:  # iterate over needed year durations
        generated = context.generate_model({'year_duration': year_duration})
        if generated:
            downside = context.calculate_downside_from_model(latest_data)
            model = context.save_model()
            deal_downside = DealDownside(deal=peer_source.deal_id,
                                         content_type=ContentType.objects.get_for_model(LinearRegressionDownside),
                                         object_id=model.pk,
                                         downside=downside,
                                         underlying=peer_source.deal_id.target_ticker)
            deal_downside.save()
            created_downsides.append(deal_downside)
    return created_downsides


def update_downside_from_model(deal_downside: DealDownside, data=None) -> float:
    """ update the regression downside for a given deal downside object using parameters from saved models."""
    downside_value: float = 0
    try:
        if deal_downside.deal.archived:
            # deal is archived, skip
            return downside_value

        # handle different downside types here
        content_type = deal_downside.content_type.model_class()
        strategy = None
        if content_type == LinearRegressionDownside:
            strategy = LinearRegressionStrategy()

        context = DownsideCalculationContext(strategy)
        context.load_model(deal_downside.content_object)
        downside_value = context.calculate_downside_from_model(data)
        deal_downside.downside = downside_value
        deal_downside.save()
    except Exception as e:
        logger.error(f'Error updating model downside {e}:\n {traceback.format_exc()}')
    return downside_value


@shared_task(bind=True)
def capture_analyst_downside(self):
    active_deals = MA_Deals.objects.filter(archived=False).values_list('id', 'deal_name', 'target_ticker',
                                                                       'acquirer_ticker')
    active_df = pd.DataFrame(active_deals, columns=['id', 'deal_name', 'target_ticker', 'acquirer_ticker'])
    analyst = AnalystDownside.objects.filter(analyst='JD').first()
    # get all chat history from the last time the task was run
    selected_time = datetime.combine(datetime.today() - timedelta(1), time(21, 30, 00))
    client = WebClient(token=settings.SLACK_BOT_TOKEN)
    chat_hist = client.conversations_history(channel='C03D2SND24Q', oldest=selected_time.timestamp())

    # Iterate through the messages
    for message in chat_hist['messages'][::-1]:
        # Check if the message has files
        if 'files' not in message:
            continue

        for file in message['files']:
            file_url = file['url_private']
            file_name = file['name']

            # Check if the file is a downside file
            if not file_name.lower().startswith('jd downsides'):
                continue
            # parse file name to get the date
            date_str: str = file_name.split(' ')[2].split('.')[0]  # date in month-day-year format
            date_str: datetime = datetime.strptime(date_str, "%m-%d-%y")

            # check last date for analyst downside to avoid duplicate insertion
            # insert only if the date is greater than the last date in the database
            last_date = HistoricalDealDownside.objects.filter(object_id=analyst.id,
                                                              content_type=ContentType.objects.get_for_model(
                                                                  AnalystDownside)
                                                              ).order_by('-date').first()
            if not (last_date is None or last_date.date < date_str.date()):
                continue

            # read the data into a dataframe
            try:
                data = download_file_from_slack(file_url)
                downside_df = pd.read_excel(data)
            except Exception as e:
                logger.error(f"Error reading downside file: {e}")
                continue
            downside_df['date'] = date_str

            # select and rename the columns we need
            downside_df = downside_df[['Deal', 'Median', 'date']]
            downside_df.rename(columns={'Deal': 'ticker', 'Median': 'downside'}, inplace=True)
            downside_df['ticker'] = downside_df['ticker'].str.upper()

            # merge the data with existing deal
            target_df = pd.merge(downside_df, active_df, left_on='ticker', right_on='target_ticker', how='left')
            acquirer_df = pd.merge(downside_df, active_df, left_on='ticker', right_on='acquirer_ticker', how='left')
            merged_df = pd.concat([target_df, acquirer_df])
            merged_df = merged_df.drop(columns=['target_ticker', 'acquirer_ticker'])
            # drop rows with NaN values
            merged_df = merged_df.dropna()

            # merging causes int columns to be converted to float, convert them back to int
            merged_df['id'] = merged_df['id'].astype(int)

            # iterate through the merged_df and save the downside
            for index, row in merged_df.iterrows():
                deal_downside = HistoricalDealDownside(content_type=ContentType.objects.get_for_model(AnalystDownside),
                                                       object_id=analyst.id,
                                                       deal_id=row['id'],
                                                       underlying=row['ticker'],
                                                       date=row['date'],
                                                       downside=row['downside'])
                deal_downside.save()
    dbutils.add_task_record()
