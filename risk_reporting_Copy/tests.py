import os
from datetime import datetime, timedelta, time

import pandas as pd
import requests
from celery import shared_task
from django.contrib.contenttypes.models import ContentType
from django.db import transaction
from slack_sdk import WebClient

import dbutils
import holiday_utils
from WicPortal_Django import settings
from risk.models import MaDownsidePeerSource, MA_Deals
from risk_reporting.deal_downside.downside_calculations import delete_regression_downside
from risk_reporting.deal_downside.downside_context import DownsideCalculationContext
from risk_reporting.deal_downside.strategies.regression_strategy import LinearRegressionStrategy
from risk_reporting.models import LinearRegressionDownside, DealDownside, AnalystDownside, HistoricalDealDownside
from slack_utils import download_file_from_slack




def test():
    deal = MA_Deals.objects.get(id=11076)
    deal_downsides = DealDownside.objects.filter(deal=deal, content_type=ContentType.objects.get_for_model(
        LinearRegressionDownside))
    selected_model = None
    for regression_downside in deal_downsides:
        if regression_downside.content_object.is_selected:
            selected_model = regression_downside
            break


def test_deletion(id=1030):
    with transaction.atomic():
        regression_downside = LinearRegressionDownside.objects.filter(peer_source_id__exact=id).first()
        deal_downsides = DealDownside.objects.filter(
            object_id=regression_downside.pk,
            content_type=ContentType.objects.get_for_model(LinearRegressionDownside)
        )

        # Delete all related DealDownside instances
        for deal_downside in deal_downsides:
            deal_downside.delete()

        # Delete the LinearRegressionDownside instance
        regression_downside.delete()


def test_recalc_from_model():
    deal_downsides_ids = test_downside_from_peer_sources()
    deal_downsides = DealDownside.objects.filter(content_type_id__in=deal_downsides_ids)
    for deal_downside in deal_downsides:
        print(f"Processing {deal_downside.id}: {deal_downside.deal.deal_name}")
        # handle different downside types here
        downside_type = deal_downside.content_type.model_class()
        if downside_type == LinearRegressionDownside:
            strategy = LinearRegressionStrategy()
        context = DownsideCalculationContext(strategy)
        context.load_model(deal_downside.content_object)
        downside = context.calculate_downside_from_model()
        print(f"Downside from model: {downside}, year duration:{context.strategy.year_duration}")


def test_downside_from_peer_sources():
    # iterate over all peer sources
    peer_sources = MaDownsidePeerSource.objects.all()
    downside_ids = []
    for peer_source in peer_sources:
        try:
            deal_name = peer_source.deal_id.deal_name
        except Exception:
            print(f"Skipping {peer_source.id}: deal_id does not exist")
            continue
        if peer_source.deal_id.archived:
            print(f"Skipping {peer_source.id} for {peer_source.deal_id.deal_name}: deal is archived")
            continue

        print(f"Processing {peer_source.id}: {deal_name}")
        context = DownsideCalculationContext(LinearRegressionStrategy(LinearRegressionDownside.DAILY_PRICES))
        context.prepare_data(peer_source)
        latest_data = context.get_new_data()
        for j in [1, 3, 5]:
            generated = context.generate_model({'year_duration': j})
            if generated:
                downside = context.calculate_downside_from_model(latest_data)
                model = context.save_model()
                deal_downside = DealDownside(deal=peer_source.deal_id,
                                             content_type=ContentType.objects.get_for_model(LinearRegressionDownside),
                                             object_id=model.pk,
                                             downside=downside,
                                             underlying=peer_source.deal_id.target_ticker)
                deal_downside.save()
                downside_ids.append(deal_downside.id)
                print(f"Downside for {peer_source.deal_id.deal_name} with {j} years duration: {downside}")
    return downside_ids
