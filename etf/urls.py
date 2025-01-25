""" URL Mapping to Controller for Securities App. """
from django.conf.urls import url
from . import views
app_name = 'etf'

urlpatterns = [
    url('get_etf_positions$', views.get_etf_positions, name='get_etf_positions'),
    url('get_etf_performances$', views.get_etf_performances, name='get_etf_performances'),
    url('get_tradegroup_etf_pnl$', views.get_tradegroup_etf_pnl, name='get_tradegroup_etf_pnl'),
    url('get_tracking_error$', views.ETFTrackingErrorView.as_view(), name='get_tracking_error'),
    url('get_tracking_error_results_json$', views.get_tracking_error_results_json,
        name='get_tracking_error_results_json'),
    url('recs/', views.EtfRecs.as_view(), name='etfrecs'),
    url('etf_pnl/', views.EtfPnl.as_view(), name='etf_pnl'),
    url('calculate_live_etf_pnl$', views.calculate_live_etf_pnl, name='calculate_live_etf_pnl'),
    url('update_custom_user_inputs$', views.update_custom_user_inputs, name='update_custom_user_inputs'),
    url('daily_pnl_crosscheck$', views.ETFDailyTrackingView.as_view(), name='check_daily_pnl'),
    url('daily_pnl_crosscheck_post$', views.daily_pnl_crosscheck_post, name='daily_pnl_crosscheck_post'),
]
