""" URL Mapping to Controller for Risk Reporting App. """
from django.conf.urls import url
from . import views
from . import risk_factors_summary

app_name = 'risk_reporting'




urlpatterns = [
    url('merger_arb_risk_attributes', views.merger_arb_risk_attributes, name='merger_arb_risk_attributes'),
    url('arb_sector_concentration_trend$', views.arb_sector_concentration_trend, name='arb_sector_concentration_trend'),
    url('merger_arb_nav_impacts', views.merger_arb_nav_impacts, name='merger_arb_nav_impacts'),
    url('formula_based_downsides$', views.FormulaDownsideView.as_view(), name='formula_based_downsides'),
    url('downside_history_report$', views.create_downside_history_streamer, name='downside_history_report'),
    url('credit_upside_downsides$', views.CreditDealsUpsideDownsideView.as_view(), name='credit_upside_downsides'),
    url('risk_factors_summary$', views.RiskFactorsSummaryView.as_view(), name='risk_factors_summary'),
    url('basecase_nav_by_approvals$', views.BaseCaseNavByApprovalsView.as_view(), name='basecase_nav_by_approvals'),
    url('get_rows_data$', risk_factors_summary.get_rows_data, name='get_rows_data'),
    url('update_credit_deals_upside_downside$', views.update_credit_deals_upside_downside, name='update_credit_deals_upside_downside'),
    url('fetch_from_bloomberg_by_spread_index$', views.fetch_from_bloomberg_by_spread_index, name='fetch_from_bloomberg_by_spread_index'),
    url('fetch_peer_index_by_tradegroup$', views.fetch_peer_index_by_tradegroup, name='fetch_peer_index_by_tradegroup'),
    url('get_custom_input$', views.get_custom_input, name='get_custom_input'),
    url('credit_deals_csv_import$', views.credit_deals_csv_import, name='credit_deals_csv_import'),
    url('get_details_from_arb$', views.get_details_from_arb, name='get_details_from_arb'),
    url('update_credit_deal_risk_limit$', views.update_credit_deal_risk_limit, name='update_credit_deal_risk_limit'),
    url('update_downside_formulae$', views.update_downside_formulae, name='update_downside_formulae'),
    url('update_risk_limit$', views.update_risk_limit, name='update_risk_limit'),
    url('formulae_downsides_new_deal_add$', views.formulae_downsides_new_deal_add, name='formulae_downsides_new_deal_add'),
    url('security_info_download$', views.security_info_download, name='security_info_download'),
    url('deal_info_download$', views.deal_info_download, name='deal_info_download'),
    url('formula_based_downsides_download$', views.formula_based_downsides_download, name='formula_based_downsides_download'),
    url('download_nav_data$', views.download_nav_data, name='download_nav_data'),
    url('get_realtime_downsides', views.get_realtime_downsides, name='get_realtime_downsides'),
    url('arb_aed_risk_multiple', views.ArbAedRiskMultipleView.as_view(), name='arb_aed_risk_multiple'),
    url('get_ess_cix_nav_risk_data$', views.get_ess_cix_nav_risk_data, name='get_ess_cix_nav_risk_data'),
    url(r'^peer_downsides$', views.peer_downsides, name='peer_downsides'),
    url(r'^add_peer_downsides$', views.add_peer_downsides, name='add_peer_downsides')
]
