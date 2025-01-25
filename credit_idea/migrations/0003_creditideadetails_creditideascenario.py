# Generated by Django 2.0.13 on 2019-07-15 13:16

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('credit_idea', '0002_creditidea_arb_tradegroup'),
    ]

    operations = [
        migrations.CreateModel(
            name='CreditIdeaDetails',
            fields=[
                ('credit_idea', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, primary_key=True, serialize=False, to='credit_idea.CreditIdea')),
                ('nav_pct_impact', models.FloatField(blank=True, null=True)),
                ('topping_big_upside', models.FloatField(blank=True, null=True)),
                ('base_case_downside', models.FloatField(blank=True, null=True)),
                ('base_case_downside_type', models.CharField(blank=True, max_length=50, null=True)),
                ('outlier_downside', models.FloatField(blank=True, null=True)),
                ('outlier_downside_type', models.CharField(blank=True, max_length=50, null=True)),
                ('target_ticker', models.CharField(blank=True, max_length=20, null=True)),
                ('acq_ticker', models.CharField(blank=True, max_length=20, null=True)),
                ('cash_consideration', models.FloatField(blank=True, null=True)),
                ('share_consideration', models.FloatField(blank=True, null=True)),
                ('deal_value', models.FloatField(blank=True, null=True)),
                ('target_dividend', models.FloatField(blank=True, null=True)),
                ('acq_dividend', models.FloatField(blank=True, null=True)),
                ('fund_assets', models.FloatField(blank=True, null=True)),
                ('float_so', models.FloatField(blank=True, null=True)),
                ('acq_pb_rate', models.FloatField(blank=True, null=True)),
                ('target_pb_rate', models.FloatField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='CreditIdeaScenario',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('scenario', models.CharField(blank=True, max_length=50, null=True)),
                ('last_price', models.FloatField(blank=True, default=0, null=True)),
                ('dividends', models.FloatField(blank=True, default=0, null=True)),
                ('rebate', models.FloatField(blank=True, default=0, null=True)),
                ('hedge', models.FloatField(blank=True, default=0, null=True)),
                ('deal_value', models.FloatField(blank=True, default=0, null=True)),
                ('spread', models.FloatField(blank=True, default=0, null=True)),
                ('gross_pct', models.FloatField(blank=True, default=0, null=True)),
                ('annual_pct', models.FloatField(blank=True, default=0, null=True)),
                ('days_to_close', models.IntegerField(blank=True, default=0, null=True)),
                ('dollars_to_make', models.FloatField(blank=True, default=0, null=True)),
                ('dollars_to_lose', models.FloatField(blank=True, default=0, null=True)),
                ('implied_prob', models.FloatField(blank=True, default=0, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('estimated_closing_date', models.DateField(blank=True, null=True)),
                ('credit_idea', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='credit_idea.CreditIdea')),
            ],
        ),
    ]
