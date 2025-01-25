# Generated by Django 2.0.13 on 2024-04-01 19:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('etf', '0017_auto_20240327_2045'),
    ]

    operations = [
        migrations.AddField(
            model_name='dailyindexpnl',
            name='closing_fx',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='dailyindexpnl',
            name='closing_price',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='dailyindexpnl',
            name='fraction_of_shares',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='dailyindexpnl',
            name='percentage_weighting',
            field=models.FloatField(null=True),
        ),
    ]
