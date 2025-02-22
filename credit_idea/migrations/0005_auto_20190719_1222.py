# Generated by Django 2.0.13 on 2019-07-19 12:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('credit_idea', '0004_creditideacreditdetails_creditideacreditscenario'),
    ]

    operations = [
        migrations.AddField(
            model_name='creditideacreditscenario',
            name='is_deal_closed',
            field=models.TextField(blank=True, default='No', null=True),
        ),
        migrations.AddField(
            model_name='creditideacreditscenario',
            name='is_downside',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='creditideacreditscenario',
            name='is_upside',
            field=models.BooleanField(default=False),
        ),
    ]
