# Generated by Django 2.0.10 on 2020-09-24 16:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('etf', '0005_auto_20200922_1543'),
    ]

    operations = [
        migrations.AddField(
            model_name='etfmonitors',
            name='unit_activity_monitor',
            field=models.TextField(null=True),
        ),
    ]
