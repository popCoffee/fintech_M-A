# Generated by Django 2.0.10 on 2020-09-22 15:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('etf', '0004_auto_20200922_1539'),
    ]

    operations = [
        migrations.AlterField(
            model_name='etflivepnl',
            name='tradegroup',
            field=models.CharField(max_length=50, null=True),
        ),
    ]
