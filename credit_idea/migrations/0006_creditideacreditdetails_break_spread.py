# Generated by Django 2.0.13 on 2019-07-22 12:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('credit_idea', '0005_auto_20190719_1222'),
    ]

    operations = [
        migrations.AddField(
            model_name='creditideacreditdetails',
            name='break_spread',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
