# Generated by Django 2.0.10 on 2020-10-14 11:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('etf', '0014_auto_20201014_1145'),
    ]

    operations = [
        migrations.AddField(
            model_name='etfrecrecords',
            name='sectype',
            field=models.CharField(max_length=20, null=True),
        ),
    ]
