# Generated by Django 2.0.13 on 2023-09-06 12:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('credit_idea', '0016_auto_20230905_1653'),
    ]

    operations = [
        migrations.AddField(
            model_name='creditstaticscreen',
            name='fund',
            field=models.CharField(blank=True, max_length=16, null=True),
        ),
    ]
