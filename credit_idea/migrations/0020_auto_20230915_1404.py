# Generated by Django 2.0.13 on 2023-09-15 14:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('credit_idea', '0019_auto_20230914_1448'),
    ]

    operations = [
        migrations.AlterField(
            model_name='creditstaticscreen',
            name='claw_end',
            field=models.DateField(blank=True, max_length=50, null=True),
        ),
    ]
