# Generated by Django 3.2.5 on 2022-04-19 14:12

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('second_phase', '0005_record_emotion_report'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='record',
            name='emotion_report',
        ),
    ]