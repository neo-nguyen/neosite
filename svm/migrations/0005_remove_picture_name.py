# Generated by Django 2.1.2 on 2018-10-19 05:47

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('svm', '0004_picture'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='picture',
            name='name',
        ),
    ]