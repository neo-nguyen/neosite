# Generated by Django 2.1.2 on 2018-11-01 14:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('svm', '0007_auto_20181101_1438'),
    ]

    operations = [
        migrations.AlterField(
            model_name='picture',
            name='picture',
            field=models.ImageField(upload_to='upload-pictures'),
        ),
    ]