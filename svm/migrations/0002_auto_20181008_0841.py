# Generated by Django 2.1.2 on 2018-10-08 08:41

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('svm', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='dataset',
            old_name='name_dataset',
            new_name='dataset_name',
        ),
        migrations.RenameField(
            model_name='image',
            old_name='dataset',
            new_name='dataset_id',
        ),
        migrations.RenameField(
            model_name='image',
            old_name='label',
            new_name='label_id',
        ),
        migrations.RenameField(
            model_name='label',
            old_name='name_label',
            new_name='label_name',
        ),
    ]
