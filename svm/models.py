from django.db import models
from random import randint
from pathlib import Path
from django import db

# Create your models here.


class Label(models.Model):
    label_name = models.CharField(max_length=50)
    description = models.CharField(max_length=100, null=True, default=None)

    def __str__(self):
        return self.label_name


class Dataset(models.Model):
    dataset_name = models.CharField(max_length=50)

    def __str__(self):
        return self.dataset_name


class Image(models.Model):
    image_path = models.CharField(max_length=300)
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)

    def __str__(self):
        return self.image_path

    """get each label 1 random img, return list img related with label"""

    def get_random_per_label(self):
        images = []
        labels = Label.objects.all()

        for label in labels:
            imgs = Image.objects.select_related(
                'label').filter(label_id=label.id)
            rand = randint(0, imgs.count()-1)
            images.append(imgs[rand])
        return images


"""get database path from app"""

db_path = db.utils.settings.DATABASES['default']['NAME']

"""get database name and create a upload folder with this name in local computer"""

db_name = Path(db_path).parts[-1]
upload_dir = 'xuantoan/Data1/upload_neosite/' + db_name


class Picture(models.Model):
    picture = models.ImageField(upload_to=upload_dir)

    def get_upload_dir(self):
        return self.picture.url
