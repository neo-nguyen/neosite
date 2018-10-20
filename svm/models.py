from django.db import models

# Create your models here.


class Label(models.Model):
    label_name = models.CharField(max_length=50)

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

class Picture(models.Model):
    picture = models.ImageField(upload_to='pictures')

