from django.contrib import admin

# Register your models here.
from .models import Label, Dataset, Image, Picture
 
admin.site.register(Label)
admin.site.register(Dataset)
admin.site.register(Image)
admin.site.register(Picture)


