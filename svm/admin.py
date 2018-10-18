from django.contrib import admin

# Register your models here.
from .models import Label, Dataset, Image
 
admin.site.register(Label)
admin.site.register(Dataset)
admin.site.register(Image)

