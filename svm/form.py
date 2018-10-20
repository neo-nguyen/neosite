from django import forms


class PictureUploadForm(forms.Form):
    picture = forms.ImageField()
