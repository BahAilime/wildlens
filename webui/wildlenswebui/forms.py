from django import forms

class TrackUploadForm(forms.Form):
    track_image = forms.ImageField()