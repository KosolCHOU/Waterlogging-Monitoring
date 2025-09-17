from django import forms
from .models import AOI

class AOIUploadForm(forms.Form):
    aoi_file = forms.FileField(
        label="Upload AOI (GeoJSON)",
        help_text="Upload a .geojson file describing your field",
        allow_empty_file=False
    )

