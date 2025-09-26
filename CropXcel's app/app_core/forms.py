from django import forms
from .models import AOI

class AOIUploadForm(forms.Form):
    aoi_file = forms.FileField(
        label="Upload AOI (GeoJSON)",
        help_text="Upload a .geojson file describing your field",
        allow_empty_file=False
    )

# --- Profile image form ---
from .models import Profile
from django import forms

class ProfileImageForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ["avatar"]

    def clean_avatar(self):
        f = self.cleaned_data.get("avatar")
        if not f:
            return f
        # Size limit: 2 MB
        if f.size > 2 * 1024 * 1024:
            raise forms.ValidationError("Please upload an image ≤ 2MB.")
        # Basic type check
        valid = {"image/jpeg", "image/png", "image/webp"}
        if getattr(f, "content_type", "") not in valid:
            raise forms.ValidationError("Use JPEG/PNG/WEBP.")
        return f


