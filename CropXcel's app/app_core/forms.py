from django import forms
from .models import Profile

class AOIUploadForm(forms.Form):
    aoi_file = forms.FileField(
        label="Upload AOI (GeoJSON)",
        help_text="Upload a .geojson file describing your field",
        allow_empty_file=False
    )

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

KH_PROVINCES = [
    ("Banteay Meanchey","Banteay Meanchey"), ("Battambang","Battambang"),
    ("Kampong Cham","Kampong Cham"), ("Kampong Chhnang","Kampong Chhnang"),
    ("Kampong Speu","Kampong Speu"), ("Kampong Thom","Kampong Thom"),
    ("Kampot","Kampot"), ("Kandal","Kandal"), ("Kep","Kep"), ("Kratié","Kratié"),
    ("Mondul Kiri","Mondul Kiri"), ("Oddar Meanchey","Oddar Meanchey"),
    ("Pailin","Pailin"), ("Phnom Penh","Phnom Penh"), ("Preah Sihanouk","Preah Sihanouk"),
    ("Preah Vihear","Preah Vihear"), ("Prey Veng","Prey Veng"),
    ("Pursat","Pursat"), ("Ratanak Kiri","Ratanak Kiri"), ("Siem Reap","Siem Reap"),
    ("Stung Treng","Stung Treng"), ("Svay Rieng","Svay Rieng"),
    ("Takeo","Takeo"), ("Tbong Khmum","Tbong Khmum"),
]

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ["main_crop", "province"]
        widgets = {
            "main_crop": forms.TextInput(attrs={"class":"input","placeholder":"e.g., Sen Kra’op"}),
            "province":  forms.Select(choices=[("", "— Select province —")]+KH_PROVINCES,
                                      attrs={"class":"input"}),
        }