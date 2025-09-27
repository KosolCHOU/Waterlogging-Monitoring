from django import forms
from .models import Profile
import re

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
        if f.size > 2 * 1024 * 1024:
            raise forms.ValidationError("Please upload an image ≤ 2MB.")
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
        fields = ["full_name", "phone", "date_of_birth", "main_crop", "province"]
        widgets = {
            "full_name": forms.TextInput(attrs={"class":"input", "placeholder":"e.g., Sokha Phan"}),
            "phone": forms.TextInput(attrs={
                "class": "input",
                "placeholder": "e.g., 012 345 678",
                "inputmode": "tel",
                "pattern": r"^[0-9+\s()-]{6,20}$"
            }),
            "date_of_birth": forms.DateInput(attrs={"type":"date", "class":"input"}),
            "main_crop": forms.Select(attrs={"class":"select"}),
            "province": forms.Select(attrs={"class":"select"}),
        }

    def clean_phone(self):
        raw = (self.cleaned_data.get("phone") or "").strip()
        if not raw:
            return raw  # allow blank

        # Remove all non-digits
        digits = re.sub(r"\D", "", raw)

        # Validate: must start with 0 or 855
        if not digits.startswith("0") and not digits.startswith("855"):
            raise forms.ValidationError("Phone must start with 0 or +855")

        # If it's +855xxx (9 digits after 855), normalize to 0xxx…
        if digits.startswith("855"):
            if len(digits) == 11:  # e.g., 85512345678
                digits = "0" + digits[3:]  # → 012345678
            else:
                raise forms.ValidationError("Invalid +855 phone number length")

        # Now expect 9 digits total (Cambodian numbers are usually 9)
        if len(digits) != 9:
            raise forms.ValidationError("Phone number must have 9 digits (e.g., 012345678).")

        # Format into groups of 3
        formatted = f"{digits[0:3]} {digits[3:6]} {digits[6:9]}"
        return formatted