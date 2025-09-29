# app_core/forms.py
from django import forms
from .models import Profile
import re
from .models import PROVINCE_CHOICES, MAIN_CROP_CHOICES 

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


# app_core/forms.py
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Profile, PROVINCE_CHOICES, MAIN_CROP_CHOICES
from django import forms
import re

class SignupForm(UserCreationForm):
    # All optional fields
    full_name = forms.CharField(required=False, label="Full name")
    phone = forms.CharField(
        required=False, label="Phone",
        widget=forms.TextInput(attrs={
            "inputmode": "tel",
            "placeholder": "e.g., 012 345 678",
            "pattern": r"^[0-9+\s()-]{6,20}$"
        })
    )
    date_of_birth = forms.DateField(required=False, label="Date of birth",
                                    widget=forms.DateInput(attrs={"type":"date"}))
    main_crop = forms.ChoiceField(
        required=False, label="Main rice variety",
        choices=[("", "— (optional) —")] + list(MAIN_CROP_CHOICES)
    )
    province = forms.ChoiceField(
        required=False, label="Province",
        choices=[("", "— (optional) —")] + list(PROVINCE_CHOICES)
    )

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username",)

    def clean_phone(self):
        raw = (self.cleaned_data.get("phone") or "").strip()
        if not raw:
            return raw
        digits = re.sub(r"\D", "", raw)
        if not digits.startswith("0") and not digits.startswith("855"):
            raise forms.ValidationError("Phone must start with 0 or +855")
        if digits.startswith("855"):
            if len(digits) == 11:
                digits = "0" + digits[3:]
            else:
                raise forms.ValidationError("Invalid +855 phone number length")
        if len(digits) != 9:
            raise forms.ValidationError("Phone number must have 9 digits (e.g., 012345678).")
        return f"{digits[0:3]} {digits[3:6]} {digits[6:9]}"

    def save(self, commit=True):
        # 1) create user
        user = super().save(commit=False)
        if commit:
            user.save()

        # 2) optionally split full name to User.first/last
        full = (self.cleaned_data.get("full_name") or "").strip()
        if full:
            parts = full.split()
            if not (user.first_name or user.last_name):
                user.first_name = parts[0]
                user.last_name = " ".join(parts[1:]) if len(parts) > 1 else ""
                user.save(update_fields=["first_name", "last_name"])

        # 3) write profile fields (never None for CharField)
        prof, _ = Profile.objects.get_or_create(user=user)
        prof.full_name     = full
        prof.phone         = (self.cleaned_data.get("phone") or "").strip()
        prof.date_of_birth = self.cleaned_data.get("date_of_birth") or None

        crop = (self.cleaned_data.get("main_crop") or "").strip()
        prov = (self.cleaned_data.get("province") or "").strip()
        if crop:
            prof.main_crop = crop
        if prov:
            prof.province = prov

        prof.save()
        return user

class CropRecForm(forms.Form):
    N = forms.FloatField(min_value=0, label="Nitrogen (N)")
    P = forms.FloatField(min_value=0, label="Phosphorus (P)")
    K = forms.FloatField(min_value=0, label="Potassium (K)")
    temperature = forms.FloatField(label="Temperature (°C)")
    humidity = forms.FloatField(min_value=0, max_value=100, label="Humidity (%)")
    pH = forms.FloatField(min_value=0, max_value=14, label="Soil pH")
    rainfall = forms.FloatField(min_value=0, label="Rainfall (mm)")

    def cleaned_features(self):
        cd = self.cleaned_data
        return {
            "N": cd["N"],
            "P": cd["P"],
            "K": cd["K"],
            "temperature": cd["temperature"],
            "humidity": cd["humidity"],
            "pH": cd["pH"],
            "rainfall": cd["rainfall"],
        }
    
# app_core/forms.py
from django import forms
from django.core.exceptions import ValidationError

def dfield(minv, maxv, step=None):
    w = forms.NumberInput(attrs={
        "class": "number",
        "min": str(minv), "max": str(maxv),
        "step": str(step if step is not None else 1)
    })
    return forms.DecimalField(min_value=minv, max_value=maxv, widget=w)

class CropRecommendForm(forms.Form):
    N  = dfield(0, 300, 1)
    P  = dfield(0, 200, 1)
    K  = dfield(0, 250, 1)
    temperature = dfield(10, 50, 0.1)
    humidity    = dfield(0, 100, 1)
    pH          = dfield(3.5, 9.5, 0.1)
    rainfall    = dfield(0, 600, 1)

    def clean(self):
        data = super().clean()
        # example cross-field sanity: if rainfall>400 and N>200 → warn
        if data.get("rainfall") and data.get("N"):
            if float(data["rainfall"]) > 400 and float(data["N"]) > 200:
                raise ValidationError("Very high rainfall + very high nitrogen may increase lodging risk.")
        return data
