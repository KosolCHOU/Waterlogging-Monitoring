# app_core/templatetags/form_extras.py
from django import template
register = template.Library()

@register.filter
def add_class(field, css):
    # preserve existing attrs and append classes
    attrs = field.field.widget.attrs.copy()
    prev = attrs.get("class", "")
    attrs["class"] = (prev + " " + css).strip()
    return field.as_widget(attrs=attrs)
