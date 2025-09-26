# app_core/decorators.py
from django.contrib.auth.decorators import login_required
from django.contrib import messages

def login_required_with_message(view_func):
    """
    Same as login_required, but adds a friendly message.
    """
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.info(request, "⚠️ Please sign in to access this page.")
        return login_required(view_func)(request, *args, **kwargs)
    return wrapper
