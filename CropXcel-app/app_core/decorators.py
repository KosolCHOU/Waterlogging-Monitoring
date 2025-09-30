# app_core/decorators.py
from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect

def login_required_with_message(view_func):
    """
    Same as login_required, but adds a friendly message
    shown on the login page, not after redirect.
    """
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.info(request, "⚠️ Please sign in to access this page.")
            # send them to login explicitly
            return redirect(f"{settings.LOGIN_URL}?next={request.path}")
        return view_func(request, *args, **kwargs)
    return wrapper
