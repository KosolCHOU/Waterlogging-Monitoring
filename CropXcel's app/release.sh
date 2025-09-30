#!/usr/bin/env bash
# Release phase script for Heroku
python manage.py collectstatic --noinput
python manage.py migrate --noinput
