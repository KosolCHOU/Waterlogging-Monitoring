#!/usr/bin/env python
import os
import sys
import django

# Add the project directory to the Python path
sys.path.append("/app")

# Set the Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cropxcel_project.settings")

# Setup Django
django.setup()

from app_core.models import AnalysisJob

# Get the latest 3 jobs
jobs = AnalysisJob.objects.order_by("-id")[:3]

print(f"Found {jobs.count()} jobs")

for job in jobs:
    print(f"Job {job.id}: status={job.status}")
    results = job.result or {}

    plot_url = results.get("plot_url", "None")
    insights_url = results.get("insights_csv_url", "None")
    area_calc = results.get("area_calculation", "None")

    print(f"  plot_url: {plot_url}")
    print(f"  insights_csv_url: {insights_url}")
    print(f"  area_calculation: {area_calc}")
    print("---")
