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

# Get job 7 specifically
job = AnalysisJob.objects.get(id=7)

print(f"Job {job.id}:")
print(f"  Field ID: {job.field_id}")
print(f"  Status: {job.status}")
print(f"  Created at: {job.created_at}")
print(f"  Finished at: {job.finished_at}")
print(f"  Message: {job.message}")
print(f"  Result: {job.result}")
