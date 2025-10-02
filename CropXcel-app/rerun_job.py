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
from app_core.tasks import run_waterlogging_analysis

# Get the latest completed job
job = AnalysisJob.objects.filter(status="done").order_by("-id").first()

if job:
    print(f"Found job {job.id}, re-running it to test new code...")

    # Reset the job status
    job.status = "running"
    job.result = {}
    job.save()

    # Trigger the task
    task = run_waterlogging_analysis.delay(job.id)

    print(f"Task {task.id} submitted for job {job.id}")
else:
    print("No completed jobs found")
