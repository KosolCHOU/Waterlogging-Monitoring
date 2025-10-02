import django

django.setup()

from app_core.models import AnalysisJob

print(f"Total jobs: {AnalysisJob.objects.count()}")

jobs = AnalysisJob.objects.all()[:3]
for job in jobs:
    print(f"Job {job.id}: status={job.status}")
    print(f"  - plot_url: {job.results.get('plot_url', 'None')}")
    print(f"  - insights_csv_url: {job.results.get('insights_csv_url', 'None')}")
    print(f"  - area_calculation: {job.results.get('area_calculation', 'None')}")
    print("---")
