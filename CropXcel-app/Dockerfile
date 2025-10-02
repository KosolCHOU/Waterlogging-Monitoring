# Dockerfile for CropXcel
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=cropxcel_project.settings_production

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
        build-essential \
        libpq-dev \
        gdal-bin \
        libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app/
# Normalize line endings for shell script on Windows checkouts
RUN sed -i 's/\r$//' start.sh

RUN chmod +x start.sh

# Create media directory structure
RUN mkdir -p /app/media/probes /app/media/overlays /app/media/hotspots /app/media/insights \
    /app/media/plots /app/media/stacks /app/media/timeseries /app/media/aoi /app/media/avatars

# Expose port
EXPOSE 8000

# Default command starts migrations, collectstatic, and gunicorn (honors $PORT)
CMD ["./start.sh"]
