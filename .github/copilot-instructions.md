# 🌾 CropXcel Waterlogging Monitoring - AI Agent Guide

## 🎯 Project Overview

CropXcel is a Django-based early warning system for Cambodian farmers 🇰🇭, detecting waterlogging via Sentinel-1/2 satellite data 🛰️. The app transforms complex satellite signals into actionable farming insights with risk mapping 🗺️, hotspot detection 🔥, and AI crop recommendations 🌱.

## 🏗️ Architecture & Core Components

### 🐳 Multi-Service Docker Architecture

- **🌐 Web Service**: Django app (port 8001) with gunicorn serving
- **⚙️ Worker Service**: Celery background tasks for satellite analysis
- **🗄️ Database**: PostgreSQL for user data, field boundaries, analysis jobs
- **📦 Cache**: Redis for Celery broker and Django caching
- **💾 Media Storage**: Local filesystem (`./media/`) for generated files

### 🔄 Key Service Boundaries

```
🖥️ Frontend (Leaflet.js maps) → 🐍 Django Views → 📋 Celery Tasks → 🔬 Analysis Engine → 🛰️ Earth Engine API
                               ↓
                    📁 Media Files (overlays, plots, CSVs)
```

### ⚡ Critical Data Flow

1. 👤 User draws field boundary → `FieldAOI` model stores GeoJSON 📍
2. 🚀 Analysis request → `AnalysisJob` queued via Celery 📋
3. 🤖 Worker calls `analysis/engine.py` → fetches Sentinel-1 data from Google Earth Engine 🛰️
4. 🎨 `analysis/analysis.py` → generates risk overlay PNG + hotspots GeoJSON 🗺️
5. 📊 `analysis/insights.py` → creates temporal plots + insights CSV 📈
6. 💾 Results stored in `AnalysisJob.result` JSON field + media files 📁

## 🔧 Development Workflow

### 🐳 Docker Commands (Required)

```bash
# 🚀 Start all services (primary development command)
docker compose up -d --build

# 👀 View logs (essential for debugging)
docker compose logs -f web
docker compose logs -f worker

# 🐚 Shell access for debugging
docker compose exec web python manage.py shell
docker compose exec web bash

# 🔄 Reset database when models change
docker compose down -v && docker compose up -d --build
```

### 🛰️ Earth Engine Modes

- **🚀 Production**: Real Google Earth Engine API (requires service account)
- **🎭 Demo Mode**: `USE_DEMO_MODE=true` generates synthetic 11-band Sentinel-1 data
- **❌ Disabled**: `DISABLE_EARTH_ENGINE=true` disables satellite processing

### 📁 Media File Structure

```
media/
├── 🎨 overlays/     # Risk overlay PNGs for map display
├── 🔥 hotspots/     # GeoJSON files with problem area polygons
├── 📊 plots/        # Temporal analysis plots (matplotlib)
├── 💡 insights/     # CSV files with per-pass risk data
├── 📚 stacks/       # Intermediate satellite data files
└── 🗺️ aoi/          # Field boundary uploads
```

## 📁 Key Files & Patterns

### 🏢 Model Architecture (`app_core/models.py`)

- `FieldAOI`: Stores field boundaries as PostGIS geometry, links to users 🗺️
- `AnalysisJob`: Tracks async analysis status, stores results as JSON 📋
- `Profile`: User metadata with province/crop preferences 👤

### 🔬 Analysis Pipeline

- `analysis/engine.py`: Earth Engine integration, exports 11-band Sentinel-1 stacks 🛰️
- `analysis/analysis.py`: Risk calculation from satellite bands, PNG overlay generation 🎨
- `analysis/insights.py`: Temporal analysis, matplotlib plotting, CSV generation 📊
- `app_core/tasks.py`: Celery tasks orchestrating the analysis pipeline ⚙️

### ⚠️ Critical Pattern: Demo Mode Fallback

```python
# 🛰️ All Earth Engine calls must handle demo mode
if ee_available:
    # ✅ Real satellite data processing
    collection = ee.ImageCollection('COPERNICUS/S1_GRD')
else:
    # 🎭 Generate synthetic data with exact band structure
    return create_demo_stack_with_11_bands()
```

### 🌍 Environment Configuration

Docker Compose handles all environment variables:

- `DEBUG=True` for development 🧪
- `USE_DEMO_MODE=true` for development without Earth Engine 🎭
- `ALLOWED_HOSTS=*` for Docker networking 🐳
- `DATABASE_URL`, `REDIS_URL` for service connections 🔗

## 🐛 Common Issues & Solutions

### 🖼️ Empty Overlay PNGs (145 bytes)

- **❌ Cause**: Demo mode band count mismatch (6 vs 11 bands expected)
- **✅ Fix**: Ensure `export_stack_from_geom()` creates exactly 11 bands in demo mode
- **🔍 Debug**: Check band structure with `rasterio.open(stack_path).count`

### ⚙️ Celery Task Failures

- **🔍 Debug**: `docker compose logs worker` for error details
- **❌ Common**: Missing geospatial dependencies, Earth Engine auth issues
- **🔧 Fix**: Rebuild container after dependency changes

### 🗄️ Database Migrations in Docker

- **📝 Pattern**: Always run migrations after model changes

```bash
docker compose exec web python manage.py makemigrations
docker compose exec web python manage.py migrate
```

### 🌐 Frontend-Backend Communication

- Views return JSON with media URLs: `{"overlay_png_url": "/media/overlays/risk_xyz.png"}` 📡
- Leaflet.js consumes these URLs for map overlay display 🗺️
- All file paths use Django's `MEDIA_URL` configuration ⚙️

## 📊 Testing & Validation

### 🧪 Analysis Job Testing

```python
# 🐚 In Django shell
from app_core.models import AnalysisJob, FieldAOI
from app_core.tasks import run_waterlogging_analysis

field = FieldAOI.objects.first()
job = AnalysisJob.objects.create(field=field, status="queued")
run_waterlogging_analysis.delay(job.id)
```

### 🎭 Demo Mode Validation

Essential for development without Earth Engine access. Always verify:

1. 📊 Stack files have 11 bands (S1_VV_CURR through S1_VH_STD)
2. 🎯 Risk calculation produces non-zero values
3. 🖼️ Overlay PNGs contain visual data (>1000 bytes)

## 🔍 Project-Specific Conventions

### 📝 File Naming Patterns

- 🎨 Risk overlays: `risk_{uuid}.png`
- 🔥 Hotspots: `hotspots_{uuid}.geojson`
- 📊 Plots: `plot_{field_id}_{timestamp}.png`
- 📚 Stacks: `stack_field_{field_id}_{timestamp}.tif`

### 🇰🇭 Cambodian Context Integration

- 🗺️ Province choices hard-coded for Cambodia administrative divisions
- 🌾 Rice variety recommendations (Sen Kra'ob, Phka Rumduol)
- 📅 Khmer calendar integration in UI (commented references)

### ⚠️ Error Handling Pattern

```python
# 🛡️ Graceful degradation for missing dependencies
try:
    from analysis.analysis import run_analysis_from_notebook
    ANALYSIS_AVAILABLE = True
except Exception as e:
    print(f"[TASKS] Analysis functions not available: {e}")
    ANALYSIS_AVAILABLE = False
```

This pattern allows the app to start even with missing geospatial libraries 📦, enabling incremental development and deployment debugging 🐛.
