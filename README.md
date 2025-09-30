# Waterlogging Monitoring (CropXcel)

## Overview
CropXcel is a Django-based web application for monitoring waterlogging risk in agricultural fields using Sentinel-1 satellite data. It provides farmers and agronomists with actionable insights, risk alerts, and visual analytics to support irrigation and drainage decisions.

## Features
- **Interactive Map:** Visualize field boundaries, overlays, and hotspots using Leaflet.js.
- **Per-pass Insights Table:** See waterlogging risk status (Healthy, Watch, Alert) for each satellite pass, with recommended actions.
- **Analysis Scale:** View total area breakdown by risk level and animated donut chart.
- **Technical Details:** Switch to advanced indicators for agronomists.
- **Plot History:** Analyze 4-month waterlogging risk trends.
- **Crop Recommendation:** Enhanced button styling for key actions.

## Technologies
- **Backend:** Django, Python, Pandas
- **Frontend:** JavaScript, Leaflet.js, HTML/CSS
- **Data:** Sentinel-1 satellite, CSV, GeoJSON

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/KosolCHOU/Waterlogging-Monitoring.git
   cd Waterlogging-Monitoring/CropXcel's app
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run migrations:**
   ```bash
   python manage.py migrate
   ```
4. **Start the server:**
   ```bash
   python manage.py runserver
   ```
5. **Access the dashboard:**
   Open [http://127.0.0.1:8000/dashboard/](http://127.0.0.1:8000/dashboard/)

## Folder Structure
- `analysis/` - Waterlogging risk engine and insights calculation
- `app_core/` - Django app logic, models, views, serializers
- `cropxcel_project/` - Django project settings
- `media/` - Generated plots, overlays, CSVs, GeoJSON
- `static/` - CSS, JS, images
- `templates/` - HTML templates

## Contributing
Pull requests and suggestions are welcome! Please open an issue for major changes.

## License
MIT License

---
For more details, see the code and comments in each module.
