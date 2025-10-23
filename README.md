# Waterlogging Monitoring (CropXcel)

## Overview
CropXcel provides **early warning systems for Cambodian farmers** to combat waterlogging - a silent threat that can destroy weeks of hard work in just days. Using Sentinel-1 and Sentinel-2 satellite data, we give farmers "eyes in the sky" to detect hidden water stress before visible damage appears.

**The Problem:** Heavy rains flood fields, rice turns yellow, and by the time farmers see the damage, yield is already lost. Many farmers lack guidance on when and how to act.

**Our Solution:** Transform complex satellite signals into simple, actionable insights delivered every few days. CropXcel turns data into better harvests by providing early detection, clear guidance, and smart recommendations for irrigation, drainage, and crop selection.

## Features

### 🗺️ **Interactive Risk Mapping**
- **Real-time field visualization** with color-coded risk levels (Healthy/Watch/Alert)
- **Hotspot detection** showing specific problem areas in your fields
- **Overlay analysis** combining satellite data with field boundaries
- **Click-anywhere probing** for instant waterlogging risk assessment

### 📊 **Smart Analytics Dashboard**
- **Per-pass Insights Table:** Track each satellite pass like a field diary with risk status and recommended actions
- **Weather Integration:** Current conditions + 72-hour rain forecasts + 7-day planning table
- **Analysis Scale:** Visual breakdown of healthy vs. risky field areas with animated donut charts
- **Trend Monitoring:** 4-month risk history for smarter seasonal planning

### 💧 **Water Advice Cards**
- **Actionable recommendations:** Drain immediately, reduce irrigation, or maintain routine
- **Timing guidance:** Know when to act before damage becomes visible
- **Severity indicators:** Understand urgency levels for different field conditions

### 🌱 **Crop Compass (AI Recommendation)**
- **Smart crop matching:** Analyze soil nutrients (N, P, K, pH) and weather patterns
- **Top 3 suggestions:** Data-driven variety recommendations with confidence scores
- **Risk reduction:** Choose crops that fit your specific soil and climate conditions
- **Farmer-friendly results:** Clear explanations without technical jargon

### 🔧 **Technical Features**
- **Dual-mode interface:** Simple farmer view + detailed technical indicators for agronomists
- **Multi-satellite integration:** Sentinel-1 (radar) + Sentinel-2 (optical) for cloud-penetrating analysis
- **Real-time processing:** Automatic updates when new satellite data becomes available
- **Mobile-responsive:** Access insights from any device, anywhere in the field

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
2. **Activate Environment**
   ```bash
   source '~/Waterlogging-Monitoring/.venv/bin/activate'
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
