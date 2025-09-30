# 🚀 Deployment Guide for CropXcel

This guide covers multiple deployment options for your CropXcel application.

## 📋 Prerequisites

- Python 3.11+
- Git
- A hosting platform account (Heroku, Railway, DigitalOcean, etc.)

## 🔧 Pre-deployment Setup

1. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your production values
   ```

2. **Update ALLOWED_HOSTS**
   Add your domain to the `.env` file:
   ```
   ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
   ```

## 🌐 Deployment Options

### Option 1: Render.com (Easiest - Recommended)

1. **Go to Render.com**
   - Visit [render.com](https://render.com)
   - Sign up with your GitHub account
   - Click "Connect GitHub"

2. **Create Web Service**
   - Click "New +" → "Web Service"
   - Select your `Waterlogging-Monitoring` repository
   - Use these settings:

   **Basic Configuration:**
   - **Name:** `cropxcel-demo`
   - **Root Directory:** `CropXcel's app`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements-minimal.txt && python manage.py collectstatic --noinput && python manage.py migrate`
   - **Start Command:** `gunicorn cropxcel_project.wsgi:application --host 0.0.0.0 --port $PORT`

   **Environment Variables:**
   ```
   DEBUG=False
   SECRET_KEY=TR0x8TXZ7hLuE1AoVZG6JC2asNmXsPwzvT2N5XgZf8DhbhdzEjtiOGh_E3Fach7jWyM
   DJANGO_SETTINGS_MODULE=cropxcel_project.settings_production
   ALLOWED_HOSTS=cropxcel-demo.onrender.com,localhost,127.0.0.1
   DATABASE_URL=sqlite:///./db.sqlite3
   ```

3. **Deploy**
   - Click "Create Web Service"
   - Render will automatically deploy from your GitHub repo
   - Your app will be live at: `https://cropxcel-demo.onrender.com`

4. **Important Notes:**
   - Make sure **NOT** to select "Docker" - choose "Python 3" environment
   - The build will take 5-10 minutes for first deployment
   - Render free tier sleeps after 15 minutes of inactivity

### Option 2: Heroku (Requires payment verification)

1. **Install Heroku CLI**
   ```bash
   # Install from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Heroku App**
   ```bash
   heroku create your-cropxcel-app
   heroku addons:create heroku-postgresql:mini
   heroku addons:create heroku-redis:mini
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set DEBUG=False
   heroku config:set SECRET_KEY="your-secret-key-here"
   heroku config:set DJANGO_SETTINGS_MODULE=cropxcel_project.settings_production
   ```

4. **Deploy**
   ```bash
   git push heroku main
   heroku run python manage.py migrate
   heroku run python manage.py createsuperuser
   ```

5. **Open your app**
   ```bash
   heroku open
   ```

### Option 2: Railway (Modern alternative)

1. **Connect to Railway**
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub repository
   - Railway will auto-detect Django

2. **Add Environment Variables**
   ```
   DEBUG=False
   SECRET_KEY=your-secret-key-here
   DJANGO_SETTINGS_MODULE=cropxcel_project.settings_production
   ```

3. **Add PostgreSQL & Redis**
   - Click "Add Service" → PostgreSQL
   - Click "Add Service" → Redis

4. **Deploy**
   - Railway automatically deploys on git push

### Option 3: Docker (Any VPS)

1. **Build and Run**
   ```bash
   docker-compose up -d
   ```

2. **Run Migrations**
   ```bash
   docker-compose exec web python manage.py migrate
   docker-compose exec web python manage.py createsuperuser
   ```

### Option 4: DigitalOcean App Platform

1. **Create App**
   - Go to DigitalOcean App Platform
   - Connect your GitHub repository

2. **Configure**
   - App Spec: Use the provided settings
   - Add PostgreSQL & Redis databases

3. **Environment Variables**
   ```
   DEBUG=False
   SECRET_KEY=your-secret-key-here
   DATABASE_URL=auto-filled
   REDIS_URL=auto-filled
   ```

## 🗃️ Database Setup

For production, you'll need:

1. **PostgreSQL** (recommended)
2. **Redis** (for Celery tasks)

Most cloud providers offer these as managed services.

## 🔐 Security Checklist

- [ ] Set `DEBUG=False`
- [ ] Use a strong `SECRET_KEY`
- [ ] Configure `ALLOWED_HOSTS`
- [ ] Set up SSL/HTTPS
- [ ] Use environment variables for secrets
- [ ] Configure database backups

## 📊 Monitoring & Maintenance

1. **Set up logging**
2. **Monitor disk space** (for media files)
3. **Regular database backups**
4. **Update dependencies** regularly

## 🆘 Troubleshooting

### Common Issues:

1. **Static files not loading**
   ```bash
   python manage.py collectstatic
   ```

2. **Database connection errors**
   - Check DATABASE_URL
   - Ensure database is running

3. **Import errors**
   - Check all dependencies in requirements.txt
   - Verify Python version compatibility

## 🌍 Custom Domain

1. **Add your domain to ALLOWED_HOSTS**
2. **Configure DNS** to point to your hosting provider
3. **Set up SSL certificate** (usually automatic with modern providers)

## 📞 Support

If you encounter issues:
1. Check the logs: `heroku logs --tail` (for Heroku)
2. Verify environment variables
3. Ensure all migrations are applied

---

**Your CropXcel app will be accessible at:** `https://your-app-name.herokuapp.com` (or your chosen domain)

Happy farming! 🌾
