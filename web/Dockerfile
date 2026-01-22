FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip

# PRO TIP: Install heavy dependencies separately to cache them layer-wise
# and avoid timeouts on the massive single install.
RUN pip install torch transformers rdkit --no-cache-dir --default-timeout=1000

# Install remaining
RUN pip install -r requirements.txt --no-cache-dir # Force update

# Copy project
COPY . /app/

# Collect static files (optional, can be done in entrypoint)
# RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Run gunicorn
CMD ["gunicorn", "ProjectAegis.wsgi:application", "--bind", "0.0.0.0:8000"]
