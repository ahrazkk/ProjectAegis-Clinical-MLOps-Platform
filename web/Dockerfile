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

# Install CPU-only PyTorch first to avoid downloading CUDA wheels in Cloud Run.
RUN pip install --no-cache-dir --default-timeout=1000 --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu

# Install remaining dependencies.
RUN pip install -r requirements.txt --no-cache-dir --default-timeout=1000

# Copy project
COPY . /app/

# Collect static files (optional, can be done in entrypoint)
# RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Run with gunicorn.
# PubMedBERT model initialization can exceed the default 30s worker timeout on cold starts.
CMD ["gunicorn", "ProjectAegis.wsgi:application", "--bind", "0.0.0.0:8000", "--timeout", "180", "--graceful-timeout", "180"]
