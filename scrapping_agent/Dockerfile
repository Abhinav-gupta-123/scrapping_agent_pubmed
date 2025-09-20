# Use official Python runtime base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Run FastAPI with Uvicorn, 10 workers, on host 0.0.0.0 port 8000
CMD ["uvicorn", "scrapper_main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "10"]
