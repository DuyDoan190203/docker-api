# Use official Python image (slim for small size)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY /app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the code
COPY . .

# Expose port
EXPOSE 8080

# Run the app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]