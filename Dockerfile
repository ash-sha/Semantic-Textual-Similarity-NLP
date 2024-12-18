# Use an appropriate base image for your project
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy only the requirements file first to take advantage of Docker cache
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Copy the rest of your application files into the container
COPY . /app

# Set environment variables for model, result, log, and data directories
ENV MODEL_DIR=/app/models
ENV RESULT_PATH=/app/results
ENV LOG_PATH=/app/logs
ENV DATA_DIR=/app/data/clinic_c.csv

# Expose a port if necessary
EXPOSE 8000

# Run the application (for example, FastAPI)
ENTRYPOINT ["python","src/server.py"]
