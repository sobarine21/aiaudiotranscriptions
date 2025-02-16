# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install ffmpeg (required by pydub)
RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run Streamlit when the container starts
CMD ["streamlit", "run", "app.py"]
