# Use the official Python 3.8 image from the Docker Hub as the base image
FROM python:3.8.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install the dependencies specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the rest of the application code into the container at /app
COPY . /app

# Expose the port you want your application to run on (if needed)
EXPOSE 5002

# Define environment variable (if needed)
ENV PYTHONUNBUFFERED=1

# Set the default command to run when starting the container
CMD ["streamlit", "run", "gui.py"]