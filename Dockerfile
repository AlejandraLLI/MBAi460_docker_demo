# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container's /app folder
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run train_model.py when the container launches
CMD ["python", "main.py"]