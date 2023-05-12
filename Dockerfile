# Use the official Python 3.9 image as the base image
FROM python:3.9

# Expose the port
EXPOSE 7860

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the PYNGROK_CONFIG environment variable
ENV PYNGROK_CONFIG /tmp/pyngrok.yml

# Set the NGROK_PATH environment variable to a writable location
ENV NGROK_PATH /tmp/ngrok

# Copy requirements.txt into the container
COPY requirements.txt .

# Upgrade pip and install the required packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install sudo and create the necessary directories before copying the files
RUN apt-get update && \
    apt-get install -y sudo && \
    mkdir -p /code/image

# Creates a non-root user with an explicit UID and adds permission to access the /code folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && \
    usermod -aG sudo appuser && \
    usermod -aG root appuser && \
    chown -R appuser:appuser /code

# Create the pyngrok bin directory and set the ownership and permissions for appuser
RUN mkdir -p /usr/local/lib/python3.9/site-packages/pyngrok/bin && \
    chown -R appuser:appuser /usr/local/lib/python3.9/site-packages/pyngrok/bin && \
    chmod -R 777 /usr/local/lib/python3.9/site-packages/pyngrok/bin

RUN mkdir -p /.ngrok2 && \
    chown -R appuser:appuser /.ngrok2 && \
    chmod -R 777 /.ngrok2

RUN apt-get update && \
    apt-get install -y curl

# Set the working directory and copy the files
WORKDIR /code

# Set the ownership and permissions for the /code directory and its contents
RUN chown -R appuser:appuser /code && \
    chmod -R 777 /code

RUN mkdir -p /code/image && \
    chown -R appuser:appuser /code/image && \
    chmod -R 777 /code/image

COPY . /code


# Copy the pyngrok.yml configuration file
COPY pyngrok.yml /tmp/pyngrok.yml

# Set the TRANSFORMERS_CACHE environment variable to a cache directory inside /tmp
ENV TRANSFORMERS_CACHE /tmp/transformers_cache
ENV TORCH_HOME /tmp/torch_cache

USER appuser

# Start the application using pyngrok
# CMD python main.py
# Get the public IP address and display it
RUN curl -s https://api.ipify.org | xargs echo "Public IP:"

# Start the Uvicorn server
# ENTRYPOINT ["python", "main.py"]
# CMD ["sh", "-c", "python main.py & sleep infinity"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]