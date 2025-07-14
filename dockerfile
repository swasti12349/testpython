# Dockerfile
FROM python:3.10

# 1. Install nginx & supervisor
RUN apt-get update \
 && apt-get install -y nginx supervisor \
 && rm -rf /var/lib/apt/lists/*

# 2. Copy configs
COPY nginx.conf       /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.confd

# 3. Copy app & install dependencies
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 4. Expose port 80
EXPOSE 80

# 5. Launch both processes
# CMD ["supervisord", "-c", "supervisord.conf"]
CMD ["./start.sh"]



# Uncomment the following lines if you want to use Nginx as a reverse proxy
# FROM python:3.10

# # Set working directory
# WORKDIR /app

# # Install Nginx and other dependencies
# RUN apt-get update && \
#     apt-get install -y nginx && \
#     apt-get clean

# # Copy requirements and install Python packages
# COPY requirements.txt requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy app and nginx config
# COPY . .
# COPY nginx.conf /etc/nginx/nginx.conf

# # Expose Nginx port
# EXPOSE 80

# # Start both Nginx and Streamlit
# CMD service nginx 


# Dockerfile for Streamlit app with Nginx and Supervisor
# FROM python:3.10

# WORKDIR /app

# # Install nginx and supervisor
# RUN apt-get update && \
#     apt-get install -y nginx supervisor && \
#     apt-get clean

# # Install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy your app code
# COPY . .

# # Copy nginx config
# COPY nginx.conf /etc/nginx/nginx.conf

# # Copy supervisor config
# COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# # Expose port 80 for Nginx
# EXPOSE 80

# # Start supervisor
# CMD ["/usr/bin/supervisord"]
