# Use the official Ollama Docker image as the base image
FROM ollama/ollama:latest

# Expose the necessary port
EXPOSE 11434

# Copy the entrypoint script into the container
COPY entrypoint.sh /entrypoint.sh

# Create a directory for the models, where the volume will be mounted
RUN mkdir /models

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Define the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
