#!/bin/sh

# Start Ollama service
ollama serve &

# Run the specified model
if [ -z "$OLLAMA_MODEL" ]; then
  echo "No model specified. Exiting."
  exit 1
fi

# Define the model file path
MODEL_FILE_PATH="/models/$OLLAMA_MODEL/$OLLAMA_MODEL.modelfile"

# Print out the values of the OLLAMA_MODEL and MODEL_FILE_PATH variables
echo "OLLAMA_MODEL: $OLLAMA_MODEL"
echo "MODEL_FILE_PATH: $MODEL_FILE_PATH"

# List the contents of the /models directory
echo "Contents of /models:"
ls /models

# Create and run the model
ollama create $OLLAMA_MODEL -f $MODEL_FILE_PATH
ollama run $OLLAMA_MODEL

# Keep the container running
tail -f /dev/null