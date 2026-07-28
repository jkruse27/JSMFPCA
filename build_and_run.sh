#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e 

echo "Building the Docker container..."
# Build the image, passing the host's user and group IDs to match permissions[cite: 3]
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t nongaussian -f Dockerfile .

echo "Starting the container..."
# Run the container interactively, mounting the current directory to /workspace[cite: 3]
docker run -it --rm -v "$(pwd)":/workspace nongaussian