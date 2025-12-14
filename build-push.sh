#!/bin/bash
set -e

# ============================================
# Build and Push GPU Base Image
# ============================================
# Usage: ./build-push.sh [username] [tag]
# Example: ./build-push.sh myuser latest
# ============================================

USERNAME="${1:-yourusername}"
TAG="${2:-latest}"
IMAGE_NAME="gpu-base"

echo "Building ${USERNAME}/${IMAGE_NAME}:${TAG}"
echo "============================================"

# Build
docker build -t ${USERNAME}/${IMAGE_NAME}:${TAG} .

# Tag as latest if not already
if [ "$TAG" != "latest" ]; then
    docker tag ${USERNAME}/${IMAGE_NAME}:${TAG} ${USERNAME}/${IMAGE_NAME}:latest
fi

echo ""
echo "Build complete!"
echo ""
echo "To push to Docker Hub:"
echo "  docker login"
echo "  docker push ${USERNAME}/${IMAGE_NAME}:${TAG}"
echo ""
echo "To run:"
echo "  docker run -it --gpus all ${USERNAME}/${IMAGE_NAME}:${TAG}"
