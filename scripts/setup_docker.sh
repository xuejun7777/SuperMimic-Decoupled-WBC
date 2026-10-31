#!/usr/bin/env bash
# Pull the pre-built Isaac Lab image and open a shell with this repository
# mounted into the container.
#
# Usage:
#   bash scripts/setup_docker.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_REPO="/workspace/IsaacLab-Decoupled-WBC"
IMAGE="chrisyrniu/htd-wbc:isaaclab-2.2.0"
OMNI_DIR="/opt/conda/envs/isaac_wbc/lib/python3.11/site-packages/omni"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/htd-wbc/isaac-sim-5.0"
VULKAN_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker is not installed or is not available on PATH." >&2
    exit 1
fi

if ! command -v nvidia-container-cli >/dev/null 2>&1; then
    echo "[ERROR] NVIDIA Container Toolkit is not installed." >&2
    echo "        See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" >&2
    exit 1
fi

mkdir -p "$CACHE_DIR"/{cache,data,logs}

echo "[setup] Pulling ${IMAGE}..."
docker pull "$IMAGE"

echo "[setup] Mounting ${REPO_DIR} at ${CONTAINER_REPO}"
echo "[setup] Using Isaac Sim cache at ${CACHE_DIR}"
echo "[setup] Opening the container shell..."
# Select only the image's NVIDIA manifests. NVIDIA Container Toolkit also
# injects host manifests, and loading both can enumerate the same GPU twice.
exec docker run --gpus all --rm -it \
    -e "PYTHONPATH=${CONTAINER_REPO}" \
    -e "VK_DRIVER_FILES=${VULKAN_ICD}" \
    -e "VK_ICD_FILENAMES=${VULKAN_ICD}" \
    -e "__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json" \
    -v "${REPO_DIR}:${CONTAINER_REPO}" \
    -v "${CACHE_DIR}/cache:${OMNI_DIR}/cache" \
    -v "${CACHE_DIR}/data:${OMNI_DIR}/data" \
    -v "${CACHE_DIR}/logs:${OMNI_DIR}/logs" \
    -w "$CONTAINER_REPO" \
    "$IMAGE" bash
