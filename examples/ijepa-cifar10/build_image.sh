#!/usr/bin/env bash
# Build + push the AutoJEPA Phase-2 runtime image to GHCR.
#
# Prereq: docker login to ghcr.io with a token that has write:packages
# scope. The author's setup is `gh auth login --scopes write:packages`
# followed by `gh auth token | docker login ghcr.io -u <user>
# --password-stdin`. Verify with `cat ~/.docker/config.json`.
#
# See ADR-016 for the image's role; the resulting tag is referenced
# from examples/ijepa-cifar10/config.yaml::target.basilica.image.

set -euo pipefail

IMAGE_REGISTRY="${IMAGE_REGISTRY:-ghcr.io/epappas}"
IMAGE_NAME="${IMAGE_NAME:-autojepa-runtime}"
IMAGE_TAG="${IMAGE_TAG:-phase2}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL="${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building ${FULL} from ${DIR}/Dockerfile ..."
docker build -t "${FULL}" -f "${DIR}/Dockerfile" "${DIR}"

echo "Pushing ${FULL} ..."
docker push "${FULL}"

echo "OK: pushed ${FULL}"
echo "Wire it into config.yaml: target.basilica.image: ${FULL}"
