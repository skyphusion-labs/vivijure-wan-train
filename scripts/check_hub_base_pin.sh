#!/usr/bin/env bash
# Verify the .runpod/Dockerfile base pin is real and self-consistent.
#
# The Hub listing builds .runpod/Dockerfile, which is a thin FROM of the published GHCR image
# (see that file for why). A stale or wrong pin makes the Hub listing build a different artifact
# than production runs, or fail outright. This check resolves the pin against GHCR anonymously
# (the same reach the Hub builder has) and asserts:
#   1. an anonymous pull token is issuable for the repo (the package is public),
#   2. the pinned TAG resolves on GHCR, and
#   3. the pinned @sha256 DIGEST is exactly what that tag resolves to.
#
# Usage: scripts/check_hub_base_pin.sh [path/to/Dockerfile]
set -euo pipefail

DOCKERFILE="${1:-.runpod/Dockerfile}"
[ -f "${DOCKERFILE}" ] || { echo "::error::${DOCKERFILE} not found"; exit 1; }

FROM_LINE="$(grep -m1 -E '^FROM ' "${DOCKERFILE}" || true)"
[ -n "${FROM_LINE}" ] || { echo "::error::no FROM line in ${DOCKERFILE}"; exit 1; }

REF="${FROM_LINE#FROM }"
case "${REF}" in
  ghcr.io/*:*@sha256:*) : ;;
  *) echo "::error::base must be pinned as ghcr.io/<repo>:<tag>@sha256:<digest>; got: ${REF}"; exit 1 ;;
esac

REPO="${REF%%:*}"; REPO="${REPO#ghcr.io/}"
REST="${REF#ghcr.io/${REPO}:}"
TAG="${REST%%@*}"
DIGEST="${REST#*@}"

echo "repo=${REPO} tag=${TAG} digest=${DIGEST}"

TOKEN_JSON="$(curl -fsS "https://ghcr.io/token?scope=repository:${REPO}:pull&service=ghcr.io" || true)"
TOKEN="$(printf %s "${TOKEN_JSON}" | python3 -c 'import json,sys
raw = sys.stdin.read().strip()
print(json.loads(raw).get("token", "") if raw else "")' 2>/dev/null || true)"
[ -n "${TOKEN}" ] || { echo "::error::no anonymous pull token for ${REPO} (package private, or repo does not exist); the Hub builder could not pull it either"; exit 1; }

ACCEPT='application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json'

resolve() {  # $1 = tag or digest reference -> prints docker-content-digest
  curl -fsSI -H "Authorization: Bearer ${TOKEN}" -H "Accept: ${ACCEPT}" \
    "https://ghcr.io/v2/${REPO}/manifests/$1" \
    | tr -d '\r' | awk 'tolower($1)=="docker-content-digest:"{print $2}'
}

TAG_DIGEST="$(resolve "${TAG}" || true)"
[ -n "${TAG_DIGEST}" ] || { echo "::error::tag ${REPO}:${TAG} does not resolve anonymously on GHCR (tag deleted, or package not public)"; exit 1; }

if [ "${TAG_DIGEST}" != "${DIGEST}" ]; then
  echo "::error::pin drift: ${REPO}:${TAG} now resolves to ${TAG_DIGEST}, but ${DOCKERFILE} pins ${DIGEST}"
  exit 1
fi

PINNED_DIGEST="$(resolve "${DIGEST}" || true)"
[ "${PINNED_DIGEST}" = "${DIGEST}" ] || { echo "::error::digest ${DIGEST} does not resolve anonymously on GHCR"; exit 1; }

echo "OK: ${REPO}:${TAG} == ${DIGEST} and is anonymously pullable (Hub can build this)."
