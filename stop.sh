#!/usr/bin/env bash
set -euo pipefail

# Stop the SixteenGigsOfVibes container (idempotent).
#
# Leaves the stopped container in place for post-mortem (podman logs / inspect);
# re-running ./start.sh removes it and recreates it with fresh config.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${SCRIPT_DIR}/.env"
fi

CONTAINER_NAME="${CONTAINER_NAME:-SixteenGigsOfVibes-llama}"

if ! podman ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} not found; nothing to stop."
  exit 0
fi

if podman ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping ${CONTAINER_NAME} (60 s grace period) ..."
  podman stop -t 60 "${CONTAINER_NAME}" >/dev/null
  echo "Stopped ${CONTAINER_NAME} (kept for post-mortem: podman logs ${CONTAINER_NAME})"
else
  echo "Container ${CONTAINER_NAME} is already stopped."
fi
