#!/usr/bin/env bash
set -euo pipefail

# SixteenGigsOfVibes — local llama.cpp (GGUF) server for 16 GB RAM/VRAM machines.
#
# Order of operations: preflight -> backend detection -> image pull -> model
# download (resumable, SHA256SUMS-verified) -> container start -> wait for
# /health -> endpoint info. The script is idempotent: re-running skips the
# download and reports "already running".
#
# Variable precedence (highest first):
#   1. variables exported in the shell
#   2. entries in ./.env
#   3. inline defaults below / presets defined for MODEL

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

# ---- model preset ------------------------------------------------------------
# Each preset is a complete set: HF repo, GGUF file, model alias (the name
# shown in /v1/models), an optional vision projector and a default context.
MODEL="${MODEL:-qwen38}"
case "${MODEL}" in
  qwen38)
    PRESET_REPO="empero-ai/Qwen3.8-9B-Distill-GGUF"
    PRESET_FILE="Qwen3.8-9B-Q4_K_M.gguf"
    PRESET_ALIAS="qwen3.8-9b-distill-q4_k_m"
    PRESET_MMPROJ=""                     # repo has no vision projector
    PRESET_CTX=32768
    PRESET_NEED_GIB=8
    ;;
  ornith)
    PRESET_REPO="bartowski/Ornith-1.5-9B-GGUF"
    PRESET_FILE="Ornith-1.5-9B-Q4_K_M.gguf"
    PRESET_ALIAS="ornith-1.5-9b-q4_k_m"
    PRESET_MMPROJ="mmproj-Ornith-1.5-9B-f16.gguf"
    PRESET_CTX=32768
    PRESET_NEED_GIB=8
    ;;
  bonsai27)
    PRESET_REPO="prism-ml/Ternary-Bonsai-2-27B-gguf"
    PRESET_FILE="Ternary-Bonsai-2-27B-Q2_0.gguf"
    PRESET_ALIAS="bonsai2-27b-q2_0"
    PRESET_MMPROJ="Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
    PRESET_CTX=16384
    PRESET_NEED_GIB=10
    PRESET_KV="q8_0"                     # 27B: KV in q8_0 fits in 16 GB
    ;;
  *)
    echo "ERROR: unknown MODEL='${MODEL}' (allowed: qwen38 | ornith | bonsai27)"
    exit 1
    ;;
esac

# Only when the variable is not set in .env/shell — the loader skips empty
# values in .env, so "defined but empty" is respected here as well.
HF_REPO="${HF_REPO-${PRESET_REPO}}"
HF_FILE="${HF_FILE-${PRESET_FILE}}"
MODEL_ALIAS="${MODEL_ALIAS-${PRESET_ALIAS}}"
MMPROJ="${MMPROJ-${PRESET_MMPROJ}}"
CTX_SIZE="${CTX_SIZE:-${PRESET_CTX}}"
KV_CACHE_TYPE="${KV_CACHE_TYPE-${PRESET_KV:-}}"

HF_REVISION="${HF_REVISION:-}"            # empty = main; provide a SHA for reproducibility
VISION="${VISION:-}"                      # non-empty + MMPROJ -> --mmproj
PARALLEL="${PARALLEL:-1}"                 # number of slots; context is split between them
BACKEND="${BACKEND:-auto}"                # auto | cuda | rocm | vulkan | cpu
CONTAINER_NAME="${CONTAINER_NAME:-SixteenGigsOfVibes-llama}"
PORT="${PORT:-8080}"                      # host port; container listens on 8080
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/.SixteenGigsOfVibes.log}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
PODMAN_EXTRA_ARGS="${PODMAN_EXTRA_ARGS:-}"

resolve_path() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    "~/"*) printf '%s/%s' "${HOME}" "${1#\~/}" ;;
    *) printf '%s/%s' "${SCRIPT_DIR}" "${1#./}" ;;
  esac
}
MODEL_DIR="$(resolve_path "${MODEL_DIR}")"
LOG_FILE="$(resolve_path "${LOG_FILE}")"

# ---- preflight ---------------------------------------------------------------
command -v podman >/dev/null 2>&1 || { echo "podman is not on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is not on PATH"; exit 1; }
[[ "${CTX_SIZE}" =~ ^[0-9]+$ ]] || { echo "ERROR: CTX_SIZE must be an integer (got '${CTX_SIZE}')"; exit 1; }
[[ "${PARALLEL}" =~ ^[0-9]+$ ]] || { echo "ERROR: PARALLEL must be an integer (got '${PARALLEL}')"; exit 1; }

# ---- backend: auto -> cuda > rocm > vulkan > cpu -----------------------------
if [[ "${BACKEND}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    BACKEND="cuda"
  elif [[ -e /dev/kfd ]]; then
    BACKEND="rocm"
  elif [[ -d /dev/dri ]]; then
    BACKEND="vulkan"
  else
    BACKEND="cpu"
  fi
fi

DEV_ARGS=()
GPU_LAYERS=()
case "${BACKEND}" in
  cuda)
    # Requires nvidia-container-toolkit (CDI) on the host — see README.
    DEV_ARGS=(--gpus all --ulimit memlock=-1:-1 --ipc host)
    GPU_LAYERS=(--n-gpu-layers 99)
    ;;
  rocm)
    DEV_ARGS=(--device /dev/kfd --device /dev/dri --group-add keep-groups --ipc host --ulimit memlock=-1:-1)
    GPU_LAYERS=(--n-gpu-layers 99)
    if ! id -nG | grep -qw render; then
      echo "warning: current user is not in the 'render' group; rootless /dev/kfd access may fail"
    fi
    ;;
  vulkan)
    DEV_ARGS=(--device /dev/dri --group-add keep-groups --ipc host)
    GPU_LAYERS=(--n-gpu-layers 99)
    ;;
  cpu)
    DEV_ARGS=()
    ;;
  *)
    echo "ERROR: unknown BACKEND='${BACKEND}' (allowed: auto|cuda|rocm|vulkan|cpu)"
    exit 1
    ;;
esac

if [[ -z "${IMAGE:-}" ]]; then
  case "${BACKEND}" in
    cpu) IMAGE="ghcr.io/ggml-org/llama.cpp:server" ;;
    *)   IMAGE="ghcr.io/ggml-org/llama.cpp:server-${BACKEND}" ;;
  esac
fi

echo "Model:   ${MODEL} (${HF_REPO}/${HF_FILE})"
echo "Backend: ${BACKEND}   Image: ${IMAGE}"
echo "Context: ${CTX_SIZE} tok   Slots: ${PARALLEL}   Vision: ${VISION:-off}"

# ---- image -------------------------------------------------------------------
if podman image exists "${IMAGE}"; then
  echo "Image ${IMAGE} is cached; skipping pull."
else
  echo "Pulling image ${IMAGE} ..."
  podman pull "${IMAGE}"
fi

# ---- model (host-side, resumable, SHA256SUMS-verified) -----------------------
MODEL_MARKER="${MODEL_DIR}/.download-complete"
VISION_FLAG=0
[[ -n "${VISION}" ]] && VISION_FLAG=1
MODEL_SPEC="${MODEL}|${HF_REPO}/${HF_FILE}@${HF_REVISION:-main}|vision=${VISION_FLAG}"

download_file() {
  local repo="$1" file="$2"
  local rev_args=()
  [[ -n "${HF_REVISION}" ]] && rev_args=(--revision "${HF_REVISION}")
  local attempt=1 rc=0
  while true; do
    if command -v hf >/dev/null 2>&1; then
      if hf download "${repo}" "${file}" "${rev_args[@]}" --local-dir "${MODEL_DIR}"; then
        return 0
      else
        rc=$?    # in the else branch $? is the exit code of the failed hf download
      fi
    elif python3 -c "import huggingface_hub" 2>/dev/null; then
      if HF_REPO_ENV="${repo}" HF_FILE_ENV="${file}" HF_REVISION_ENV="${HF_REVISION}" \
         MODEL_DIR_ENV="${MODEL_DIR}" python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id=os.environ["HF_REPO_ENV"],
    filename=os.environ["HF_FILE_ENV"],
    revision=os.environ.get("HF_REVISION_ENV") or None,
    local_dir=os.environ["MODEL_DIR_ENV"],
)
PY
      then
        return 0
      else
        rc=$?    # same — exit code of the failed hf_hub_download
      fi
    else
      echo "ERROR: neither 'hf' nor 'huggingface_hub' is available on the host."
      echo "       Install with:  pip install 'huggingface_hub[cli]'"
      echo "       Or download the file manually to ${MODEL_DIR}"
      return 127
    fi
    if (( attempt >= 3 )); then
      echo "ERROR: download of ${file} failed after ${attempt} attempts."
      return 1
    fi
    attempt=$((attempt + 1))
    echo "Download interrupted (exit ${rc}); retrying in 10 s ($((4 - attempt)) left) ..."
    sleep 10
  done
}

fetch_sums() {
  local url="https://huggingface.co/${HF_REPO}/raw/${HF_REVISION:-main}/SHA256SUMS"
  local hdr=()
  [[ -n "${HF_TOKEN:-}" ]] && hdr=(-H "Authorization: Bearer ${HF_TOKEN}")
  if curl -fsSL "${hdr[@]}" "${url}" -o "${MODEL_DIR}/SHA256SUMS.tmp" 2>/dev/null; then
    mv "${MODEL_DIR}/SHA256SUMS.tmp" "${MODEL_DIR}/SHA256SUMS"
  else
    rm -f "${MODEL_DIR}/SHA256SUMS.tmp"
    echo "note: ${HF_REPO} does not ship SHA256SUMS — skipping checksum verification."
  fi
}

verify_files() {
  local sums="${MODEL_DIR}/SHA256SUMS" f
  [[ -s "${sums}" ]] || { echo "warning: no SHA256SUMS — skipping checksum verification"; return 0; }
  for f in "$@"; do
    if awk -v f="${f}" '$2 == f { found = 1 } END { exit !found }' "${sums}"; then
      echo "Verifying ${f} ..."
      (cd "${MODEL_DIR}" && awk -v f="${f}" '$2 == f' SHA256SUMS | sha256sum -c -) || {
        echo "ERROR: SHA256SUMS mismatch for ${f} — delete the file and re-run ./start.sh"
        return 1
      }
    else
      echo "note: no SHA256SUMS entry for ${f} — skipping"
    fi
  done
}

WANT_MMPROJ=""
if [[ -n "${VISION}" && -n "${MMPROJ}" ]]; then
  WANT_MMPROJ="${MMPROJ}"
elif [[ -n "${VISION}" && -z "${MMPROJ}" ]]; then
  echo "warning: MODEL=${MODEL} has no vision projector in the GGUF repo — text-only mode"
fi

if [[ -f "${MODEL_MARKER}" && "$(cat "${MODEL_MARKER}" 2>/dev/null)" == "${MODEL_SPEC}" \
      && -f "${MODEL_DIR}/${HF_FILE}" \
      && ( -z "${WANT_MMPROJ}" || -f "${MODEL_DIR}/${WANT_MMPROJ}" ) ]]; then
  echo "Model already downloaded to ${MODEL_DIR}; skipping download."
else
  echo "Model missing or incomplete at ${MODEL_DIR} — downloading ..."
  mkdir -p "${MODEL_DIR}"

  need_gib="${PRESET_NEED_GIB}"
  [[ -n "${WANT_MMPROJ}" ]] && need_gib=$((need_gib + 2))
  avail_gib=$(( $(df -Pk "${MODEL_DIR}" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
  if (( avail_gib < need_gib )); then
    echo "ERROR: only ${avail_gib} GiB free at ${MODEL_DIR}; need ~${need_gib} GiB."
    exit 1
  fi

  download_file "${HF_REPO}" "${HF_FILE}" || exit 1
  if [[ -n "${WANT_MMPROJ}" ]]; then
    download_file "${HF_REPO}" "${WANT_MMPROJ}" || exit 1
  fi

  [[ -f "${MODEL_DIR}/${HF_FILE}" ]] || {
    echo "ERROR: download finished but ${HF_FILE} not found in ${MODEL_DIR}"
    exit 1
  }

  fetch_sums
  verify_files "${HF_FILE}" ${WANT_MMPROJ:+"${WANT_MMPROJ}"} || exit 1

  printf '%s\n' "${MODEL_SPEC}" > "${MODEL_MARKER}"
  echo "[$(date -Is)] model ready at ${MODEL_DIR}"
fi

# ---- container lifecycle (idempotent) ----------------------------------------
if podman ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if podman ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "Health: OK"
    else
      echo "Health: not ready yet — follow with: podman logs -f ${CONTAINER_NAME}"
    fi
    echo "OpenAI base URL: http://127.0.0.1:${PORT}/v1"
    exit 0
  fi
  echo "Removing stopped container ${CONTAINER_NAME} ..."
  podman rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting ${CONTAINER_NAME} (port ${PORT}, backend ${BACKEND}, context ${CTX_SIZE})"

SERVE_ARGS=(
  -m "/models/${HF_FILE}"
  --host 0.0.0.0
  --port 8080
  --alias "${MODEL_ALIAS}"
  -c "${CTX_SIZE}"
  -np "${PARALLEL}"
  --jinja
)
[[ -n "${KV_CACHE_TYPE}" ]] && SERVE_ARGS+=(-ctk "${KV_CACHE_TYPE}" -ctv "${KV_CACHE_TYPE}")
[[ -n "${WANT_MMPROJ}" ]] && SERVE_ARGS+=(--mmproj "/models/${WANT_MMPROJ}")
SERVE_ARGS+=("${GPU_LAYERS[@]}")

# --ipc host is mandatory for ROCm; :z relabels the bind for SELinux.
# The llama.cpp image entrypoint is the llama-server binary itself, so the
# arguments are passed directly after the image name.
EXTRA_ARGS=()
[[ -n "${PODMAN_EXTRA_ARGS}" ]] && read -r -a EXTRA_ARGS <<< "${PODMAN_EXTRA_ARGS}"
podman run -d \
  --name "${CONTAINER_NAME}" \
  "${DEV_ARGS[@]}" \
  -p "${PORT}:8080" \
  -v "${MODEL_DIR}:/models:ro,z" \
  "${EXTRA_ARGS[@]}" \
  "${IMAGE}" \
  "${SERVE_ARGS[@]}" \
  >/dev/null

echo "Spawned container ${CONTAINER_NAME}"

# ---- readiness ---------------------------------------------------------------
log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]]; then
    # setsid runs the pipe in its own process group — the signal must hit the
    # whole group, otherwise `podman logs -f` survives and the wait below hangs the script.
    kill -- "-${log_follow_pid}" 2>/dev/null \
      || kill "${log_follow_pid}" 2>/dev/null || true
    wait "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

setsid bash -c 'podman logs -f "$1" 2>&1 | tee -a "$2"' _ "${CONTAINER_NAME}" "${LOG_FILE}" &
log_follow_pid=$!

READY_URL="http://127.0.0.1:${PORT}/health"
echo "Waiting for ${READY_URL} (timeout ${READY_TIMEOUT}s)"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
heartbeat=0
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! podman ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  if (( $(date +%s) >= deadline )); then
    echo "Timed out after ${READY_TIMEOUT}s waiting for readiness"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  if (( heartbeat % 6 == 0 )); then
    echo "  still starting..."
  fi
  heartbeat=$((heartbeat + 1))
  sleep 5
done

echo
echo "Model ${MODEL_ALIAS} is ready."
echo "OpenAI base URL:  http://127.0.0.1:${PORT}/v1"
echo "Model id in OMP:  llama.cpp/${MODEL_ALIAS}"
echo "Launch OMP:       omp --model llama.cpp/${MODEL_ALIAS}"
echo "Sanity check:     curl -s http://127.0.0.1:${PORT}/v1/models"