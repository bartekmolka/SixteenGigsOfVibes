# LLM server in Podman for 16 GB RAM machines

Serves lightweight **llama.cpp (GGUF)** models in a rootless Podman container. It is a
recipe (preflight → image → checkpoint download → container → health-check) for small machines:
it uses **official llama.cpp images** from `ghcr.io` and GGUF presets that fit in 16 GB of RAM/VRAM.

One command brings it up; one command tears it down.

```bash
cp .env.sample .env   # optional — every variable has a default
MODEL=qwen38 ./start.sh
./stop.sh
```

## What's here

| File         | Role                                                                                                                                                     |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `start.sh`   | One-command entrypoint: preflight → backend autodetection → image pull (cached) → model download (resumable, SHA256SUMS-verified when the repo ships them) → run → health-poll. |
| `stop.sh`    | Stops the container (60 s grace period); idempotent.                                                                                                     |
| `.env.sample`| Optional overrides — copy to `.env`.                                                                                                                     |

There is no `Containerfile` here (unlike the original): we run the ready-made
`ghcr.io/ggml-org/llama.cpp:server[-cuda|-rocm|-vulkan]` images, so nothing gets compiled.
The model is **not** baked into the image; it is bind-mounted read-only at `/models`.

## Presets

Pick a preset with `MODEL=…` (default: `qwen38`). Each preset selects the HF repo, the GGUF file,
the model alias visible in `/v1/models`, an optional vision projector, the context size and the KV cache type.

| Preset     | Model                                                                                           | File                             | Size    | mmproj         | Context | Notes                             |
| ---------- | ----------------------------------------------------------------------------------------------- | -------------------------------- | ------- | -------------- | ------- | --------------------------------- |
| `qwen38`   | [empero-ai/Qwen3.8-9B-Distill-GGUF](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF)    | `Qwen3.8-9B-Q4_K_M.gguf`         | 5.78 GB | —              | 32K     | text-only, reasoning, **default** |
| `ornith`   | [bartowski/Ornith-1.5-9B-GGUF](https://huggingface.co/bartowski/Ornith-1.5-9B-GGUF)              | `Ornith-1.5-9B-Q4_K_M.gguf`      | 5.91 GB | 0.92 GB (f16)  | 32K     | multimodal (images)               |
| `bonsai27` | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)  | `Ternary-Bonsai-2-27B-Q2_0.gguf` | 7.63 GB | 0.63 GB (Q8_0) | 16K     | 27B ternary 2-bit; KV in `q8_0`   |

### Memory budget (16 GB)

Weights + KV cache + (optionally) the mmproj + OS/browser all have to fit in 16 GB:

| Preset     | Weights | mmproj | Left for KV + system |
| ---------- | ------- | ------ | -------------------- |
| `qwen38`   | 5.8 GB  | —      | ~10 GB               |
| `ornith`   | 5.9 GB  | 0.9 GB | ~9 GB                |
| `bonsai27` | 7.6 GB  | 0.6 GB | ~7 GB (KV in `q8_0`) |

When it gets tight (symptoms: slow loads, OOM killer, "allocation failed" in the log):

1. Lower `CTX_SIZE` (e.g. `CTX_SIZE=16384`) — no re-download needed.
2. Set `KV_CACHE_TYPE=q8_0` — half the KV size at a small quality cost (the `bonsai27` preset does this by default).
3. Keep `PARALLEL=1` — the context is split between slots.
4. `qwen38` and `ornith` use a hybrid architecture (gated-deltanet), so KV grows more slowly than with full attention;
   `bonsai27` at 16K fits within the ~7 GB headroom.

## Requirements

- **Podman** (rootless) — `podman --version` ≥ 5.x
- `curl`, `python3`, and `hf` (from `pip install huggingface_hub[cli]`) or the `huggingface_hub`
  Python package on the host — for the first model download
- **~10 GiB free disk** for the chosen model + 1–10 GB for the backend image (ROCm/CUDA images are large; Vulkan/CPU are small)
- 16 GB RAM/VRAM; a GPU (dGPU or iGPU) is a big speedup, but CPU-only mode works everywhere

GPU backends additionally need on the host:

- **CUDA**: NVIDIA driver + `nvidia-container-toolkit` (podman uses `--gpus all` via CDI)
- **ROCm**: `/dev/kfd` and `/dev/dri` + the user in the `render` group (`id -nG`)
- **Vulkan**: `/dev/dri` + a Vulkan driver (e.g. `mesa-vulkan-drivers`)

If none of these is present, `start.sh` falls back to **CPU** — slower, but it always works.

## Quick start

```bash
cp .env.sample .env      # optional
MODEL=qwen38 ./start.sh
```

First run:

1. Picks and pulls the backend image (autodetection: `nvidia-smi` → cuda, `/dev/kfd` → rocm, `/dev/dri` → vulkan, otherwise cpu).
2. Downloads the GGUF file into `./models` (resumable; verifies SHA256SUMS when the repo ships them).
3. Starts the container and streams the server log to `.SixteenGigsOfVibes.log`.
4. Exits "ready" once `/health` returns 200.

Then:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq -r '.data[0].id'
# -> qwen3.8-9b-distill-q4_k_m
```

Tear down:

```bash
./stop.sh
```

Re-running `./start.sh` skips the downloads and reaches ready in seconds. **Switching models**
(`MODEL=ornith ./start.sh`) only downloads the new file — the old one stays in `./models`
and works again when you switch back to its preset.

## Configuration

Copy `.env.sample` to `.env` and edit. Precedence (highest first):
**exported shell env → `.env` → start.sh defaults (preset-provided).**

| Var                 | Default                              | Meaning                                                            |
| ------------------- | ------------------------------------ | ------------------------------------------------------------------ |
| `MODEL`             | `qwen38`                             | Preset: `qwen38` \| `ornith` \| `bonsai27`                          |
| `HF_REPO`           | *from preset*                        | HF repo override                                                   |
| `HF_FILE`           | *from preset*                        | GGUF file in the repo (e.g. a different quant)                     |
| `MODEL_ALIAS`       | *from preset*                        | Model id in the `/v1/models` API                                   |
| `VISION`            | *(empty = off)*                      | Non-empty → adds `--mmproj` (only presets with a projector)         |
| `HF_REVISION`       | *(main)*                             | Pin a commit SHA for reproducibility                               |
| `HF_TOKEN`          | *(empty)*                            | HuggingFace token — higher download rate limits                    |
| `BACKEND`           | `auto`                               | `auto` \| `cuda` \| `rocm` \| `vulkan` \| `cpu`                     |
| `IMAGE`             | *auto per backend*                   | Override the image tag                                             |
| `CONTAINER_NAME`    | `SixteenGigsOfVibes-llama`                       | Container name                                                     |
| `PORT`              | `8080`                               | Host port (container listens on 8080)                              |
| `MODEL_DIR`         | `./models`                           | Directory with the GGUF files                                      |
| `LOG_FILE`          | `./.SixteenGigsOfVibes.log`                      | Server log                                                         |
| `READY_TIMEOUT`     | `600`                                | Seconds to wait for `/health`                                      |
| `CTX_SIZE`          | *from preset* (32K/32K/16K)          | Context window in tokens                                           |
| `PARALLEL`          | `1`                                  | Server slots; context is split between them                        |
| `KV_CACHE_TYPE`     | *from preset* (`q8_0` for bonsai27)  | KV cache type (`-ctk`/`-ctv`)                                      |
| `PODMAN_EXTRA_ARGS` | *(empty)*                            | Extra `podman run` arguments (space-separated)                     |

### A different quant of the same model

A preset is only a default — you can swap the file without touching the script:

```bash
MODEL=qwen38  HF_FILE=Qwen3.8-9B-Q5_K_M.gguf    MODEL_ALIAS=qwen3.8-9b-q5_k_m ./start.sh    # 6.64 GB
MODEL=ornith  HF_FILE=Ornith-1.5-9B-IQ4_XS.gguf MODEL_ALIAS=ornith-1.5-9b-iq4_xs ./start.sh # 5.24 GB
```

## Image input (ornith, bonsai27)

The server loads the projector (`--mmproj`) only when you set `VISION=1`:

```bash
MODEL=ornith VISION=1 ./start.sh
```

An image goes in as an `image_url` part of a user message — a `data:` URI or a local path
visible inside the container (the server does not fetch remote URLs). One image costs from
a handful up to ~1,000 tokens depending on its size.

## How it works

1. `start.sh` reads `.env`, picks the preset and resolves paths against the script's directory.
2. Preflight: `podman`, `curl`, `python3`, `CTX_SIZE`/`PARALLEL` sanity checks.
3. Backend autodetection (cuda → rocm → vulkan → cpu) and image selection `ghcr.io/ggml-org/llama.cpp:server[-backend]`.
4. Image pull (idempotent, Podman cache).
5. Model download on the host via `hf`/`huggingface_hub` (resumable, 3 attempts); SHA256SUMS
   verification when the repo ships them (currently only `empero-ai/...`). The
   `models/.download-complete` marker records the full set: preset|repo/file@revision|vision.
6. `podman run -d` with `--device /dev/kfd --device /dev/dri` (ROCm/Vulkan), `--gpus all` (CUDA),
   `--ipc host`, `--ulimit memlock=-1:-1`; `models/` bind-mounted `:ro,z` (SELinux relabel).
7. `llama-server` starts with `--jinja` (chat templates from the GGUF metadata), the model alias,
   `-c ${CTX_SIZE}`, `-np ${PARALLEL}`; health-poll on `/health` with a ~30 s heartbeat.

## Example request

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-9b-distill-q4_k_m",
    "messages": [{"role": "user", "content": "Explain what a KV cache is, briefly."}],
    "max_tokens": 256
  }' | jq -r '.choices[0].message'
```

Reasoning-mode responses contain an explicit thinking block inside the message content;
running with `--jinja` uses the chat template embedded in the GGUF metadata, so the thinking
behaviour matches upstream.

## Troubleshooting

**CUDA: `--gpus all` fails** — install `nvidia-container-toolkit` and generate the CDI spec:
`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`; check `podman info | grep -i cdi`.

**ROCm: `Permission denied` on `/dev/kfd`** — add yourself to the `render` group and log back in.

**OOM / container dies while loading** — 16 GB is tight for the current config:
lower `CTX_SIZE`, set `KV_CACHE_TYPE=q8_0`, close the browser, check `dmesg | grep -i oom`.

**Host OOM under load** — as above; `PARALLEL>1` multiplies the KV requirement.

**`hf: command not found`** — install on the host:

```bash
pip install 'huggingface_hub[cli]'
```

**No SHA256SUMS in the repo** — `start.sh` prints `note: ... skipping checksum verification` and
continues (currently only `empero-ai/Qwen3.8-9B-Distill-GGUF` ships checksums). A corrupted file
would only surface when the model is loaded.

**Container exits immediately** — check the logs:

```bash
podman logs SixteenGigsOfVibes-llama
```

Common causes: missing GPU driver, OOM while loading the weights, or a GGUF architecture
unsupported by that llama.cpp build (update the image: `podman pull ${IMAGE}`).

**SELinux `Permission denied` on `models/`** — `start.sh` sets `:z` on the bind; after a manual
relabel just re-run `./start.sh`.

## Docker instead of Podman

The recipe works 1:1 — swap `podman` for `docker` in the calls (`run/pull/ps/rm/stop/logs`);
the container flags are the same (`--gpus all` / `--device ...`, `--ipc host`; `:z` is SELinux-specific
and simply ignored by Docker when SELinux is off).

## Differences from the original recipe (MiMo)

| Original (Strix Halo 128 GB)                       | This adaptation (16 GB)                                    |
| -------------------------------------------------- | ---------------------------------------------------------- |
| Kyojin/ExLlamaV3, EXL3, custom `Containerfile` (10–20 min build) | Official llama.cpp images (pull, zero build)  |
| One model, ~107 GB                                 | GGUF presets, 5.8–7.6 GB (resumable single-file downloads) |
| DFlash speculative decoding, 256K context          | `CTX_SIZE` 16–32K, `KV_CACHE_TYPE=q8_0` for the 27B         |
| Backend: ROCm gfx1151 only                         | Autodetection: cuda / rocm / vulkan / cpu                  |
| Vision: tower packed with the model                | Vision: `--mmproj` + `VISION=1`                            |

## Licence

- `Qwen3.8-9B-Distill` (base) and `Ternary-Bonsai-2-27B` — Apache-2.0; `Ornith-1.5-9B` — MIT
  (see the model cards on Hugging Face for details).
- llama.cpp — MIT (the `ghcr.io/ggml-org/llama.cpp` images).
- The script skeleton is adapted from [piotreknow02/MiMo-V2.6-Flash-Strix-Halo](https://github.com/piotreknow02/MiMo-V2.6-Flash-Strix-Halo)
  — provided as-is, without warranty.
