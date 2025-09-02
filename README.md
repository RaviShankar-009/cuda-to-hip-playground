Blog Link here: https://medium.com/@ravi95014/i-set-out-to-test-whether-with-minimal-prior-knowledge-of-cuda-and-rocm-i-could-use-chatgpt-b92926d8e610

# CUDA -> HIP Playground

# cuda-to-hip-playground

Minimal, reproducible demos showing the *same* vector addition program running on:

- **NVIDIA (CUDA)** — via Google Colab (T4 GPU)
- **AMD (ROCm)** — via AMD Developer Cloud (MI300X)

The goal is to prove “one concept, two vendors” and document the porting flow **CUDA → HIP** (via `hipify`) with no prior deep expertise.

---

## Why this repo?

- **CUDA dominates** deep learning but is **NVIDIA-only**.
- I’m pivoting toward **GenAI inference**; I wanted a cross-vendor exercise that leverages my **SW/FW/HW** background and produces public artifacts (code + steps).

---

## Repo layout

---

## Quickstart A — NVIDIA on Colab (CUDA)

> In Colab: **Runtime → Change runtime type → GPU**. Create a cell and paste the block below (first line must be exactly `%%script /bin/sh`).

```bash
%%script /bin/sh
set -eux
if ! command -v nvidia-smi >/dev/null; then echo "Enable GPU runtime"; exit 1; fi
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.' | head -n1)"; [ -n "$SM" ] || SM=75
ARCH="sm_${SM}"

rm -rf /content/cuda-samples /content/vectorAdd* || true
git clone --depth=1 https://github.com/NVIDIA/cuda-samples.git /content/cuda-samples

NVCC=/usr/local/cuda/bin/nvcc
SRC=/content/cuda-samples/Samples/0_Introduction/vectorAdd/vectorAdd.cu
INC=/content/cuda-samples/Common
OUT=/content/vectorAdd

"$NVCC" -O3 -std=c++17 -I"$INC" -arch="$ARCH" -o "$OUT" "$SRC"
echo "Running vectorAdd..."
"$OUT"

---

# 1) Sanity
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}
rocminfo | grep -m1 gfx     # expect gfx942
hipcc --version

# 2) Get the repo
mkdir -p ~/code && cd ~/code
git clone https://github.com/{your-username}/cuda-to-hip-playground.git
cd cuda-to-hip-playground

# 3) Convert CUDA -> HIP (hipify-perl avoids needing CUDA headers)
which hipify-perl || sudo apt-get update && sudo apt-get install -y hipify-perl
mkdir -p hip_src
hipify-perl cuda_src/vectorAdd.cu > hip_src/vectorAdd_hip.cpp

# 4) Build & run for MI300X
mkdir -p build/hip
hipcc -O3 --offload-arch=gfx942 -o build/hip/vectorAdd_hip hip_src/vectorAdd_hip.cpp
./build/hip/vectorAdd_hip 10000000

// host (CPU)
float* h_A = (float*)malloc(size);

// device (GPU)
float* d_A = nullptr;
hipMalloc((void**)&d_A, size);   // hipcc replaces cudaMalloc → hipMalloc
hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);

Result snapshot

Default vector Addition of 50000 elements
Colab T4 (CUDA): Result = PASS, time ~130 ms (varies).

MI300X (ROCm): vectorAdd OK | N=16777216 | time=1.030 ms | BW=195.38 GB/s

Please note the elapsed time is not a apple to apple comparision between Nvidia and AMD GPU HW. For true end to end timing I need to pin host memory, run multiple iteration and report average time for better accuracy.
