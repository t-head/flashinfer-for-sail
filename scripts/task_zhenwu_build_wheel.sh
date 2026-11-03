#!/usr/bin/env bash
# Build the FlashInfer PPU (SAIL) wheel inside the Zhenwu pod.
#
# Pure-JIT main wheel: ships the flashinfer Python package, its JIT runtime
# and the kernel *sources*/headers (csrc, include, cutlass/spdlog/cccl) — NO
# ahead-of-time kernel pre-compilation. Kernels are sailified to hgcc and
# compiled on first use at runtime. The main wheel's build_wheel() never
# compiled kernels anyway (it only stages csrc/include via _create_data_dir),
# so the old `python -m flashinfer.aot` step merely filled the throwaway
# runtime workspace cache; dropping it keeps the wheel byte-for-byte the same
# while removing the bulk of the build time.
#
# Adapted to the GitHub Actions ppu-scheduler-action environment: the source
# tree is staged at $PPU_SOURCE_DIR and the wheel is published to $ARTIFACT_DIR
# on the NAS mount so the workflow can upload it as an artifact.
set -eu

cd "${PPU_SOURCE_DIR:-/workspace/source}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo '=== Zhenwu PPU Wheel Build ==='
echo "Hostname: $(hostname), Node: ${NODE_NAME:-}"

# PPU SDK / PCCL: the Zhenwu image ships both; source envsetup only when
# present so local cuda-free runs keep the native toolchain.
# The envsetup scripts reference unset positional/loop vars (e.g. pccl's
# `$1`), which `set -u` treats as a fatal "unbound variable" — relax -u
# around the sourcing, exactly like the TE ppu-build workflow does.
if [ -f /usr/local/PPU_SDK/envsetup.sh ]; then
  # shellcheck disable=SC1091
  set +u; source /usr/local/PPU_SDK/envsetup.sh; set -u
fi
if [ -f /usr/local/pccl/envsetup.sh ]; then
  # shellcheck disable=SC1091
  set +u; source /usr/local/pccl/envsetup.sh; set -u
fi

echo '=== Fetch git submodules ==='
if git rev-parse --git-dir >/dev/null 2>&1; then
  git submodule sync --recursive
  git submodule update --init --recursive
else
  echo 'Not a git working tree; assume submodules already present.'
fi

# cuda-free bootstrap: without the CUDA compatibility SDK the stale
# CUDA_HOME/CUDA_PATH/CUDA_SDK exports point at a deleted directory — drop
# them so the native PPU toolchain wins detection. PyTorch-for-SAIL still
# links the CUDA ABI, so generate minimal loadable stubs
# (libcuda/libcublas/libcublasLt/libcudnn) and put the pip wheel's
# libcudart on the loader path; `import torch` then succeeds without any
# CUDA runtime, and kernel builds link libhggcrt1 only.
if [ ! -d /usr/local/PPU_SDK/CUDA_SDK ]; then
  unset CUDA_HOME CUDA_PATH CUDA_SDK
  CUDA_RT_LIB_DIR=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'] + '/nvidia/cu13/lib')" 2>/dev/null || true)
  if [ -n "$CUDA_RT_LIB_DIR" ] && [ ! -f "$CUDA_RT_LIB_DIR/libcudart.so.13" ]; then
    CUDA_RT_LIB_DIR=""
  fi
  if [ -n "$CUDA_RT_LIB_DIR" ]; then
    python scripts/gen_cuda_abi_stubs.py --out "$PWD/.cuda_abi_stubs"
  else
    python scripts/gen_cuda_abi_stubs.py --out "$PWD/.cuda_abi_stubs" --with-cudart
  fi
  export LD_LIBRARY_PATH="$PWD/.cuda_abi_stubs${CUDA_RT_LIB_DIR:+:$CUDA_RT_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

if [ -n "${CUDA_PATH:-}" ]; then
  export NCCL_INCLUDE_DIR=${CUDA_PATH}/include
fi
if [ -n "${NCCL_HOME:-}" ]; then
  export NCCL_LIB_DIR=${NCCL_HOME}/lib
fi

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UMD_PLATFORM_TYPE=0            # enable CModel for build
export HGGC_DRIVER_CANDIDATE=FAKE
export ALIPPU_HW_TYPE=MODEL_BEHAVIOR
export FLASHINFER_CUDA_ARCH_LIST="8.0"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-8}"

pip install packaging setuptools build pyproject_hooks --force-reinstall --no-deps
pip install --upgrade filelock
pip install tomli
pip install apache-tvm-ffi
pip install pynvml==13.0.1

# Pure JIT: no `python -m flashinfer.aot` pre-compile. build_wheel() ships
# sources + JIT runtime only; kernels are compiled on first use at runtime.
echo '=== Build wheel (pure JIT, no AOT pre-compile) ==='
SECONDS=0
rc=0
FLASHINFER_BUILD_NO_PIP=1 BUILD_NVEP=0 python -m build --no-isolation --wheel 2>&1 | tee ppu_wheel_build.log || rc=$?
echo "[build] wheel 构建耗时 ${SECONDS}s，退出码 ${rc}"
[ "${rc}" -eq 0 ] || exit "${rc}"

echo '=== Publish wheel ==='
ARTIFACT_DIR="${ARTIFACT_DIR:-/mnt/wl_nas/flashinfer-ci/artifacts}"
rm -rf "${ARTIFACT_DIR}"
mkdir -p "${ARTIFACT_DIR}"
cp -v dist/*.whl "${ARTIFACT_DIR}/"
git rev-parse HEAD > "${ARTIFACT_DIR}/commit" 2>/dev/null || true
ls -lh "${ARTIFACT_DIR}"
echo "=== DONE ==="
