"""Regression test: extra_include_dirs must join the sailify staging closure.

`gen_dsv3_fused_routing_module()` passes TRT-LLM's `csrc/nv_internal*` trees as
`extra_include_paths`. If staging skips them, their headers are neither
converted nor shadowed, and a PPU build fails on `<cuda_fp16.h>`,
`<cublasLt.h>` and `<cuda_runtime_api.h>`.
"""

import os
import re
from types import SimpleNamespace

import pytest

from flashinfer.jit import env as jit_env
from flashinfer.jit import sailify_staging

INCLUDE_RE = re.compile(r'#include\s*[<"]([^">]+)[">]')

CONVERT = {
    "cuda_fp16.h": "hggc_fp16.h",
    "cublasLt.h": "acblasLt.h",
    "cuda_runtime_api.h": "hggc_runtime_api.h",
}


def _discover_includes(src, include_dirs, visited, pending):
    try:
        text = open(src, errors="replace").read()
    except OSError:
        return
    for inc in INCLUDE_RE.findall(text):
        for cand in [os.path.join(os.path.dirname(src), inc)] + [
            os.path.join(d, inc) for d in include_dirs
        ]:
            cand = os.path.realpath(cand)
            if os.path.isfile(cand):
                if cand not in visited:
                    visited.add(cand)
                    pending.append(cand)
                break


def _preprocessor(
    output_directory,
    filepath,
    all_files,
    stats,
    clean_ctx,
    show_progress,
    project_directory,
):
    with open(filepath, errors="replace") as f:
        text = f.read()
    for old, new in CONVERT.items():
        text = text.replace(f"<{old}>", f"<{new}>")
    text = text.replace("cudaError_t", "hggcError_t")
    out = os.path.join(
        output_directory, os.path.relpath(filepath, project_directory)
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(text)
    stats.setdefault("ppuified", []).append(filepath)
    return SimpleNamespace(status="ok")


def _ensure_ppu_compat(root):
    os.makedirs(os.path.join(root, ".ppu_compat"), exist_ok=True)


@pytest.fixture
def staged(tmp_path, monkeypatch):
    csrc = tmp_path / "csrc"
    include = tmp_path / "include"
    gen = tmp_path / "gen"
    extra = tmp_path / "thirdparty" / "cutlass" / "include"
    for d in (csrc, include, gen, extra):
        d.mkdir(parents=True)

    (csrc / "fused_moe").mkdir()
    (csrc / "fused_moe" / "noAuxTcKernels.cu").write_text(
        '#include "tensorrt_llm/common/cudaUtils.h"\n'
        '#include "moeTopKFuncs.cuh"\n'
    )
    (csrc / "fused_moe" / "moeTopKFuncs.cuh").write_text("#pragma once\n")
    nv_include = csrc / "nv_internal" / "include" / "tensorrt_llm" / "common"
    nv_include.mkdir(parents=True)
    (nv_include / "cudaUtils.h").write_text(
        "#pragma once\n"
        "#include <cuda_fp16.h>\n"
        "#include <cublasLt.h>\n"
        '#include "tensorrt_llm/common/tllmException.h"\n'
        "static_assert(sizeof(cudaError_t) > 0, \"\");\n"
    )
    (nv_include / "tllmException.h").write_text(
        "#pragma once\n#include <cuda_runtime_api.h>\n"
    )
    (extra / "cutlass_util.h").write_text("#pragma once\n")

    monkeypatch.setattr(jit_env, "FLASHINFER_CSRC_DIR", csrc)
    monkeypatch.setattr(jit_env, "FLASHINFER_INCLUDE_DIR", include)
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", gen)
    monkeypatch.setattr(jit_env, "FLASHINFER_WORKSPACE_DIR", tmp_path / "cache")
    monkeypatch.setattr(
        sailify_staging,
        "_load_sailify",
        lambda: (_discover_includes, _ensure_ppu_compat, _preprocessor),
    )
    return SimpleNamespace(
        csrc=csrc,
        extra=extra,
        module_root=tmp_path / "cache" / "sailified" / "test_module",
    )


def _stage(staged, sources, extras):
    return sailify_staging.stage_module("test_module", sources, extras)


def test_extra_include_dirs_are_staged_and_converted(staged):
    csrc = staged.csrc
    sources = [csrc / "fused_moe" / "noAuxTcKernels.cu"]
    extras = [csrc / "nv_internal", csrc / "nv_internal" / "include", staged.extra]

    _, include_dirs = _stage(staged, sources, extras)

    converted = (
        staged.module_root
        / "csrc"
        / "nv_internal"
        / "include"
        / "tensorrt_llm"
        / "common"
        / "cudaUtils.h"
    )
    text = converted.read_text()
    assert "hggc_fp16.h" in text
    assert "acblasLt.h" in text
    assert "hggc_runtime_api.h" in (converted.parent / "tllmException.h").read_text()
    assert "cuda_fp16.h" not in text
    assert "cudaError_t" not in text

    dirs = [str(d) for d in include_dirs]
    staged_extra = str(staged.module_root / "csrc" / "nv_internal" / "include")
    assert staged_extra in dirs
    assert str(csrc / "nv_internal" / "include") not in dirs
    pass_through = str(staged.extra)
    assert pass_through in dirs
    assert dirs.index(staged_extra) < dirs.index(pass_through)


def test_staged_sources_come_from_the_mirror(staged):
    csrc = staged.csrc
    sources = [csrc / "fused_moe" / "noAuxTcKernels.cu"]
    staged_sources, _ = _stage(staged, sources, [])
    assert [str(s) for s in staged_sources] == [
        str(staged.module_root / "csrc" / "fused_moe" / "noAuxTcKernels.cu")
    ]


def test_cache_is_keyed_on_extra_include_dirs(staged):
    csrc = staged.csrc
    sources = [csrc / "fused_moe" / "noAuxTcKernels.cu"]
    _, without_extras = _stage(staged, sources, [])
    _, with_extras = _stage(staged, sources, [csrc / "nv_internal" / "include"])
    assert [str(d) for d in without_extras] != [str(d) for d in with_extras]
