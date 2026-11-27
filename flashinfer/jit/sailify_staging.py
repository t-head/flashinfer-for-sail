"""Online sailify staging for cuda-free (PPU/HGGC) JIT builds.

When the PPU toolchain is detected (``CUDA_SDK`` environment variable
pointing into a PPU SDK install, as set by ``envsetup.sh ppu``), every JIT
module's sources plus their ``#include`` closure are converted by sailify
into a mirror tree under ``$FLASHINFER_WORKSPACE/sailified/<module>/`` and
the ninja build compiles exclusively from that tree:

    sailified/<module>/.ppu_compat/   compat layer (sailify-generated)
    sailified/<module>/csrc/...       converted csrc sources + headers
    sailified/<module>/include/...    converted flashinfer headers
    sailified/<module>/gen/...        converted jinja-generated sources
    sailified/<module>/<file>         out-of-tree sources, staged flat

The upstream trees are never touched, so the same checkout still builds
against a real CUDA toolchain. Staged copies are synced with content
comparison: unchanged files keep their mtime so ninja stays incremental
across rebuilds.

Disable explicitly with ``FLASHINFER_JIT_SAILIFY=0``.
"""

import importlib
import importlib.util
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import env as jit_env

logger = logging.getLogger(__name__)

# module name -> (sources signature, staged_sources, include_dirs)
_STAGED: Dict[str, Tuple[tuple, List[Path], List[Path]]] = {}

# (tvm_ffi_include, dlpack_include) -> staged include dirs
_TVM_FFI_STAGED: Dict[Tuple[str, str], List[Path]] = {}

_HEADER_SUFFIXES = (".h", ".hpp", ".cuh", ".inl")
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]', re.MULTILINE)

# Targeted fixes applied to the *staged* copies after the generic sailify
# pass, for CUDA constructs the identifier-level mapping cannot express
# (same mechanism as sglang's setup_sail.py _SAILIFY_PATCHES). The upstream
# trees are never touched.
#
# The math-type headers and cooperative_groups resolve to the PPU_SDK native
# copies (whose cg matches the hggc::atomic targets the translated code uses).
# NVFP4, however, is not implemented by the PPU SDK yet; keep its type trait
# disabled unless the module explicitly enables FlashInfer FP4 support.
_SAILIFY_PATCHES = {
    "include/flashinfer/attention/prefill.cuh": [
        (
            "#if COMPATIBLE_VERSION >= 12080\ntemplate <>\nstruct is_fp4_type<__nv_fp4x2_e2m1> : std::true_type {};\n#endif",
            "#if defined(FLASHINFER_ENABLE_FP4_E2M1) && COMPATIBLE_VERSION >= 12080\ntemplate <>\nstruct is_fp4_type<__nv_fp4x2_e2m1> : std::true_type {};\n#endif",
        )
    ],
    "include/flashinfer/sampling.cuh": [
        # cuda::maximum/minimum map to hggc::maximum/minimum, but PPU SDK
        # 2.3 leaves the hggc functional dir empty. The PPU thrust port
        # ships __host__ __device__ transparent equivalents (same
        # operator<-based semantics) reachable via the cub headers.
        (
            "using MaxReduceOp = hggc::maximum<>;",
            "using MaxReduceOp = thrust::maximum<>;",
        ),
        (
            "using MinReduceOp = hggc::minimum<>;",
            "using MinReduceOp = thrust::minimum<>;",
        ),
    ],
}

# The PPU detection logic lives in flashinfer._ppu_env (dependency-free by
# design: compilation_context imports it at import time; see that module).
from .._ppu_env import ppu_sdk_root as ppu_sdk_root  # noqa: F401  (re-export)


def sailify_enabled() -> bool:
    return ppu_sdk_root() is not None


def _bundled_sailify_dir() -> Optional[Path]:
    """Locate the sailify package shipped with FlashInfer (3rdparty/sailify).

    Installed wheels carry it under flashinfer/data/sailify; a source checkout
    without a populated data dir (e.g. ``python -m flashinfer.aot``) falls
    back to the submodule itself.
    """
    for root in (
        jit_env.FLASHINFER_DATA / "sailify",
        jit_env._package_root.parent / "3rdparty" / "sailify",
    ):
        pkg = root / "sailify"
        if (pkg / "sailify_python.py").is_file():
            return pkg
    return None


def import_sailify(module: str):
    """Import ``sailify.<module>``, preferring the bundled sailify copy.

    The bundled package is bound to the top-level name ``sailify`` (its
    modules import each other relatively), so it takes precedence over any
    separately installed sailify unless one was already imported.
    """
    if "sailify" not in sys.modules:
        pkg = _bundled_sailify_dir()
        if pkg is not None:
            spec = importlib.util.spec_from_file_location(
                "sailify",
                pkg / "__init__.py",
                submodule_search_locations=[str(pkg)],
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["sailify"] = mod
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                del sys.modules["sailify"]
                raise
    return importlib.import_module(f"sailify.{module}")


def _load_sailify():
    try:
        sailify_python = import_sailify("sailify_python")
    except ImportError as exc:
        raise RuntimeError(
            "sailify is required for cuda-free (PPU) JIT builds; "
            "initialize the 3rdparty/sailify submodule (or install sailify) "
            "or set FLASHINFER_JIT_SAILIFY=0 to build upstream sources directly"
        ) from exc
    return (
        sailify_python._discover_includes,
        sailify_python._ensure_ppu_compat,
        sailify_python.preprocessor,
    )


def _staged_path(src_path: str, roots, staging: str) -> Optional[Tuple[str, str]]:
    """Map an upstream file to (staged_path, project_directory anchor).

    The anchor is the directory the preprocessor receives as
    project_directory so that output_directory + relpath(file, anchor)
    equals staged_path.
    """
    for base, sub in roots:
        if src_path.startswith(base + os.sep):
            rel = os.path.relpath(src_path, base)
            staged_rel = os.path.join(sub, rel) if sub else rel
            return (
                os.path.join(staging, staged_rel),
                os.path.dirname(base) if sub else base,
            )
    return None


def _staged_dir(real_dir: str, roots, staging: str) -> Optional[str]:
    """Map an upstream directory to its staged mirror (None if outside roots)."""
    for base, sub in roots:
        if real_dir == base:
            return os.path.join(staging, sub) if sub else staging
        if real_dir.startswith(base + os.sep):
            rel = os.path.relpath(real_dir, base)
            return (
                os.path.join(staging, sub, rel) if sub else os.path.join(staging, rel)
            )
    return None


def _mirror_join(staged_base: str, include: str, staging: str) -> Optional[str]:
    """Join an include onto a staged path, refusing to escape the mirror."""
    target = os.path.normpath(os.path.join(staged_base, include))
    root = os.path.normpath(staging)
    if target == root or target.startswith(root + os.sep):
        return target
    return None


def _sync_tree(src_dir: Path, dst_dir: Path) -> None:
    """Copy new/changed files; identical files keep their mtime.

    ninja decides staleness from mtimes, so a full re-staging must not touch
    files whose (converted) content did not change.
    """
    for path in sorted(src_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(src_dir)
        dst = dst_dir / rel
        if dst.is_file() and dst.read_bytes() == path.read_bytes():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def stage_tvm_ffi_includes(tvm_ffi_include: str, dlpack_include: str) -> List[Path]:
    """Sailify-stage the installed tvm-ffi (+dlpack) header trees for PPU JIT.

    Stock tvm-ffi distributions carry the CUDA vocabulary in
    ``tvm/ffi/extra/cuda/*`` (the ``cuda_runtime.h`` include, ``cuda*``
    calls, the cuda-vocabulary CHECK macro) — none of it resolves under a
    pure PPU toolchain, where no CUDA-named header exists. The whole
    installed include trees (small, header-only) are converted once into
    a shared staging root — one copy serves every module — and committed
    with the same mtime-preserving sync as ``stage_module`` so ninja
    stays incremental. Partially converted installs (e.g. tvm-ffi-sail,
    which keeps the CUDA spelling under ``extra/cuda``) go through the
    same path; identifiers already in hggc vocabulary are left as-is.

    Returns the staged include dirs; the caller must place them ahead of
    the unconverted originals on the ninja search path so the converted
    copies shadow them.
    """
    key = (str(tvm_ffi_include), str(dlpack_include))
    cached = _TVM_FFI_STAGED.get(key)
    if cached is not None:
        return cached

    _discover_includes, _ensure_ppu_compat, preprocessor = _load_sailify()

    sailified_root = jit_env.FLASHINFER_WORKSPACE_DIR / "sailified"
    sailified_root.mkdir(parents=True, exist_ok=True)
    staging_root = sailified_root / "_tvm_ffi"
    tmp = Path(tempfile.mkdtemp(prefix="._tvm_ffi.", dir=str(sailified_root)))

    try:
        stats: Dict[str, list] = {}
        failures: List[str] = []
        # Wheels ship dlpack inside the tvm-ffi include root; convert each
        # distinct tree once.
        src_roots = list(
            dict.fromkeys(
                Path(os.path.realpath(str(root)))
                for root in (tvm_ffi_include, dlpack_include)
            )
        )
        for src_root in src_roots:
            if not src_root.is_dir():
                continue
            # Same anchor contract as stage_module: project_directory is the
            # parent of the include root, so the output lands at
            # <staging>/include/<relpath within the root> for both trees.
            anchor = os.path.realpath(os.path.dirname(str(src_root)))
            for path in sorted(src_root.rglob("*")):
                if path.is_dir():
                    continue
                staged_rel = os.path.relpath(str(path), anchor)
                staged_file = tmp / staged_rel
                if path.suffix in _HEADER_SUFFIXES:
                    result = preprocessor(
                        output_directory=str(tmp),
                        filepath=str(path),
                        all_files=[],
                        stats=stats,
                        clean_ctx=None,
                        show_progress=False,
                        project_directory=anchor,
                    )
                    if result.status != "ok":
                        failures.append(f"{path}: {result.status}")
                    elif not staged_file.is_file():
                        failures.append(f"{path}: staged output {staged_file} missing")
                else:
                    staged_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, staged_file)

        if failures:
            raise RuntimeError(
                "sailify staging failed for the installed tvm-ffi headers:\n  "
                + "\n  ".join(failures)
            )

        _sync_tree(tmp, staging_root)
        staged_dirs = [
            staging_root / os.path.basename(str(src_root)) for src_root in src_roots
        ]
        staged_dirs = list(dict.fromkeys(staged_dirs))
        logger.info(
            "sailify: staged installed tvm-ffi headers (%d files) into %s",
            len(stats.get("ppuified", [])),
            staging_root,
        )
        _TVM_FFI_STAGED[key] = staged_dirs
        return staged_dirs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def stage_module(
    name: str,
    sources: List[Path],
    extra_include_dirs: Optional[List[Path]],
) -> Tuple[List[Path], Optional[List[Path]]]:
    """Sailify-stage `sources` (+ their #include closure) for module `name`.

    Returns (staged_sources, include_dirs) where include_dirs prepend the
    ninja search path so the staged tree shadows its upstream originals.
    ``extra_include_dirs`` with a staged mirror (e.g. TRT-LLM's
    ``csrc/nv_internal``) join the include closure and are converted like the
    rest; their mirrors replace them on the search path, in the caller's
    order. Directories outside the roots keep their original path.
    """
    signature = (
        tuple(str(s) for s in sources),
        tuple(str(d) for d in (extra_include_dirs or ())),
    )
    cached = _STAGED.get(name)
    if cached is not None and cached[0] == signature:
        return cached[1], cached[2]

    _discover_includes, _ensure_ppu_compat, preprocessor = _load_sailify()

    sailified_root = jit_env.FLASHINFER_WORKSPACE_DIR / "sailified"
    sailified_root.mkdir(parents=True, exist_ok=True)
    module_root = sailified_root / name
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=str(sailified_root)))

    try:
        # Longest root first so nested trees match the most specific entry.
        # The staged subdir must equal the basename of the upstream root:
        # _staged_path anchors at dirname(root) and derives the output path
        # from relpath(file, dirname(root)), whose first component is that
        # basename (csrc/, include/, generated/).
        def _root_entry(d: Path) -> Tuple[str, str]:
            real = os.path.realpath(str(d))
            return (real, os.path.basename(real))

        roots = sorted(
            (
                _root_entry(jit_env.FLASHINFER_CSRC_DIR),
                _root_entry(jit_env.FLASHINFER_INCLUDE_DIR),
            ),
            key=lambda kv: -len(kv[0]),
        )
        gen_dir = jit_env.FLASHINFER_GEN_SRC_DIR
        gen_real = os.path.realpath(str(gen_dir))
        if gen_dir.is_dir():
            roots.append(_root_entry(gen_dir))
            roots.sort(key=lambda kv: -len(kv[0]))

        # Directories used to resolve #include targets while walking the
        # dependency closure (mirrors the upstream -I layout).
        header_include_dirs = [
            os.path.realpath(str(jit_env.FLASHINFER_INCLUDE_DIR)),
            os.path.realpath(str(jit_env.FLASHINFER_CSRC_DIR)),
        ]
        if gen_dir.is_dir():
            header_include_dirs.append(os.path.realpath(str(gen_dir)))
        # extra_include_dirs are appended last because the ninja flags append
        # them last too; skipping them here left e.g. TRT-LLM's
        # csrc/nv_internal/include headers unstaged and therefore unconverted,
        # so their <cuda_fp16.h>/<cublasLt.h> includes never resolved on PPU.
        # Only directories with a staged mirror take part: directories outside
        # the roots would be staged by basename, where like-named headers from
        # different trees overwrite each other.
        extra_reals: List[str] = []
        for extra in extra_include_dirs or ():
            real = os.path.realpath(str(extra))
            if real in extra_reals:
                continue
            extra_reals.append(real)
            if os.path.isdir(real) and _staged_dir(real, roots, str(tmp)):
                header_include_dirs.append(real)

        staged_sources = []
        flat_used = False
        visited = {os.path.realpath(str(s)) for s in sources}
        pending = list(visited)
        stats: Dict[str, list] = {}
        failures: List[str] = []
        # A symlink inside a root can point at a file outside every root (e.g.
        # gen_root's trtllmGen_bmm_export). Its realpath has no staged mirror,
        # so remember the include-relative path the compiler will use; the
        # staged copy must land there or the qualified include stops resolving.
        logical_targets: Dict[str, str] = {}
        while pending:
            src_path = pending.pop(0)
            _discover_includes(src_path, header_include_dirs, visited, pending)
            # sailify's generic include walker requires whitespace after
            # ``include`` and limits the accepted suffixes. FlashInfer emits
            # both ``#include<...>`` and textual .inc configs, so resolve the
            # complete local include closure here as well.
            try:
                source = Path(src_path).read_text(errors="replace")
            except OSError:
                source = ""
            src_logical = logical_targets.get(src_path)
            for include in _INCLUDE_RE.findall(source):
                resolved = None
                logical = None
                sibling = os.path.join(os.path.dirname(src_path), include)
                if os.path.isfile(sibling):
                    resolved = os.path.realpath(sibling)
                    if src_logical is not None:
                        logical = _mirror_join(
                            os.path.dirname(src_logical), include, str(tmp)
                        )
                else:
                    for d in header_include_dirs:
                        candidate = os.path.join(d, include)
                        if os.path.isfile(candidate):
                            resolved = os.path.realpath(candidate)
                            staged_dir = _staged_dir(d, roots, str(tmp))
                            if staged_dir is not None:
                                logical = _mirror_join(staged_dir, include, str(tmp))
                            break
                if resolved is None:
                    continue
                if resolved not in visited:
                    visited.add(resolved)
                    pending.append(resolved)
                if logical is not None:
                    logical_targets.setdefault(resolved, logical)
            mapped = _staged_path(src_path, roots, str(tmp))
            logical = logical_targets.get(src_path)
            if mapped is not None:
                staged_path, anchor = mapped
                converted = staged_path
            elif logical is not None:
                # Converted next to the module root, then moved to the path the
                # include-relative lookup expects.
                anchor = os.path.dirname(src_path)
                converted = logical
            else:
                # Out-of-tree source (e.g. generated next to the workspace):
                # stage flat at the module root.
                anchor = os.path.dirname(src_path)
                converted = os.path.join(str(tmp), os.path.basename(src_path))
                flat_used = True
            result = preprocessor(
                output_directory=str(tmp),
                filepath=src_path,
                all_files=[],
                stats=stats,
                clean_ctx=None,
                show_progress=False,
                project_directory=anchor,
            )
            if result.status == "ok" and mapped is None and logical is not None:
                produced = os.path.join(str(tmp), os.path.basename(src_path))
                if os.path.abspath(produced) != os.path.abspath(converted):
                    os.makedirs(os.path.dirname(converted), exist_ok=True)
                    shutil.move(produced, converted)
            if result.status != "ok":
                failures.append(f"{src_path}: {result.status}")
            elif not os.path.isfile(converted):
                failures.append(f"{src_path}: staged output {converted} missing")

        if failures:
            raise RuntimeError(
                f"sailify staging failed for module {name}:\n  "
                + "\n  ".join(failures)
            )

        # Regenerate the compat layer inside the staging tree.
        _ensure_ppu_compat(str(tmp))

        # Apply targeted post-sailify patches to the staged copies only;
        # the upstream trees are never touched.
        for rel_path, replacements in _SAILIFY_PATCHES.items():
            staged_file = tmp / rel_path
            # A module only stages its own include closure, so most targeted
            # patches are legitimately absent from unrelated modules.
            if not staged_file.is_file():
                continue
            text = staged_file.read_text(encoding="utf-8", errors="replace")
            for old, new in replacements:
                text = text.replace(old, new)
            staged_file.write_text(text, encoding="utf-8")

        # Headers dropped from the shim set must not linger and hijack includes
        # (the compat dir precedes the PPU_SDK include path), but clearing the
        # whole directory is too coarse: every shim header would come back with
        # a fresh mtime, and since every staged .cu depends on them ninja would
        # then rebuild the module from scratch in each new process. Remove only
        # the orphans and leave the rest to the mtime-preserving sync below.
        compat_dir = module_root / ".ppu_compat"
        staged_compat = tmp / ".ppu_compat"
        if compat_dir.is_dir():
            # Reverse order visits children before their parent directories.
            for stale in sorted(compat_dir.rglob("*"), reverse=True):
                if stale.is_dir():
                    if not any(stale.iterdir()):
                        stale.rmdir()
                elif not (staged_compat / stale.relative_to(compat_dir)).is_file():
                    stale.unlink()
        # Commit the staging tree with mtime-preserving sync.
        _sync_tree(tmp, module_root)

        for src in sources:
            real = os.path.realpath(str(src))
            mapped = _staged_path(real, roots, str(module_root))
            if mapped is None:
                staged_sources.append(module_root / os.path.basename(real))
            else:
                staged_sources.append(Path(mapped[0]))

        include_dirs = [
            module_root / ".ppu_compat",
            module_root / "include",
            module_root / "csrc",
        ]
        if gen_dir.is_dir():
            include_dirs.append(module_root / os.path.basename(gen_real))
        if flat_used:
            include_dirs.append(module_root)

        # Shadow the extra include dirs with their staged mirrors: the
        # originals still spell CUDA vocabulary that a PPU toolchain cannot
        # resolve, and the converted copies (staged above via the closure
        # walk) must win over them. Mapping in place keeps the caller's
        # search precedence; dirs without a mirror keep their original path.
        mapped_extra_dirs: List[Path] = []
        for real in extra_reals:
            staged = _staged_dir(real, roots, str(module_root))
            if staged is not None and os.path.isdir(staged):
                mapped_extra_dirs.append(Path(staged))
            else:
                mapped_extra_dirs.append(Path(real))
                logger.debug("sailify: %s has no staged mirror for %s", real, name)
        include_dirs = include_dirs + mapped_extra_dirs
        include_dirs = list(dict.fromkeys(include_dirs))

        n_headers = max(len(stats.get("ppuified", [])) - len(sources), 0)
        logger.info(
            "sailify: staged %d sources (+%d headers) for %s into %s",
            len(sources),
            n_headers,
            name,
            module_root,
        )

        result_tuple = (staged_sources, include_dirs)
        _STAGED[name] = (signature, staged_sources, include_dirs)
        return staged_sources, include_dirs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
