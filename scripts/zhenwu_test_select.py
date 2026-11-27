#!/usr/bin/env python3
"""Emit the trimmed pytest node IDs for the Zhenwu PPU lane.

The upstream 0.7.4 parametrizations expand to several thousand cases per file,
which does not fit the lane budget.  For the wide matrices we keep only the
dimension boundaries (smallest/largest batch, hidden/vocab, head_dim, ...) so
the lane covers the same edges as the v0.7.1 (PR #17) green run without running
the interior combinations.  Everything else is kept in full.

Blackwell-only sampling tests are dropped because they skip on non-SM100/103
devices and never exercise the PPU.

The node IDs are taken from the current pytest in the environment, so they
always match the version that will run them (no cross-version drift between a
checked-in list and CI's pytest).
"""

import re
import subprocess
import sys

BLACKWELL_PREFIXES = ("test_blackwell_", "test_softmax_blackwell_")

_NORMAL = r"normal_distribution\(std=1\)"
_GUMBEL = r"gumbel_distribution\(beta=0\.1\)"
_DIST = f"({_NORMAL}|{_GUMBEL})"

# function -> regex that must match the "[...]" parameter suffix to be kept.
_RULES = {
    # activation: enable_pdl x seq_len x batch_size x dim (bottom-up ID order)
    "test_fused_silu_mul": r"\[(True|False)-(1|512)-(1|16)-(128|2048|16384)\]",
    "test_fused_gelu_mul": r"\[(True|False)-(1|512)-(1|16)-(128|2048|16384)\]",
    "test_fused_gelu_tanh_mul": r"\[(True|False)-(1|512)-(1|16)-(128|2048|16384)\]",
    # norm: contiguous x pdl x specify_out x dtype x hidden x batch
    "test_norm": r"\[(True|False)-(True|False)-(True|False)-dtype0-(111|16384)-(1|989)\]",
    "test_gemma_norm": r"\[(True|False)-(True|False)-(True|False)-dtype0-(111|16384)-(1|989)\]",
    # qknorm: contiguous x pdl x specify_out x dtype x head_dim x num_heads x batch
    "test_qknorm": r"\[(True|False)-(True|False)-(True|False)-dtype0-(64|512)-(4|16)-(1|989)\]",
    "test_fused_add_rmsnorm": r"\[(True|False)-(True|False)-dtype0-(111|16384)-(1|989)\]",
    "test_fused_add_rmsnorm_fp8_block_quant": r"\[(True|False)-dtype[01]-(256|16384)-(1|989)\]",
    "test_gemma_fused_add_rmsnorm": r"\[(True|False)-(True|False)-dtype0-(111|16384)-(1|989)\]",
    # sampling: neg_inf x temperature_arr x temperature x distribution x vocab x batch
    "test_softmax": rf"\[(True|False)-(True|False)-(1\.0|0\.1)-{_DIST}-(111|128256)-(1|989)\]",
    # top_k_mask_logits: dtype x neginf x distribution x k x vocab x batch
    "test_top_k_mask_logits": rf"\[dtype[02]-(True|False)-{_DIST}-(10|500)-(111|128256)-(1|989)\]",
    # top_k_renorm_probs: dtype x distribution x k x vocab x batch
    "test_top_k_renorm_probs": rf"\[dtype[02]-{_DIST}-(10|500)-(111|128256)-(1|989)\]",
    # Loop-heavy frequency checks (5000 / 1000 launches per case).
    "test_sampling": r"\[(111|128256)-(1|989)\]",
    "test_sampling_from_logits": r"\[(111|128256)-(1|989)\]",
    "test_top_p_sampling": r"\[(0\.1|0\.9)-(111|128256)-(1|989)\]",
    "test_top_k_sampling": r"\[(10|500)-(111|128256)-(1|989)\]",
    "test_top_k_sampling_with_variable_k": r"\[(10|500)-(111|128256)-(1|989)\]",
    "test_min_p_sampling": r"\[(0\.05|1)-(111|128256)-(1|989)\]",
    "test_top_k_top_p_joint_sampling_from_probs": r"\[(0\.1|0\.5)-(111|128256)-(1|989)\]",
    "test_top_k_top_p_joint_sampling_from_logits": r"\[(0\.1|0\.5)-(111|128256)-(1|989)\]",
}

TRIM = {name: re.compile(pattern) for name, pattern in _RULES.items()}


def select(files):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "--continue-on-collection-errors",
            *files,
        ],
        capture_output=True,
        text=True,
    )
    keep = []
    kept = set()
    for line in proc.stdout.splitlines():
        node_id = line.strip()
        if "::" not in node_id:
            continue
        func = node_id.rsplit("::", 1)[-1].split("[", 1)[0]
        if func.startswith(BLACKWELL_PREFIXES):
            continue
        rule = TRIM.get(func)
        if rule is not None:
            if "[" not in node_id or not rule.fullmatch(node_id[node_id.index("[") :]):
                continue
            kept.add(func)
        keep.append(node_id)
    # A rule that matches nothing means upstream renamed the parameters and the
    # boundary coverage silently vanished; surface it instead of running empty.
    for func, rule in _RULES.items():
        if func not in kept:
            print(
                f"zhenwu_test_select: warning: no case kept for {func} ({rule})",
                file=sys.stderr,
            )
    return keep


def main():
    files = sys.argv[1:]
    if not files:
        sys.exit("usage: zhenwu_test_select.py <test files...>")
    keep = select(files)
    if not keep:
        sys.exit("zhenwu_test_select: pytest collection produced no node IDs")
    sys.stdout.write("\n".join(keep) + "\n")


if __name__ == "__main__":
    main()