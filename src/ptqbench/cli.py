"""ptq command line. M0 ships `env-check` and a minimal `eval`."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import paths, provenance

# Exit codes
OK = 0
FAILED_CHECK = 1


def _fmt(label: str, value: Any, *, indent: int = 2) -> str:
    return f"{' ' * indent}{label:<28} {value}"


def cmd_env_check(args: argparse.Namespace) -> int:
    """Resolve and print every path, version and numerics flag; fail on anything wrong."""
    import torch

    from . import device as D

    problems: list[str] = []
    warnings: list[str] = []

    print("ptq env-check")
    print("\npaths")
    resolved = paths.describe()
    for key, value in resolved.items():
        print(_fmt(key, value))

    # sys.prefix must be the project venv, not a stray interpreter.
    prefix = sys.prefix
    root = str(paths.repo_root())
    print(_fmt("sys.prefix", prefix))
    if not prefix.startswith(root):
        problems.append(f"sys.prefix {prefix!r} is not inside the repo root {root!r}")

    print("\nplatform")
    for key, value in provenance.platform_info().items():
        print(_fmt(key, value))
    print(_fmt("git_sha", provenance.git_sha()))

    print("\npackages")
    for name, version in provenance.package_versions().items():
        print(_fmt(name, version))

    print("\ndevice")
    dev = D.resolve(args.device)
    info = D.describe(dev)
    for key, value in info.items():
        print(_fmt(key, value))

    if args.require_cuda and not torch.cuda.is_available():
        problems.append(
            "torch.cuda.is_available() is False. DISABLE_CUDA=1 is an hqq build flag and "
            "must not affect torch -- check the torch build is +cu130, not +cpu."
        )
    if torch.cuda.is_available() and "+cu" not in torch.__version__:
        warnings.append(f"CUDA is available but torch is {torch.__version__}")

    print("\nnumerics (PLAN.md 5a)")
    before = D.read_fp32_precision()
    print(_fmt("fp32_precision (before)", before))
    state = D.lock_numerics(deterministic=args.deterministic)
    print(_fmt("api", state.api))
    print(_fmt("fp32_precision (after)", state.knobs))
    print(_fmt("float32_matmul_precision", state.float32_matmul_precision))
    print(_fmt("fp16_reduced_reduction", state.fp16_reduced_reduction))
    print(_fmt("cudnn.benchmark", state.cudnn_benchmark))
    print(_fmt("deterministic", state.deterministic))
    for note in state.notes:
        warnings.append(f"numerics: {note}")
    if D.tf32_is_live():
        problems.append("TF32 is still live after lock_numerics(); perplexity gates are unreachable")

    if dev.type == "cuda":
        print("\nsanity")
        try:
            x = torch.randn(512, 512, device=dev, dtype=torch.float16)
            _ = (x @ x).float().sum().item()
            print(_fmt("fp16 matmul", "ok"))
            if torch.cuda.is_bf16_supported():
                y = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
                _ = (y @ y).float().sum().item()
                print(_fmt("bf16 matmul", "ok"))
        except Exception as exc:  # pragma: no cover
            problems.append(f"GPU sanity matmul failed: {exc}")

    if args.json:
        payload = {
            "paths": resolved,
            "sys_prefix": prefix,
            "platform": provenance.platform_info(),
            "packages": provenance.package_versions(),
            "device": info,
            "numerics": state.as_row_fields(),
            "problems": problems,
            "warnings": warnings,
        }
        print("\n" + json.dumps(payload, indent=2, default=str))

    print()
    for w in warnings:
        print(f"  warning: {w}")
    if problems:
        for p in problems:
            print(f"  FAIL: {p}")
        print(f"\nenv-check FAILED ({len(problems)} problem(s))")
        return FAILED_CHECK
    print("env-check OK")
    return OK


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval.run_eval import run_single_eval

    row = run_single_eval(
        model_id=args.model,
        dataset_key=args.dataset,
        device_spec=args.device,
        seqlen=args.seqlen,
        max_windows=args.max_windows,
        ce_chunk=args.ce_chunk,
        dtype_override=args.dtype,
        deterministic=args.deterministic,
        write=not args.no_write,
    )
    print(json.dumps(row, indent=2, default=str))
    return OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ptq", description=__doc__)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="enable torch deterministic algorithms (slower; see PLAN.md 5a)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_env = sub.add_parser("env-check", help="resolve paths, versions and numerics; fail on drift")
    p_env.add_argument("--json", action="store_true", help="also emit a JSON payload")
    p_env.add_argument(
        "--no-require-cuda",
        dest="require_cuda",
        action="store_false",
        help="do not fail when CUDA is unavailable (CI uses this)",
    )
    p_env.set_defaults(func=cmd_env_check, require_cuda=True)

    p_eval = sub.add_parser("eval", help="evaluate perplexity for one model/dataset pair")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--dataset", default="wikitext2")
    p_eval.add_argument("--seqlen", type=int, default=2048)
    p_eval.add_argument("--max-windows", type=int, default=None)
    p_eval.add_argument("--ce-chunk", type=int, default=256, help="0 disables chunking")
    p_eval.add_argument("--dtype", default=None, help="override the family dtype policy")
    p_eval.add_argument("--no-write", action="store_true", help="do not write results/runs/*.json")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
