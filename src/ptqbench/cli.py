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

    print("\nnumerics (DESIGN.md 5a)")
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
        except Exception as exc:  # noqa: BLE001 - env-check reports every failure
            # mode as a problem rather than crashing; that is the whole point of it.
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
    from .data.calibration import CalibSpec
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
        algo=args.algo,
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        act_order=args.act_order,
        true_sequential=args.true_sequential,
        percdamp=args.percdamp,
        calib=CalibSpec(args.calib, args.nsamples, args.calib_seqlen, args.seed),
        eval_mode=args.eval_mode,
        window_batch=args.stream_window_batch,
        write=not args.no_write,
    )
    print(json.dumps(row, indent=2, default=str))
    return OK


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import matrix as M

    if args.list:
        groups = M.plan(args.experiment, filter_text=args.filter, shard=args.shard, index=args.index)
        for g in M.list_groups(groups, rerun_incomplete=args.rerun_incomplete):
            print(f"  [{g['index']:3d}] {g['label']:40s} {g['done']}/{g['total']} done  {g['quant_key']}  {','.join(g['datasets'])}")
        print(f"{len(groups)} groups, {sum(g['total'] for g in M.list_groups(groups))} rows")
        return OK
    if args.dry_run:
        groups = M.plan(args.experiment, filter_text=args.filter, shard=args.shard, index=args.index)
        from .runner import execute as X

        n_rows = sum(len(r) for _, r in groups)
        pending = sum(1 for _, rs in groups for r in rs if not X.is_done(r, rerun_incomplete=args.rerun_incomplete))
        secs, notes = M.estimate_seconds(groups)
        print(f"n_groups={len(groups)} n_rows={n_rows} pending={pending} estimated_hours={secs / 3600:.2f}")
        for n in notes:
            print(f"  note: {n}")
        return OK
    summary = M.run_matrix(
        args.experiment, filter_text=args.filter, shard=args.shard, index=args.index,
        rerun_incomplete=args.rerun_incomplete, use_quant_cache=not args.no_quant_cache,
        device_spec=args.device, deterministic=args.deterministic, progress=True,
    )
    print(
        f"done: ok={summary.ok} skipped={summary.skipped} failed={summary.failed} "
        f"already_done={summary.already_done} of {summary.n_runs} rows in {summary.seconds / 60:.1f} min"
    )
    return OK if summary.failed == 0 else FAILED_CHECK


def cmd_aggregate(args: argparse.Namespace) -> int:
    from .analysis.aggregate import aggregate

    csv, md, df = aggregate()
    complete = int(((df["status"] == "ok") & (~df["partial"].astype(bool))).sum()) if not df.empty else 0
    joined = int(df["paper_ppl"].notna().sum()) if not df.empty else 0
    print(f"{len(df)} rows ({complete} complete, {joined} joined to literature) -> {csv}\n{md}")
    return OK


def cmd_plot(args: argparse.Namespace) -> int:
    from .analysis import plots

    written = plots.plot_all(models=args.models)
    for path in written:
        print(path)
    print(f"{len(written)} figure(s) -> {paths.plots_dir()}")
    return OK


def cmd_prefetch(args: argparse.Namespace) -> int:
    from .runner import matrix as M

    M.prefetch(args.experiment)
    return OK


def cmd_cache(args: argparse.Namespace) -> int:
    from .runner import quant_cache as Q

    if args.cache_command == "ls":
        for path, size, _ in Q.entries():
            print(f"  {size / 1024**3:6.2f} GB  {path.name}")
        print(f"{len(Q.entries())} entries, {Q.total_bytes() / 1024**3:.2f} GB in {paths.quant_cache_dir()}")
        return OK
    removed = Q.gc(keep_newest=args.keep_newest, max_gb=args.max_gb)
    print(f"removed {len(removed)} entries; {Q.total_bytes() / 1024**3:.2f} GB remain")
    return OK


def cmd_wizard(args: argparse.Namespace) -> int:
    from .wizard import run_wizard

    return run_wizard(device_spec=args.device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ptq", description=__doc__)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="enable torch deterministic algorithms (slower; see DESIGN.md 5a)",
    )
    sub = parser.add_subparsers(dest="command", required=False)

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
    p_eval.add_argument("--algo", default="fp", choices=["fp", "rtn", "gptq"])
    p_eval.add_argument("--bits", type=int, default=16)
    p_eval.add_argument(
        "--group-size", type=int, default=-1, help="-1 for per-row, else 128 / 64"
    )
    p_eval.add_argument("--sym", action="store_true", help="symmetric quantization")
    p_eval.add_argument("--act-order", action="store_true", help="GPTQ act-order")
    p_eval.add_argument("--true-sequential", action="store_true", help="GPTQ true-sequential")
    p_eval.add_argument("--percdamp", type=float, default=0.01)
    p_eval.add_argument("--calib", default="c4", help="c4 | wikitext2 | pile_val")
    p_eval.add_argument("--nsamples", type=int, default=128)
    p_eval.add_argument("--calib-seqlen", type=int, default=2048)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument(
        "--eval-mode", default="auto", choices=["auto", "resident", "streamed"],
        help="auto picks streamed when free VRAM < 1.3 x model bytes (DESIGN.md 2a)",
    )
    p_eval.add_argument("--stream-window-batch", type=int, default=32)
    p_eval.add_argument("--no-write", action="store_true", help="do not write results/runs/*.json")
    p_eval.set_defaults(func=cmd_eval)

    p_run = sub.add_parser("run", help="run a YAML experiment matrix")
    p_run.add_argument("experiment")
    p_run.add_argument("--filter", default=None, help="e.g. 'algo=gptq,rtn bits=4 model=opt-125m'")
    p_run.add_argument("--shard", default=None, help="k/n: take quant-key groups with index %% n == k")
    p_run.add_argument("--index", type=int, default=None, help="run only group N")
    p_run.add_argument("--list", action="store_true")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--rerun-incomplete", action="store_true", help="retry failed and skipped rows")
    p_run.add_argument("--no-quant-cache", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_agg = sub.add_parser("aggregate", help="results/runs -> results.csv + summary.md")
    p_agg.set_defaults(func=cmd_aggregate)

    p_plot = sub.add_parser("plot", help="results.csv -> results/plots/*.png")
    p_plot.add_argument("--models", nargs="*", default=None, help="model keys to plot (default all)")
    p_plot.set_defaults(func=cmd_plot)

    p_pre = sub.add_parser("prefetch", help="download weights, datasets and calibration for an experiment")
    p_pre.add_argument("experiment")
    p_pre.set_defaults(func=cmd_prefetch)

    p_cache = sub.add_parser("cache", help="inspect or evict the quantized-weight cache")
    cache_sub = p_cache.add_subparsers(dest="cache_command", required=True)
    cache_sub.add_parser("ls")
    p_gc = cache_sub.add_parser("gc")
    p_gc.add_argument("--keep-newest", type=int, default=None)
    p_gc.add_argument("--max-gb", type=float, default=None)
    p_cache.set_defaults(func=cmd_cache)

    p_wiz = sub.add_parser("wizard", help="interactive: pick model/dataset/method/bits from menus and run")
    p_wiz.set_defaults(func=cmd_wizard)

    return parser


CACHE_WARN_GB_DEFAULT = 50.0


def warn_if_cache_is_large(threshold_gb: float | None = None) -> str | None:
    """Print (and return) a warning when the quantized-weight cache exceeds the threshold.

    The cache is never capped automatically -- it is what makes a crashed 7B run
    resumable -- but it grows by gigabytes per configuration, so every `ptq`
    invocation says so once it passes PTQ_CACHE_WARN_GB (default 50).
    """
    import os

    from .runner import quant_cache

    if threshold_gb is None:
        try:
            threshold_gb = float(os.environ.get("PTQ_CACHE_WARN_GB", CACHE_WARN_GB_DEFAULT))
        except ValueError:
            threshold_gb = CACHE_WARN_GB_DEFAULT
    try:
        size_gb = quant_cache.total_bytes() / 1024**3
        n = len(quant_cache.entries())
    except OSError:
        return None
    if size_gb <= threshold_gb:
        return None
    message = (
        f"warning: the quantized-weight cache holds {size_gb:.0f} GB in {n} entries "
        f"({paths.quant_cache_dir()}). It is not capped; `ptq cache ls` to inspect, "
        f"`ptq cache gc --max-gb N` to trim. Set PTQ_CACHE_WARN_GB to change this threshold."
    )
    print(message, file=sys.stderr)
    return message


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:  # plain `ptq` opens the wizard
        args.func = cmd_wizard
    warn_if_cache_is_large()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
