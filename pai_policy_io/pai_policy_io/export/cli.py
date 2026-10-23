"""pai_export_act: export a LeRobot ACT checkpoint to a Policy I/O model artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pai_policy_io.contract_signature import contract_pipeline_hash
from pai_policy_io.signature import STRATEGIES

EXIT_EQUIVALENCE = 2


def _parser() -> argparse.ArgumentParser:
    defaults = _default_options()
    parser = argparse.ArgumentParser(prog="pai_export_act", description=__doc__)
    parser.add_argument("--pretrained", required=True, help="LeRobot ACT checkpoint: Hugging Face repo id or path")
    parser.add_argument("--contract", required=True, type=Path, help="SI Rosetta contract the model is served with")
    parser.add_argument("--robot", required=True, help="robot name recorded in the signature (e.g. so_arm101)")
    parser.add_argument("--out", required=True, type=Path, help="output directory (model.onnx, policy_signature.yaml)")
    parser.add_argument("--legacy-units", choices=["deg"], default=None, help="the checkpoint was trained in degrees")
    parser.add_argument("--dataset", default=None, help="LeRobot dataset (repo id or root) for the check frames")
    parser.add_argument("--check-frames", type=int, default=defaults.check_frames, help="equivalence frames")
    parser.add_argument("--unsafe-skip-check", action="store_true", help="allow --check-frames 0 (no equivalence)")
    parser.add_argument("--execute-steps", type=int, default=defaults.execute_steps)
    parser.add_argument("--strategy", choices=STRATEGIES, default=defaults.strategy)
    parser.add_argument("--max-staleness-s", type=float, default=defaults.max_staleness_s)
    parser.add_argument("--atol", type=float, default=defaults.atol)
    parser.add_argument("--rtol", type=float, default=defaults.rtol)
    return parser


def _default_options():
    from pai_policy_io.export.lerobot_act import ExportOptions

    return ExportOptions()


def main(argv: list[str] | None = None) -> int:
    """Run the exporter; exit 2 when the ONNX model disagrees with the LeRobot reference."""
    args = _parser().parse_args(argv)
    from pai_policy_io.export.lerobot_act import ExportEquivalenceError, ExportOptions, export_act

    opts = ExportOptions(
        legacy_units=args.legacy_units,
        execute_steps=args.execute_steps,
        strategy=args.strategy,
        max_staleness_s=args.max_staleness_s,
        check_frames=args.check_frames,
        atol=args.atol,
        rtol=args.rtol,
        dataset=args.dataset,
        unsafe_skip_check=args.unsafe_skip_check,
    )
    signature, pipeline_hash = contract_pipeline_hash(args.contract, args.robot)
    try:
        out = export_act(
            args.pretrained,
            signature,
            pipeline_hash,
            args.out,
            opts,
            on_checked=lambda err: print(f"max abs error: {err:.3e} rad (atol {opts.atol:g})", flush=True),
        )
    except ExportEquivalenceError as exc:
        print(f"export refused: {exc}", file=sys.stderr)
        return EXIT_EQUIVALENCE
    print(f"pipeline hash: {pipeline_hash}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
