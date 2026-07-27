from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.model_profiles import ModelProfile


ITERATIONS = {
    1: Path("configs/models/gpt_first_cpu.json"),
    2: Path("configs/models/gpt_42m.json"),
    3: Path("configs/models/gpt_100m.json"),
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run one stage of the BuildLLM Learning Lab."
    )
    result.add_argument("--iteration", type=int, choices=(1, 2, 3), required=True)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--epochs", type=positive_int, default=1)
    result.add_argument("--training-examples", type=int, default=0)
    result.add_argument("--validation-examples", type=int, default=0)
    result.add_argument("--batch-size", type=positive_int, default=None)
    result.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default=None)
    result.add_argument("--workers", type=int, default=None)
    result.add_argument("--resume", nargs="?", const="latest", default=None)
    result.add_argument("--train-tokens", default="data/tokens/train_tokens.pt")
    result.add_argument("--validation-tokens", default="data/tokens/validation_tokens.pt")
    result.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    result.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory for this experiment. Defaults to "
            "checkpoints/iteration_<n>."
        ),
    )
    result.add_argument("--dry-run", action="store_true")
    return result


def checkpoint_directory_for(args: argparse.Namespace) -> Path:
    if args.checkpoint_directory is not None:
        return args.checkpoint_directory
    return Path("checkpoints") / f"iteration_{args.iteration}"


def build_training_command(
    args: argparse.Namespace,
    *,
    config_path: Path,
    device: object,
) -> list[str]:
    command = [
        sys.executable,
        "train.py",
        "--model-config",
        str(config_path),
        "--device",
        str(device),
        "--epochs",
        str(args.epochs),
        "--training-examples",
        str(args.training_examples),
        "--validation-examples",
        str(args.validation_examples),
        "--train-tokens",
        args.train_tokens,
        "--validation-tokens",
        args.validation_tokens,
        "--tokenizer",
        args.tokenizer,
        "--checkpoint-directory",
        str(checkpoint_directory_for(args)),
    ]
    if args.batch_size is not None:
        command.extend(("--batch-size", str(args.batch_size)))
    if args.precision is not None:
        command.extend(("--precision", args.precision))
    if args.workers is not None:
        command.extend(("--workers", str(args.workers)))
    if args.resume is not None:
        command.append("--resume")
        if args.resume != "latest":
            command.append(args.resume)
    return command


def main() -> int:
    args = parser().parse_args()

    from src.runtime import resolve_device

    config_path = ITERATIONS[args.iteration]
    profile = ModelProfile.load(config_path)
    device = resolve_device(args.device)
    runtime = profile.runtime_for(device.type)
    command = build_training_command(
        args,
        config_path=config_path,
        device=device,
    )

    print(f"Iteration: {args.iteration}")
    print(f"Model: {profile.name}")
    print(f"Objective: {profile.learning_objective}")
    print(f"Device: {device}")
    print(f"Profile batch size for this device: {runtime.batch_size}")
    print(f"Profile precision for this device: {runtime.precision}")
    if device.type == "cpu" and args.iteration > 1:
        print("CPU mode is supported for this architecture, but full training will be slow.")
        print("Use bounded example counts for the course exercise before attempting a full run.")
    print("Command:")
    print(" ".join(command))

    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
