from __future__ import annotations

import argparse
import json
import random
import signal
from dataclasses import replace
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

import numpy as np
import torch

from src.config import ModelConfig, config
from src.datamodule import DataModuleConfig, LanguageModelDataModule
from src.dataset import TokenDataset
from src.loss import LanguageModelLoss
from src.model import GPTModel
from src.model_profiles import ModelProfile
from src.runtime import resolve_device, resolve_precision
from src.tokenizer import BPETokenizer
from src.training.checkpoint import CheckpointManager
from src.training.engine import TrainingEngine
from src.training.optimizer import OptimizerConfig, OptimizerFactory
from src.training.scheduler import LearningRateScheduler, SchedulerConfig
from src.training.trainer import Trainer, TrainerConfig, TrainerEpochRecord
from src.training.validator import Validator
from src.training_state import TrainingState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the GPT model from pre separated token files."
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Optional JSON model profile. Explicit CLI values override it.",
    )
    parser.add_argument(
        "--train-tokens",
        type=Path,
        default=Path("data/tokens/train_tokens.pt"),
    )
    parser.add_argument(
        "--validation-tokens",
        type=Path,
        default=Path("data/tokens/validation_tokens.pt"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizer/tokenizer.json"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("checkpoints/default_run"),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument(
        "--training-examples",
        type=int,
        default=0,
        help="Maximum training examples. Use 0 for the full dataset.",
    )
    parser.add_argument(
        "--validation-examples",
        type=int,
        default=0,
        help="Maximum validation examples. Use 0 for the full dataset.",
    )
    parser.add_argument("--embedding-dimension", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--attention-heads", type=int, default=None)
    parser.add_argument("--feed-forward-dimension", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from the latest checkpoint or from a specified checkpoint file.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16", "fp16"),
        default=None,
        help="Training precision. Profile value is used when supplied; otherwise auto.",
    )
    return parser.parse_args()



def apply_model_profile(
    args: argparse.Namespace,
    profile: ModelProfile | None,
    device: torch.device,
) -> None:
    defaults = {
        "batch_size": 8,
        "sequence_length": 64,
        "embedding_dimension": 128,
        "layers": 2,
        "attention_heads": 4,
        "feed_forward_dimension": 512,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "precision": "auto",
        "workers": 0,
    }
    runtime = profile.runtime_for(device.type) if profile else None
    profile_values = {
        "batch_size": runtime.batch_size if runtime else defaults["batch_size"],
        "sequence_length": profile.sequence_length if profile else defaults["sequence_length"],
        "embedding_dimension": profile.embedding_dimension if profile else defaults["embedding_dimension"],
        "layers": profile.layers if profile else defaults["layers"],
        "attention_heads": profile.attention_heads if profile else defaults["attention_heads"],
        "feed_forward_dimension": profile.feed_forward_dimension if profile else defaults["feed_forward_dimension"],
        "learning_rate": profile.learning_rate if profile else defaults["learning_rate"],
        "weight_decay": profile.weight_decay if profile else defaults["weight_decay"],
        "precision": runtime.precision if runtime else defaults["precision"],
        "workers": runtime.workers if runtime else defaults["workers"],
    }
    for field_name, value in profile_values.items():
        if getattr(args, field_name) is None:
            setattr(args, field_name, value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_files(args: argparse.Namespace) -> None:
    for path in (
        args.train_tokens,
        args.validation_tokens,
        args.tokenizer,
    ):
        if not path.exists():
            raise FileNotFoundError(f"required file does not exist: {path}")


def create_model_config(
    args: argparse.Namespace,
    tokenizer: BPETokenizer,
    device: torch.device,
) -> ModelConfig:
    model_config = replace(
        config,
        vocabulary_size=tokenizer.vocabulary_size,
        maximum_sequence_length=args.sequence_length,
        embedding_dimension=args.embedding_dimension,
        number_of_layers=args.layers,
        number_of_attention_heads=args.attention_heads,
        feed_forward_dimension=args.feed_forward_dimension,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        device=str(device),
        checkpoint_directory=str(args.checkpoint_directory),
    )
    model_config.validate()
    return model_config


class LiveTrainingProgress:
    def __init__(
        self,
        *,
        total_steps: int,
        batches_per_epoch: int,
        maximum_epochs: int,
        initial_step: int = 0,
    ) -> None:
        self.total_steps = total_steps
        self.batches_per_epoch = batches_per_epoch
        self.maximum_epochs = maximum_epochs
        self.loss_sum = 0.0
        self.loss_count = 0
        self.starting_tokens = 0
        self.progress_bar = tqdm(
            total=total_steps,
            initial=min(initial_step, total_steps),
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            mininterval=0.2,
            smoothing=0.1,
        )

    def update(self, result: Any, training_state: TrainingState) -> None:
        self.loss_sum += result.loss
        self.loss_count += 1
        current_step = min(training_state.global_step, self.total_steps)
        delta = current_step - self.progress_bar.n
        if delta > 0:
            self.progress_bar.update(delta)

        completed_zero_based = max(training_state.global_step - 1, 0)
        epoch_number = min(
            completed_zero_based // self.batches_per_epoch + 1,
            self.maximum_epochs,
        )
        batch_number = completed_zero_based % self.batches_per_epoch + 1
        epoch_percent = 100.0 * batch_number / self.batches_per_epoch
        overall_percent = 100.0 * current_step / self.total_steps
        average_loss = self.loss_sum / self.loss_count
        elapsed = max(self.progress_bar.format_dict.get("elapsed", 0.0), 1e-9)
        tokens_per_second = max(training_state.tokens_processed - self.starting_tokens, 0) / elapsed

        self.progress_bar.set_postfix_str(
            f"epoch {epoch_number}/{self.maximum_epochs} "
            f"batch {batch_number}/{self.batches_per_epoch} "
            f"epoch {epoch_percent:6.2f}% "
            f"overall {overall_percent:6.2f}% "
            f"loss {result.loss:.4f} "
            f"avg {average_loss:.4f} "
            f"lr {result.learning_rate:.2e} "
            f"tok/s {tokens_per_second:,.0f}",
            refresh=True,
        )

    def write(self, message: str) -> None:
        self.progress_bar.write(message)

    def close(self) -> None:
        self.progress_bar.close()


class ProtectedInterrupts:
    def __enter__(self) -> None:
        self.previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        signal.signal(signal.SIGINT, self.previous_handler)


def main() -> None:
    args = parse_args()
    model_profile = (
        ModelProfile.load(args.model_config)
        if args.model_config is not None
        else None
    )
    device = resolve_device(args.device)
    apply_model_profile(args, model_profile, device)
    validate_files(args)

    if args.epochs <= 0:
        raise ValueError("epochs must be greater than zero")

    precision_name, autocast_dtype = resolve_precision(args.precision, device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = BPETokenizer(args.tokenizer)
    model_config = create_model_config(args, tokenizer, device)
    seed_everything(model_config.random_seed)

    training_dataset = TokenDataset(
        args.train_tokens,
        sequence_length=model_config.maximum_sequence_length,
        stride=model_config.maximum_sequence_length,
        maximum_examples=args.training_examples or None,
    )
    validation_dataset = TokenDataset(
        args.validation_tokens,
        sequence_length=model_config.maximum_sequence_length,
        stride=model_config.maximum_sequence_length,
        maximum_examples=args.validation_examples or None,
    )

    data_module = LanguageModelDataModule(
        training_dataset,
        DataModuleConfig(
            batch_size=model_config.batch_size,
            validation_fraction=0.1,
            number_of_workers=args.workers,
            random_seed=model_config.random_seed,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            drop_last_training_batch=False,
        ),
        validation_dataset=validation_dataset,
    )
    data_module.setup()
    training_loader = data_module.training_dataloader()
    validation_loader = data_module.validation_dataloader()

    model = GPTModel(model_config).to(device)
    optimizer = OptimizerFactory.create(
        model,
        OptimizerConfig(
            learning_rate=model_config.learning_rate,
            weight_decay=model_config.weight_decay,
            fused=device.type == "cuda",
        ),
    )

    maximum_training_steps = max(1, len(training_loader) * args.epochs)
    warmup_steps = min(maximum_training_steps // 10, maximum_training_steps - 1)
    scheduler = LearningRateScheduler(
        optimizer,
        SchedulerConfig(
            scheduler_type="cosine",
            warmup_steps=warmup_steps,
            maximum_training_steps=maximum_training_steps,
            minimum_learning_rate=model_config.learning_rate * 0.1,
        ),
    )

    loss_function = LanguageModelLoss()
    validator = Validator(
        loss_function=loss_function,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    training_state = TrainingState()
    checkpoint_manager = CheckpointManager(
        checkpoint_directory=args.checkpoint_directory,
        maximum_checkpoints=5,
    )

    resumed_from: Path | None = None
    if args.resume is not None:
        resumed_from = (
            checkpoint_manager.latest_checkpoint_path()
            if args.resume == "latest"
            else Path(args.resume)
        )
        if resumed_from is None:
            raise FileNotFoundError("no checkpoint is available to resume")
        restored_state, restored_config, restored_metadata = checkpoint_manager.load(
            checkpoint_path=resumed_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        if restored_config != model_config.to_dict():
            raise ValueError(
                "the resume checkpoint model configuration does not match the requested run"
            )
        training_state = restored_state
        print(
            json.dumps(
                {
                    "resumed_from": str(resumed_from),
                    "restored_epoch": training_state.epoch,
                    "restored_global_step": training_state.global_step,
                    "checkpoint_type": restored_metadata.get("checkpoint_type"),
                },
                indent=4,
            )
        )

    progress = LiveTrainingProgress(
        total_steps=maximum_training_steps,
        batches_per_epoch=len(training_loader),
        maximum_epochs=args.epochs,
        initial_step=training_state.global_step,
    )
    progress.starting_tokens = training_state.tokens_processed
    training_engine = TrainingEngine(
        loss_function=loss_function,
        device=device,
        gradient_clip_norm=model_config.gradient_clip_norm,
        autocast_dtype=autocast_dtype,
        use_gradient_scaler=precision_name == "fp16",
        step_callbacks=(progress.update,),
    )

    def save_checkpoint(
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LearningRateScheduler,
        training_state: TrainingState,
        checkpoint_name: str,
        metadata: dict[str, Any],
    ) -> Path:
        complete_metadata = {
            **metadata,
            "checkpoint_name": checkpoint_name,
            "tokenizer_file": str(args.tokenizer),
            "training_token_file": str(args.train_tokens),
            "validation_token_file": str(args.validation_tokens),
            "model_profile": model_profile.name if model_profile else None,
            "model_config_file": str(args.model_config) if args.model_config else None,
        }
        if checkpoint_name == "interrupted":
            progress.write(
                "Interrupt received. Saving the current training state. Do not close PowerShell."
            )
        progress.write(
            f"Saving {checkpoint_name} checkpoint. Additional Ctrl+C presses are ignored until the save finishes."
        )
        with ProtectedInterrupts():
            checkpoint_path = checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                training_state=training_state,
                model_config=model_config,
                metadata=complete_metadata,
                is_best=checkpoint_name == "best",
            )
        progress.write(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def report_epoch(record: TrainerEpochRecord) -> None:
        validation_text = (
            f"{record.validation_loss:.4f}"
            if record.validation_loss is not None
            else "not run"
        )
        progress.write(
            f"Epoch {record.epoch}: "
            f"training loss {record.training_result.average_loss:.4f}, "
            f"validation loss {validation_text}, "
            f"steps {training_state.global_step}"
        )

    trainer = Trainer(
        training_engine=training_engine,
        trainer_config=TrainerConfig(
            maximum_epochs=args.epochs,
            validation_frequency=1,
            checkpoint_frequency=1,
            save_best_checkpoint=True,
            save_last_checkpoint=True,
            save_periodic_checkpoints=True,
            save_interrupted_checkpoint=True,
        ),
        validation_callable=validator.validate,
        checkpoint_save_callable=save_checkpoint,
        epoch_callbacks=(report_epoch,),
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "model": model_profile.name if model_profile else "Custom CLI model",
                "model_config": str(args.model_config) if args.model_config else None,
                "architecture": {
                    "sequence_length": model_config.maximum_sequence_length,
                    "embedding_dimension": model_config.embedding_dimension,
                    "layers": model_config.number_of_layers,
                    "attention_heads": model_config.number_of_attention_heads,
                    "feed_forward_dimension": model_config.feed_forward_dimension,
                },
                "device": str(device),
                "precision": precision_name,
                "tensor_cores_enabled": device.type == "cuda" and precision_name in ("bf16", "fp16"),
                "tf32_enabled": device.type == "cuda",
                "fused_adamw": device.type == "cuda",
                "parameters": parameter_count,
                "training_examples": len(training_dataset),
                "validation_examples": len(validation_dataset),
                "training_batches": len(training_loader),
                "maximum_training_steps": maximum_training_steps,
                "checkpoint_directory": str(args.checkpoint_directory),
            },
            indent=4,
        )
    )

    try:
        result = trainer.fit(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            training_data_loader=training_loader,
            validation_data_loader=validation_loader,
        )
    finally:
        progress.close()

    best_path = checkpoint_manager.best_checkpoint_path
    checkpoint_to_verify = (
        best_path
        if best_path.exists()
        else checkpoint_manager.latest_checkpoint_path()
    )

    if checkpoint_to_verify is None:
        raise RuntimeError("training completed without creating a checkpoint")

    restored_model = GPTModel(model_config).to(device)
    restored_state, restored_config, restored_metadata = checkpoint_manager.load(
        checkpoint_path=checkpoint_to_verify,
        model=restored_model,
        map_location=device,
    )

    if restored_config != model_config.to_dict():
        raise RuntimeError("reloaded checkpoint model configuration did not match")

    print(
        json.dumps(
            {
                "training_complete": not result.interrupted,
                "interrupted": result.interrupted,
                "completed_epochs": result.completed_epochs,
                "global_step": result.global_step,
                "best_validation_loss": result.best_validation_loss,
                "verified_checkpoint": str(checkpoint_to_verify),
                "restored_epoch": restored_state.epoch,
                "restored_checkpoint_type": restored_metadata.get(
                    "checkpoint_type"
                ),
            },
            indent=4,
        )
    )


if __name__ == "__main__":
    main()
