# Understand Checkpoints and Recovery

## What you will learn

In this lesson you will learn what BuildLLM saves, how atomic checkpoint writes reduce corruption risk, how `--resume` selects and restores state, what is not restored inside an interrupted epoch, and how to perform a safe recovery exercise.

## Where you are in the build

```text
training step
-> updated model and optimizer state
-> periodic, best, last-state, or interrupted save request
-> atomic checkpoint file
-> verified reload
-> RESUME AND CONTINUE <- you are here
```

A long run is only useful if progress can survive an interruption. Recovery is part of the training design, not an optional convenience added afterward.

## Before you begin

Perform the recovery exercise only in an experiment directory that you can recreate. Never practice interruption against the only copy of a valuable long-running checkpoint.

When you press `Ctrl+C`, press it once and wait. The application protects the checkpoint write from additional interrupts, but closing the terminal or losing power can still prevent the save from finishing.

## Files you should already have

Use the completed or restartable Iteration 1 experiment:

```text
tokenizer/learning_50m_tokenizer.json
data/tokens/learning_50m/train_tokens.pt
data/tokens/learning_50m/validation_tokens.pt
checkpoints/iteration_1_learning_50m
```

## Files this lesson will create

Checkpoint files use these paths:

```text
checkpoints/<experiment>/checkpoint_step_<global_step>.pt
checkpoints/<experiment>/best_checkpoint.pt
```

The trainer requests periodic, best, last-state, and interrupted saves. Their `checkpoint_name` and `checkpoint_type` are stored in metadata, while the checkpoint manager writes numbered step files. A best save also writes `best_checkpoint.pt`. Do not expect separate files named `last_checkpoint.pt` or `interrupted_checkpoint.pt`; use the path printed by `Checkpoint saved:`.

## Key ideas in plain language

### A training checkpoint is more than model weights

```text
format_version
model_state_dict
optimizer_state_dict
scheduler_state_dict
training_state
model_config
metadata
```

| Component | Why it is needed |
| --- | --- |
| Model state dictionary | Contains the learned weights used for training and generation |
| Optimizer state dictionary | Preserves AdamW's moving gradient estimates |
| Scheduler state dictionary | Preserves the current warmup or cosine-decay position |
| Training state | Preserves epoch, global step, losses, best validation loss, learning rate, elapsed work, examples, and tokens |
| Model configuration | Reconstructs and validates the architecture |
| Metadata | Records checkpoint type, artifact paths, profile, and save reason |

Model weights alone are enough for inference. They are not enough to continue the same optimization process.

### Periodic, best, last-state, and interrupted describe why a save occurred

Periodic saves occur at the configured epoch frequency and support rollback. Best saves preserve the lowest validation-loss state in `best_checkpoint.pt`. A last-state save is requested when the trainer exits normally or after an interruption. An interrupted save is requested after a controlled `Ctrl+C`.

Several save reasons can occur at the same global step, so the numbered path can be rewritten atomically with newer metadata at that step. Trust the printed path and metadata rather than inferring save reason only from the filename.

### Atomic writing protects the final filename

The checkpoint manager writes a `.tmp` file in the destination directory and then replaces the final path. If serialization fails before replacement, the previous completed final file is not presented as a partially written checkpoint.

Atomic replacement does not protect against full disks, failed storage, loss of the entire directory, or synchronization software copying an incomplete temporary file. Valuable checkpoints need a second storage location and a documented backup policy.

### Resume reconstructs before it restores

```text
same CLI and model profile
-> construct model, optimizer, and scheduler
-> choose latest numbered checkpoint or explicit file
-> load state dictionaries
-> compare checkpoint model configuration with requested configuration
-> report restored epoch, global step, and checkpoint type
-> continue training
```

`--resume` by itself means the latest numbered checkpoint in the selected checkpoint directory. `--resume <path>` selects a specific checkpoint file.

### Resume does not restore every mid-epoch detail

The checkpoint restores weights, optimizer, scheduler, counters, and tracked metrics. It does not persist the exact shuffled sampler permutation and cursor, all Python/NumPy/CPU/CUDA random-number-generator states, or data-loader worker random state.

An incomplete epoch restarts its data iteration. The run remains recoverable, but its later batches will not be bit-for-bit identical to an uninterrupted run.

## Run the recovery lab

Use a disposable directory so the exercise cannot disturb the learning checkpoint.

### Step 1: start a bounded three-epoch run

PowerShell:

```powershell
python run_lab.py `
  --iteration 1 `
  --device cpu `
  --epochs 3 `
  --training-examples 2000 `
  --validation-examples 200 `
  --train-tokens data/tokens/learning_50m/train_tokens.pt `
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --checkpoint-directory checkpoints/recovery_exercise_iteration_1
```

Linux and macOS:

```bash
python run_lab.py \
  --iteration 1 \
  --device cpu \
  --epochs 3 \
  --training-examples 2000 \
  --validation-examples 200 \
  --train-tokens data/tokens/learning_50m/train_tokens.pt \
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt \
  --tokenizer tokenizer/learning_50m_tokenizer.json \
  --checkpoint-directory checkpoints/recovery_exercise_iteration_1
```

### Step 2: interrupt safely

After several completed optimizer steps, press `Ctrl+C` once. Wait for:

```text
Interrupt received. Saving the current training state. Do not close PowerShell.
Saving interrupted checkpoint. Additional Ctrl+C presses are ignored until the save finishes.
Checkpoint saved: <path>
```

Also wait for the final JSON. Record `interrupted`, `global_step`, `verified_checkpoint`, `restored_epoch`, and `restored_checkpoint_type`.

### Step 3: resume with an identical command

PowerShell:

```powershell
python run_lab.py `
  --iteration 1 `
  --device cpu `
  --epochs 3 `
  --training-examples 2000 `
  --validation-examples 200 `
  --train-tokens data/tokens/learning_50m/train_tokens.pt `
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --checkpoint-directory checkpoints/recovery_exercise_iteration_1 `
  --resume
```

Linux and macOS use the same arguments with backslash line continuations and `--resume` on the final line.

The resume report should include:

```text
resumed_from
restored_epoch
restored_global_step
checkpoint_type
```

### Step 4: generate from the recovered result

Use the `verified_checkpoint` printed at completion and the same 50M tokenizer. Do not assume `best_checkpoint.pt` exists if no completed validation established a best checkpoint before interruption.

## What the command is doing

The fresh command creates all training components and saves state after the interruption. The resume command creates compatible components again, loads the latest numbered checkpoint, verifies that the stored `ModelConfig` exactly matches the requested model configuration, restores optimizer and scheduler state, and continues toward the original epoch target.

## What success looks like

The exercise succeeds when:

- the interrupted save prints a completed path;
- final JSON accurately reports `interrupted: true`;
- `--resume` reports the selected checkpoint and restored counters;
- the restored global step is not reset to zero;
- training continues without a model-configuration mismatch;
- a later checkpoint is saved and verified;
- you can explain that the incomplete epoch's data position was not restored exactly.

## Stop and check

Before resuming a valuable run, confirm:

- same `--iteration` and model profile;
- same tokenizer file;
- same training and validation token files;
- same epoch target;
- same batch size, precision, workers, and architecture settings;
- same checkpoint directory;
- enough free disk space for another save;
- the chosen checkpoint belongs to this experiment;
- a backup exists for irreplaceable progress.

Do not use resume to switch from the 50M corpus to the balanced or 800M corpus. That is a new training experiment, not recovery.

## Common problems and exact responses

| Problem | Likely cause | Exact response |
| --- | --- | --- |
| `no checkpoint is available to resume` | The selected directory has no numbered checkpoint | Inspect the directory and use the correct experiment path or an explicit existing checkpoint. |
| Resume configuration mismatch | Model shape or model-profile setting changed | Restore the original iteration and settings; start a new directory for a different architecture. |
| Best checkpoint is absent | No validation improvement was saved before interruption | Use the verified numbered checkpoint printed by the trainer. |
| Latest checkpoint is not the one intended | The directory contains a later step from another attempt | Supply `--resume <exact checkpoint path>` after verifying its metadata and run identity. |
| Disk fills during saving | Checkpoint and temporary copy require additional space | Free verified expendable storage outside the active run, preserve existing checkpoints, and retry only with adequate headroom. |
| User presses `Ctrl+C` repeatedly | The terminal appears paused during serialization | Stop pressing keys and wait for `Checkpoint saved:`; checkpoint writes can take time. |
| Resumed run does not reproduce exact later batches | Mid-epoch sampler and RNG position are not saved | Report this limitation; compare recoverability rather than claiming bit-for-bit continuation. |

## What to record

Record original command, interruption time, last visible completed step, save messages, checkpoint path, file size, backup location, final interrupted JSON, resume command, `resumed_from`, restored epoch, restored global step, checkpoint type, new completed steps, loss after resume, and the mid-epoch reproducibility limitation.

## Under the hood

Read `src/training/checkpoint.py` for atomic save, retention, latest selection, load validation, and checkpoint contents. Read `src/training/trainer.py` for the save reasons and `KeyboardInterrupt` handling. Read `train.py` for protected interrupts, metadata, configuration matching, resume reporting, and fresh-model verification.

The manager retains a limited number of numbered checkpoints and removes older numbered files after successful newer saves. `best_checkpoint.pt` is maintained separately. Backup policy must account for retention so a required rollback point is not removed automatically.

Resume examples for the established experiment identities are:

| Experiment | Tokenizer | Token directory | Checkpoint directory |
| --- | --- | --- | --- |
| Iteration 1 controlled comparison | `tokenizer/learning_50m_tokenizer.json` | `data/tokens/learning_50m` | `checkpoints/iteration_1_learning_50m` |
| Iteration 2 controlled comparison | `tokenizer/learning_50m_tokenizer.json` | `data/tokens/learning_50m` | `checkpoints/iteration_2_learning_50m` |
| Iteration 2 larger-data path | `tokenizer/balanced_tokenizer.json` | `data/tokens/balanced` | `checkpoints/iteration_2_balanced` |
| Iteration 3 controlled comparison | `tokenizer/learning_50m_tokenizer.json` | `data/tokens/learning_50m` | `checkpoints/iteration_3_learning_50m` |
| Iteration 3 larger-data path | `tokenizer/800m_tokenizer.json` | `data/tokens/800m` | `checkpoints/iteration_3_800m` |

## Check your understanding

1. Why are model weights alone insufficient for an exact optimizer continuation?
2. What protection does atomic replacement provide?
3. What does `--resume` select when no path follows it?
4. Why can several save reasons share one numbered step filename?
5. What is restored after interruption?
6. What mid-epoch information is not restored?
7. Why is switching tokenizers during resume invalid?

You are ready to continue when you can safely interrupt, identify the saved state, resume with an identical experiment contract, and describe the recovery limitation accurately.

## Next lesson

Next: [Plan and train Iteration 3](12_ITERATION_THREE.md).
