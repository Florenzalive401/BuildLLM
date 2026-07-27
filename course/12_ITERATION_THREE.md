# Plan and Train Iteration 3

## What you will learn

In this lesson you will work with the approximately 100M-parameter reference model as a long-running systems-engineering project. You will plan and run a controlled 50M experiment, prepare the complete 800M data path, understand the eighteen-epoch run, monitor useful evidence, interrupt and resume safely, and generate with the correct tokenizer.

## Where you are in the build

```text
Controlled comparison path
50M corpus + 50M tokenizer -> Iteration 1
                             -> Iteration 2
                             -> ITERATION 3 <- controlled experiment

Larger-data training path
balanced corpus + balanced tokenizer -> Iteration 2
800M corpus + 800M tokenizer --------> ITERATION 3 <- long-running experiment
```

Iteration 3 is the current reference implementation. It does not use a separate trainer: it combines the shared codebase with a larger model profile, BF16 CUDA execution, Tensor Core settings, fused AdamW where supported, validation, checkpoint verification, and recovery.

## Before you begin

Treat the 800M path as an operational commitment rather than a command to paste casually. You need adequate GPU memory, sustained cooling, reliable power, enough storage for corpus artifacts and multiple checkpoints, a realistic runtime estimate, and a backup plan.

Complete the Iteration 2 and checkpoint-recovery lessons first. Choose fixed prompts before training and preserve all settings used in earlier model comparisons.

## Files you should already have

For the controlled comparison:

```text
configs/models/gpt_100m.json
tokenizer/learning_50m_tokenizer.json
data/tokens/learning_50m/train_tokens.pt
data/tokens/learning_50m/validation_tokens.pt
data/tokens/learning_50m/encoding_report.json
```

For the 800M path:

```text
configs/corpus_800m.yaml
data/processed/wikipedia_simple.jsonl
prepare_800m_corpus.py
```

The 800M builder obtains RFC material and streams FineWeb Edu according to the YAML configuration. It does not download the entire remote FineWeb dataset before processing.

## Files this lesson will create

Controlled comparison:

```text
checkpoints/iteration_3_learning_50m
```

800M preparation and training:

```text
data/processed/training_corpus_800m.jsonl
data/reports/training_corpus_800m_report.json
tokenizer/800m_tokenizer.json
data/tokens/800m/train_tokens.pt
data/tokens/800m/validation_tokens.pt
data/tokens/800m/encoding_report.json
checkpoints/iteration_3_800m
```

## Key ideas in plain language

### What the 100M model changes

| Setting | Iteration 2 | Iteration 3 | Operational effect |
| --- | ---: | ---: | --- |
| Parameters | 42,112,000 | 98,506,080 | More weights, gradients, and AdamW state |
| Context length | 256 | 256 | Attention window remains constant |
| Embedding width | 512 | 720 | Wider representations and matrices |
| Blocks | 8 | 12 | More activation storage and sequential layer work |
| Attention heads | 8 | 12 | More attention projections; head width is 60 |
| Feed-forward width | 2,048 | 2,880 | Larger nonlinear transformation in every block |
| CUDA batch default | 16 | 16 | Equal default count does not imply equal memory use |
| CUDA precision | BF16 | BF16 | Intended lower-precision accelerator path |

The model has 98,506,080 parameters with the included 32,768-token vocabulary. Calling it the “100M model” is a readable approximation; record the exact count in experiments.

### Controlled and larger-data results must remain separate

The controlled run uses the 50M tokenizer, 20,000 training examples, 1,000 validation examples, and three epochs, matching the bounded course comparison. It helps show how the architecture behaves when the data representation and selected example counts are held steady.

The 800M run changes corpus content and scale, tokenizer, available token count, parameter count relative to Iteration 2, and the eighteen-epoch compute budget. It shows the intended larger system, but it cannot prove that architecture alone caused a behavioral improvement.

### Eighteen epochs are eighteen passes, not a model feature

An epoch is one pass through the selected training examples. Eighteen epochs over the 800M corpus produce far more optimizer work than three bounded epochs over 20,000 examples. Record maximum steps and processed tokens so the compute difference remains visible.

The epoch target also determines the scheduler's total planned steps. Changing it when resuming changes the experiment contract and learning-rate plan.

### Resource planning is part of model training

Before the full run, measure a bounded segment with the same model, device, precision, and intended batch size. Record processed tokens per second, optimizer steps per second, peak GPU memory, checkpoint file size, checkpoint save time, validation time, temperature, and power behavior.

Use the measured rate and the application's `maximum_training_steps` to estimate runtime. Add overhead and uncertainty. Do not publish a fixed duration as if every GPU and corpus produces the same rate.

### Monitoring should lead to decisions

Watch epoch-average training loss, validation loss, the gap between them, gradient behavior, learning rate, processed tokens, global step, throughput, GPU memory, temperature, checkpoint saves, and free storage.

One noisy batch is not a reason to restart. A non-finite loss, repeated checkpoint failure, sustained validation deterioration, thermal instability, or storage exhaustion is a reason to stop and investigate.

## Run the lab

## Part A: controlled 50M comparison

### Step 1: inspect the reference architecture

PowerShell:

```powershell
python inspect_model.py `
  --model-config configs/models/gpt_100m.json
```

Linux and macOS:

```bash
python inspect_model.py \
  --model-config configs/models/gpt_100m.json
```

Confirm 98,506,080 parameters, 12 blocks, a width of 720, 12 heads, a head width of 60, a feed-forward width of 2,880, and a sequence length of 256.

### Step 2: inspect the bounded controlled run

PowerShell:

```powershell
python run_lab.py `
  --iteration 3 `
  --device cuda `
  --epochs 3 `
  --training-examples 20000 `
  --validation-examples 1000 `
  --train-tokens data/tokens/learning_50m/train_tokens.pt `
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --checkpoint-directory checkpoints/iteration_3_learning_50m `
  --dry-run
```

Linux and macOS use the same arguments with backslash line continuations.

Review the startup plan, run a smaller bounded measurement if needed, and then remove `--dry-run` to start the controlled experiment.

### Step 3: generate controlled samples

PowerShell:

```powershell
python generate_text.py `
  --checkpoint checkpoints/iteration_3_learning_50m/best_checkpoint.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --prompt "The purpose of a firewall is" `
  --max-new-tokens 64 `
  --temperature 0.7 `
  --top-k 40 `
  --top-p 0.9 `
  --repetition-penalty 1.15 `
  --no-repeat-ngram-size 3 `
  --seed 42 `
  --device cuda
```

Use all five fixed prompts and the same decoding settings as Iterations 1 and 2.

## Part B: complete 800M preparation

### Option 1: run each stage and inspect every boundary

This is the recommended learning route because you can inspect each report before continuing.

Build the corpus.

PowerShell:

```powershell
python build_training_corpus.py `
  --config configs/corpus_800m.yaml
```

Linux and macOS:

```bash
python build_training_corpus.py \
  --config configs/corpus_800m.yaml
```

The builder processes Simple Wikipedia first, RFCs second, and FineWeb Edu last. FineWeb Edu fills the remaining accepted-character budget up to the 800M global limit. Inspect `data/reports/training_corpus_800m_report.json` before training a tokenizer.

Train the dedicated tokenizer.

PowerShell:

```powershell
python train_tokenizer.py `
  --corpus data/processed/training_corpus_800m.jsonl `
  --format jsonl `
  --vocabulary-size 32768 `
  --output tokenizer/800m_tokenizer.json
```

Linux and macOS:

```bash
python train_tokenizer.py \
  --corpus data/processed/training_corpus_800m.jsonl \
  --format jsonl \
  --vocabulary-size 32768 \
  --output tokenizer/800m_tokenizer.json
```

Encode and split by document.

PowerShell:

```powershell
python encode_corpus.py `
  --corpus data/processed/training_corpus_800m.jsonl `
  --format jsonl `
  --tokenizer tokenizer/800m_tokenizer.json `
  --output-directory data/tokens/800m
```

Linux and macOS:

```bash
python encode_corpus.py \
  --corpus data/processed/training_corpus_800m.jsonl \
  --format jsonl \
  --tokenizer tokenizer/800m_tokenizer.json \
  --output-directory data/tokens/800m
```

Inspect `data/tokens/800m/encoding_report.json`. Record `training_documents`, `validation_documents`, `training_tokens`, `validation_tokens`, `characters_encoded`, `split_seed`, `training_file`, and `validation_file`.

### Option 2: use the preparation helper

PowerShell, Linux, and macOS:

```text
python prepare_800m_corpus.py
```

The helper runs the build, tokenizer, and encoding stages with the standard 800M paths. It prints the exact eighteen-epoch `run_lab.py` command but never starts model training.

Use `--skip-build`, `--skip-tokenizer`, or `--skip-encode` only when you deliberately reuse an existing verified artifact. Skipping tokenizer training means the existing `tokenizer/800m_tokenizer.json` must be the tokenizer intended for this exact corpus.

## Part C: plan and start the eighteen-epoch run

Inspect the exact command first.

PowerShell:

```powershell
python run_lab.py `
  --iteration 3 `
  --device cuda `
  --epochs 18 `
  --training-examples 0 `
  --validation-examples 0 `
  --train-tokens data/tokens/800m/train_tokens.pt `
  --validation-tokens data/tokens/800m/validation_tokens.pt `
  --tokenizer tokenizer/800m_tokenizer.json `
  --checkpoint-directory checkpoints/iteration_3_800m `
  --dry-run
```

Linux and macOS use the same arguments with backslash line continuations.

Read `training_examples`, `validation_examples`, `training_batches`, and `maximum_training_steps`. Estimate runtime and storage. Remove `--dry-run` only after the run passes the go/no-go checklist.

### Go/no-go checklist

- Correct 800M corpus and encoding reports have been reviewed.
- The 800M tokenizer verification completed.
- The token files and tokenizer paths match.
- CUDA, BF16 or documented fallback precision, Tensor Core state, and fused AdamW state are known.
- A bounded Iteration 3 measurement fits in GPU memory.
- Runtime was estimated from measured throughput.
- Checkpoint storage has adequate free space and backup capacity.
- Power, cooling, sleep settings, and access to the machine are appropriate.
- The checkpoint directory is new for a fresh run or verified for an exact resume.
- Fixed prompts and run-record fields are prepared before training.

## Part D: resume safely

Use the identical command with `--resume`:

```powershell
python run_lab.py `
  --iteration 3 `
  --device cuda `
  --epochs 18 `
  --training-examples 0 `
  --validation-examples 0 `
  --train-tokens data/tokens/800m/train_tokens.pt `
  --validation-tokens data/tokens/800m/validation_tokens.pt `
  --tokenizer tokenizer/800m_tokenizer.json `
  --checkpoint-directory checkpoints/iteration_3_800m `
  --resume
```

Confirm `resumed_from`, `restored_epoch`, `restored_global_step`, and `checkpoint_type` before allowing the run to continue.

## Part E: generate from the 800M result

```powershell
python generate_text.py `
  --checkpoint checkpoints/iteration_3_800m/best_checkpoint.pt `
  --tokenizer tokenizer/800m_tokenizer.json `
  --prompt "The purpose of a firewall is" `
  --max-new-tokens 64 `
  --temperature 0.7 `
  --top-k 40 `
  --top-p 0.9 `
  --repetition-penalty 1.15 `
  --no-repeat-ngram-size 3 `
  --seed 42 `
  --device cuda
```

Use the exact verified numbered checkpoint if `best_checkpoint.pt` is absent.

## What the commands are doing

The controlled command constructs the 100M architecture but reuses the 50M data contract. The preparation sequence creates an entirely separate 800M corpus, tokenizer, split, and token stream. The eighteen-epoch command uses the full encoded datasets because both example limits are zero, constructs a scheduler for the planned total optimizer steps, validates every epoch, saves recoverable state, and verifies a saved checkpoint.

On CUDA, `train.py` enables supported matrix acceleration, resolves BF16/FP16/FP32, requests fused AdamW with fallback, uses autocast, applies gradient scaling only for FP16, clips gradients, and advances the scheduler after completed optimizer steps.

## What success looks like

Preparation succeeds when the corpus and encoding reports contain real accepted counts and paths, tokenizer verification passes, and all three token artifacts exist.

Training succeeds operationally when startup identity is correct, losses are finite, validation runs, checkpoint saves finish, interruption state is accurate, and `verified_checkpoint` reloads. A long run can be operationally successful even when model-quality evidence says the next experiment should change data or optimization.

Generation succeeds when the correct 800M tokenizer loads with the selected 800M checkpoint and all fixed prompt outputs are retained.

## Stop and check

Stop the run and preserve evidence if:

- loss becomes `nan` or `inf`;
- checkpoint saves repeatedly fail;
- free storage is approaching the space needed for a temporary and final checkpoint;
- the GPU is thermally unstable;
- the resolved device or precision is not what was planned;
- the tokenizer or token paths are wrong;
- validation deteriorates persistently enough to trigger the experiment's review criteria.

Do not stop solely because one batch loss moves upward. Training data varies from batch to batch.

## Common problems and exact responses

| Problem | Likely cause | Exact response |
| --- | --- | --- |
| Corpus build takes a long time | Streaming, cleaning, scoring, balancing, and near-duplicate checks process substantial text | Monitor reportable progress and storage; do not start a duplicate builder against the same output. |
| 800M output is smaller than the target | Accepted sources did not fill the budget after quality and duplicate filtering | Use the actual report totals and decide whether source configuration needs a separate future experiment. |
| CUDA out of memory | Batch and 100M activation state exceed the GPU | Reduce `--batch-size`, record the new step count and schedule implications, and verify with a bounded fresh run. |
| Throughput changes after hours | Thermals, other workloads, validation, saving, or data delivery changed | Correlate throughput with temperature, GPU utilization, validation periods, and checkpoint timing. |
| Save appears to pause training | Checkpoint serialization is using CPU, memory, and storage | Wait for `Checkpoint saved:` and record save duration; do not interrupt the write. |
| Resume restarts work within an epoch | Exact shuffled sampler cursor is not checkpointed | Accept and document the incomplete-epoch replay; do not claim bit-for-bit continuity. |
| Best validation loss improves but samples remain weak | Aggregate next-token prediction improved without reliable behavior on selected prompts | Keep all evidence, examine training budget and corpus coverage, and avoid selecting one lucky sample. |

## What to record

Record hardware and software environment, complete 800M artifact chain, corpus report totals, tokenizer path and vocabulary, encoding report counts, model profile, parameter count, sequence length, device, precision, Tensor Core/TF32/fused AdamW fields, batch size, workers, epoch target, maximum steps, completed steps, processed tokens, per-epoch losses, learning rate, gradient norms, throughput, temperatures, peak memory, validation and checkpoint durations, checkpoint paths, backups, interruptions, resume state, samples, conclusions, and limitations.

## Under the hood

Iteration 3 remains the same training framework as Iterations 1 and 2. Its reliability comes from the interaction of the model profile, runtime resolution, mixed precision, optimizer factory, scheduler, validator, trainer lifecycle, atomic checkpoint manager, protected interrupt handling, and final fresh-model reload.

The profile, tokenizer, token files, epoch target, runtime settings, and checkpoint directory form the run identity. Keep them together:

| Training path | Corpus YAML | Tokenizer | Token directory | Epochs | Checkpoint directory |
| --- | --- | --- | --- | ---: | --- |
| Controlled comparison | `configs/corpus_learning_50m.yaml` | `tokenizer/learning_50m_tokenizer.json` | `data/tokens/learning_50m` | 3 bounded | `checkpoints/iteration_3_learning_50m` |
| Iteration 2 larger-data predecessor | `configs/corpus_balanced.yaml` | `tokenizer/balanced_tokenizer.json` | `data/tokens/balanced` | 3 full | `checkpoints/iteration_2_balanced` |
| Iteration 3 larger-data path | `configs/corpus_800m.yaml` | `tokenizer/800m_tokenizer.json` | `data/tokens/800m` | 18 full | `checkpoints/iteration_3_800m` |

## Check your understanding

1. Why is the exact model called approximately 100M parameters?
2. What does the controlled Iteration 3 experiment hold constant?
3. What changes in the 800M experiment?
4. Why is an eighteen-epoch count insufficient to describe completed work?
5. What measurements are needed to estimate runtime?
6. Why can a checkpoint be operationally valid even if model quality is disappointing?
7. What conditions justify stopping a long run?

You are ready to continue when you can reproduce the complete artifact chain, defend the go/no-go decision, recover the run safely, and keep controlled and larger-data conclusions separate.

## Next lesson

Next: [Compare all three models](13_COMPARE_THE_MODELS.md).
