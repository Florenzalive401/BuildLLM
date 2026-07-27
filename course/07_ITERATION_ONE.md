# Train Iteration 1 on CPU

## What you will learn

In this lesson you will verify the complete training path, run the first meaningful CPU learning experiment, inspect its evidence, and decide whether the result is healthy enough to evaluate. Iteration 1 proves that the same framework used by the larger models works from corpus to checkpoint.

## Where you are in the build

```text
Repository
-> Python environment
-> downloaded source documents
-> cleaned 50M corpus
-> trained 50M tokenizer
-> training and validation token files
-> Iteration 1 model configuration
-> TRAINING RUN <- you are here
-> checkpoint
-> generated samples
-> model comparison
```

This is the first lesson that changes model weights. You will start with a bounded pipeline verification, then run a larger learning experiment in a separate checkpoint directory.

## Before you begin

Close memory-heavy applications when practical, connect a laptop to power, and make sure the computer will not sleep during training. Run every command from the repository root with the virtual environment activated.

The pipeline verification answers, “Can every stage complete?” It does not answer, “Is this a capable language model?” The learning experiment processes more examples and epochs so you can observe a real loss trend and early language behavior.

Choose three evaluation prompts now and do not change them after seeing results:

```text
The purpose of a firewall is
In the history of computing,
Water changes from a liquid to
```

## Files you should already have

```text
configs/models/gpt_first_cpu.json
data/processed/training_corpus_learning_50m.jsonl
data/reports/training_corpus_learning_50m_report.json
tokenizer/learning_50m_tokenizer.json
data/tokens/learning_50m/train_tokens.pt
data/tokens/learning_50m/validation_tokens.pt
data/tokens/learning_50m/encoding_report.json
```

If any file is missing, return to [Build the corpus](03_BUILD_THE_CORPUS.md) or [Train the tokenizer and encode the dataset](04_TOKENIZER_AND_DATASET.md). Do not substitute artifacts from a different corpus.

## Files this lesson will create

The verification run writes checkpoints under:

```text
checkpoints/iteration_1_pipeline_verification
```

The learning experiment writes checkpoints under:

```text
checkpoints/iteration_1_learning_50m
```

Depending on completion and validation improvement, a directory can contain numbered `checkpoint_step_<global_step>.pt` files and `best_checkpoint.pt`. Last-state and interrupted saves use numbered paths and record their save reason in metadata. Use the exact path printed by the trainer.

## Key ideas in plain language

### Why start with a pipeline verification

A short bounded run checks that the token tensors load, input and target windows align, the forward pass produces logits, backpropagation produces gradients, validation runs, and a checkpoint can be saved and loaded. Finding a path or shape problem here is much cheaper than finding it hours into a larger experiment.

### Why the learning run is separate

The learning run uses 20,000 training examples, 1,000 validation examples, and three epochs. Its checkpoint directory is separate so a quick verification checkpoint cannot be mistaken for the learning result or selected by `--resume`.

### What Iteration 1 can and cannot prove

Iteration 1 has 8,092,800 parameters, a 128-token context, four transformer blocks, and a default CPU batch size of four. It is large enough to exercise real transformer training and show early statistical learning, but it is not expected to produce consistently factual, coherent, or safe answers.

## Run the lab

### Step 1: inspect the model

PowerShell:

```powershell
python inspect_model.py `
  --model-config configs/models/gpt_first_cpu.json
```

Linux and macOS:

```bash
python inspect_model.py \
  --model-config configs/models/gpt_first_cpu.json
```

Confirm the report shows 8,092,800 parameters, four layers, an embedding dimension of 192, six attention heads, a head dimension of 32, a feed-forward dimension of 768, and a sequence length of 128.

### Step 2: inspect the exact training command

PowerShell:

```powershell
python run_lab.py `
  --iteration 1 `
  --device cpu `
  --epochs 1 `
  --training-examples 500 `
  --validation-examples 100 `
  --train-tokens data/tokens/learning_50m/train_tokens.pt `
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --checkpoint-directory checkpoints/iteration_1_pipeline_verification `
  --dry-run
```

Linux and macOS:

```bash
python run_lab.py \
  --iteration 1 \
  --device cpu \
  --epochs 1 \
  --training-examples 500 \
  --validation-examples 100 \
  --train-tokens data/tokens/learning_50m/train_tokens.pt \
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt \
  --tokenizer tokenizer/learning_50m_tokenizer.json \
  --checkpoint-directory checkpoints/iteration_1_pipeline_verification \
  --dry-run
```

Read the printed `train.py` command from left to right. Confirm every input and output path before continuing.

### Step 3: run the pipeline verification

PowerShell:

```powershell
python run_lab.py `
  --iteration 1 `
  --device cpu `
  --epochs 1 `
  --training-examples 500 `
  --validation-examples 100 `
  --train-tokens data/tokens/learning_50m/train_tokens.pt `
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --checkpoint-directory checkpoints/iteration_1_pipeline_verification
```

Linux and macOS:

```bash
python run_lab.py \
  --iteration 1 \
  --device cpu \
  --epochs 1 \
  --training-examples 500 \
  --validation-examples 100 \
  --train-tokens data/tokens/learning_50m/train_tokens.pt \
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt \
  --tokenizer tokenizer/learning_50m_tokenizer.json \
  --checkpoint-directory checkpoints/iteration_1_pipeline_verification
```

### Step 4: review the verification evidence

Find the startup JSON, the epoch summary, checkpoint messages, and final JSON. Save them with your run record.

The checkpoint save messages use:

```text
Saving <name> checkpoint. Additional Ctrl+C presses are ignored until the save finishes.
Checkpoint saved: <path>
```

The run should report one completed epoch and a nonempty `verified_checkpoint`.

### Step 5: run the learning experiment

Start this command only after the verification run passes.

PowerShell:

```powershell
python run_lab.py `
  --iteration 1 `
  --device cpu `
  --epochs 3 `
  --training-examples 20000 `
  --validation-examples 1000 `
  --train-tokens data/tokens/learning_50m/train_tokens.pt `
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --checkpoint-directory checkpoints/iteration_1_learning_50m
```

Linux and macOS:

```bash
python run_lab.py \
  --iteration 1 \
  --device cpu \
  --epochs 3 \
  --training-examples 20000 \
  --validation-examples 1000 \
  --train-tokens data/tokens/learning_50m/train_tokens.pt \
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt \
  --tokenizer tokenizer/learning_50m_tokenizer.json \
  --checkpoint-directory checkpoints/iteration_1_learning_50m
```

## What the command is doing

`run_lab.py` selects the Iteration 1 model profile and passes the resolved values to the production `train.py` framework. `train.py` loads the matching tokenizer and token tensors, creates shifted windows, builds data loaders, constructs the model and AdamW optimizer, creates a warmup and cosine-decay scheduler, trains and validates for each epoch, saves checkpoints, and verifies that a saved checkpoint loads into a fresh model.

| Flag | Effect on this experiment |
| --- | --- |
| `--iteration 1` | Uses `configs/models/gpt_first_cpu.json` |
| `--device cpu` | Keeps the run on the CPU even if CUDA exists |
| `--epochs 3` | Makes three passes through the selected 20,000 examples |
| `--training-examples 20000` | Selects a bounded learning set rather than every available window |
| `--validation-examples 1000` | Selects held-out windows for validation |
| `--train-tokens` | Supplies the 50M corpus training tensor |
| `--validation-tokens` | Supplies the 50M corpus validation tensor |
| `--tokenizer` | Preserves the 50M token-to-ID mapping |
| `--checkpoint-directory` | Keeps this run isolated from all other experiments |

With a default CPU batch size of four, 20,000 training examples produce approximately 5,000 batches per epoch when the final batch behavior permits it. Use the application's printed `training_batches` and `maximum_training_steps` as the authoritative values.

## What success looks like

Pipeline verification succeeds when all of the following are true:

- the startup JSON identifies Iteration 1, CPU, FP32, and the intended checkpoint directory;
- the application completes a forward pass, backward pass, optimizer step, and validation;
- the epoch line reports finite training and validation loss;
- at least one checkpoint is saved;
- final JSON reports the completion state accurately;
- `verified_checkpoint` names a checkpoint that was loaded successfully.

The learning experiment succeeds operationally when it completes three epochs or saves a valid interrupted checkpoint after a deliberate interruption. It succeeds educationally when you can explain the loss trend, processed work, and generation evidence rather than merely pointing to a file.

Do not invent an expected loss value. Initial and final values vary with corpus contents, tokenizer training, random initialization, and machine configuration. Look for finite values and an interpretable trend across epochs.

Early generated text may contain fragments, repetition, malformed words, plausible grammar without factual meaning, or abrupt topic shifts. Those are observations about a small partially trained model, not evidence that the pipeline failed.

## Stop and check

Continue to generation only if:

- the checkpoint was verified;
- training and validation losses are finite;
- the completed or interrupted state matches what happened;
- the tokenizer and checkpoint belong to the same 50M experiment;
- you recorded `global_step`, processed work, and the checkpoint path;
- you kept the original prompts and generation settings for comparison.

Stop and investigate if loss is non-finite, a checkpoint cannot reload, the run resolved to an unintended device, the example counts are wrong, or the output directory contains checkpoints from a different experiment.

## Common problems and exact responses

| Problem | Likely cause | Exact response |
| --- | --- | --- |
| A token file is not found | The corpus was not encoded or the command was run outside the repository root | Check the current directory and confirm the exact three files listed under prerequisites. |
| Token IDs exceed the vocabulary | The tensor and tokenizer do not belong together | Return to the encoding report and use the tokenizer that created those token files. |
| The run is much longer than expected | The limits, batch size, or device differ from the course command | Read `training_examples`, `training_batches`, `maximum_training_steps`, `device`, and `precision` in startup JSON before allowing it to continue. |
| Validation loss is noisy | The validation subset is bounded and the run is short | Record every epoch, avoid conclusions from one value, and use the larger learning run before changing architecture. |
| Training loss decreases but generated text is poor | Loss aggregates many next-token decisions and the model is still small | Keep fixed prompts and sampling settings, compare several checkpoints, and do not choose only a lucky sample. |
| `Ctrl+C` was pressed | The trainer is saving recoverable state | Wait for `Checkpoint saved: <path>` and the final JSON; do not close PowerShell during the save. |
| A checkpoint directory already contains another run | Resume selection could restore incompatible state | Use a new explicit directory or deliberately resume only after verifying every experiment identity field. |

## What to record

Record both commands, model profile, corpus profile, tokenizer path, encoding report path, training and validation token counts, device, precision, batch size, training and validation example limits, epoch target, training batches, maximum steps, completed epochs, global step, processed tokens, per-epoch training loss, per-epoch validation loss, best validation loss, elapsed time, throughput, checkpoint paths, `verified_checkpoint`, interruption state, fixed prompts, and your conclusion.

Your conclusion should separate three questions:

1. Did the engineering pipeline work?
2. Did the loss evidence show learning?
3. What behavior did generation reveal?

## Under the hood

`src/dataset.py` returns equal-length input and target windows offset by one token:

```text
input : [The] [cat] [sat] [on]
target: [cat] [sat] [on] [the]
```

The transformer emits one vocabulary-sized logit vector for every input position. One 128-token window therefore supplies many next-token learning decisions, not just one decision at the end.

Trace one batch through:

1. `src/dataset.py`
2. `src/datamodule.py`
3. `src/model.py`
4. `src/loss.py`
5. `src/training/engine.py`
6. `src/training/trainer.py`
7. `src/checkpoint.py`

Write the shape at each boundary using `B` for batch, `T` for sequence length, `C` for embedding width, and `V` for vocabulary size. The landmarks are token IDs `[B,T]`, hidden states `[B,T,C]`, and logits `[B,T,V]`.

The controlled comparison path will later train Iterations 2 and 3 with these same 50M token files and tokenizer. This holds the data representation steady so model architecture is the main planned difference. The larger-data training path will use separate balanced and 800M corpora, separate tokenizers, and larger compute budgets; those runs answer a different question.

## Check your understanding

1. What does the 500-example run prove, and what does it not prove?
2. Why are the verification and learning checkpoints stored separately?
3. Why must the validation token file remain held out from optimizer updates?
4. How many target decisions does a 128-token window provide?
5. Which fields would you use to describe completed work if training were interrupted?
6. Why should you choose evaluation prompts before looking at generated results?
7. Why would a lower training loss alone be insufficient evidence to move on?

You are ready to continue when you have a verified checkpoint, finite loss evidence, a complete run record, and an explanation of what the experiment did and did not establish.

## Next lesson

Next: [Generate and evaluate text](08_GENERATION_AND_EVALUATION.md).
