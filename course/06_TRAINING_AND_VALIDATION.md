# Understand Training and Validation

## What you will learn

In this lesson you will follow one optimizer step from a batch of token IDs to an update of the model's weights. You will learn how loss, backpropagation, AdamW, learning-rate scheduling, gradient clipping, mixed precision, validation, and checkpoint decisions work together.

## Where you are in the build

```text
Repository
-> Python environment
-> downloaded source documents
-> cleaned corpus
-> trained tokenizer
-> training and validation token files
-> model configuration
-> TRAINING AND VALIDATION <- you are here
-> checkpoint
-> generated samples
-> model comparison
```

The model can already produce logits, but its initial weights do not yet represent useful language patterns. Training repeatedly measures the model's errors and makes small changes that should reduce those errors.

## Before you begin

You should have completed the tokenizer, dataset, and transformer lessons. You should know that each training example has an input sequence and a target sequence shifted forward by one token.

No model training is required in this conceptual lesson. The command below uses `--dry-run`, which prints the resolved training command without starting `train.py`.

## Files you should already have

For the recommended 50M path:

```text
configs/models/gpt_first_cpu.json
data/tokens/learning_50m/train_tokens.pt
data/tokens/learning_50m/validation_tokens.pt
tokenizer/learning_50m_tokenizer.json
data/tokens/learning_50m/encoding_report.json
```

## Files this lesson will create

The dry-run command creates no training artifacts. Add your annotated startup fields, step trace, and training-decision table to your lab notes.

## Key ideas in plain language

### Training units describe different amounts of work

| Unit | Meaning |
| --- | --- |
| Character | One text character before tokenization |
| Token | One tokenizer unit presented to the model |
| Example | One fixed-length input window and its shifted target window |
| Batch | Several examples processed together before one update |
| Optimizer step | One completed weight update |
| Epoch | One pass through the selected set of training examples |

Epochs are not equal across experiments. One epoch over 20,000 examples is far less work than one epoch over millions of examples. Record processed tokens and optimizer steps when comparing runs.

### One optimizer step

```text
load one batch
-> move input and target tensors to the selected device
-> clear gradients left by the previous step
-> run the model forward to create logits
-> calculate cross-entropy loss
-> run backpropagation to calculate gradients
-> validate and clip gradients
-> AdamW updates the weights
-> scheduler updates the learning rate
-> counters and metrics are recorded
```

If the batch contains four examples, one optimizer step uses all four examples to calculate one combined update. Increasing batch size can improve device utilization, but it also changes memory use, steps per epoch, and the statistical behavior of each update.

### Cross-entropy measures next-token error

At every valid target position, cross-entropy compares the logit for the correct next token with the logits for all other vocabulary tokens. A lower average loss means the model assigns more probability to the correct targets.

Loss is not a percentage and does not have a universal passing score. It is most useful as a trend across comparable checkpoints using the same tokenizer, dataset split, and loss calculation.

Perplexity is `exp(loss)`. It can be interpreted as an effective number of competing choices, but it must not be ranked directly across runs that use different tokenizers because the prediction units changed.

### Backpropagation tells each parameter how it affected the error

Backpropagation applies the chain rule through the model from the loss back to every trainable parameter. The resulting gradient says which direction would locally increase the loss; the optimizer moves in the opposite direction.

A gradient is not a permanent model change. The weights change only when the optimizer completes its step.

### AdamW controls the weight update

AdamW keeps moving estimates of recent gradients and squared gradients so each parameter receives an adaptive update. Weight decay discourages some weights from growing unnecessarily and is applied separately from Adam's gradient-based update.

The optimizer state must be saved for an exact training continuation. Restoring model weights without AdamW's moving estimates starts a different optimization trajectory.

### Warmup and cosine decay change the learning rate over steps

The learning rate controls update size. BuildLLM starts with warmup, which gradually raises the rate while the model and optimizer statistics settle. It then uses cosine decay to reduce the rate toward the configured minimum.

The scheduler advances after each optimizer step. Changing the batch size, selected example count, or number of epochs changes the number of steps and therefore changes the schedule.

### Gradient clipping limits an unusually large update

BuildLLM calculates the total gradient norm and clips it to the configured threshold before the optimizer step. This protects the run from an individual update becoming arbitrarily large.

Clipping is a safety control, not a cure for bad data, an unsuitable learning rate, or unstable numeric settings. Repeatedly extreme gradient norms are evidence to investigate.

### FP32, BF16, and FP16 are numeric formats

FP32 is 32-bit floating point and is the default CPU training precision. BF16 is bfloat16, a 16-bit format with an exponent range similar to FP32. FP16 is IEEE half precision and has a smaller exponent range.

On supported CUDA GPUs, BF16 or FP16 can reduce activation memory and increase Tensor Core throughput. FP16 uses gradient scaling in this application to protect small gradients from underflow. BF16 does not use that scaler. The larger models default to BF16 on CUDA where supported.

### Validation asks whether learning transfers to held-out text

Training loss is measured on examples used to update weights. Validation loss is measured on held-out examples that do not update weights. The validator switches the model to evaluation mode, disables gradient tracking, calculates token-weighted loss, and then restores the previous training mode.

If training loss falls while validation loss rises over a sustained period, the model may be fitting the training examples without improving on unseen examples. That pattern is called overfitting. A small or unrepresentative validation set can also create noisy results, so inspect several measurements rather than one point.

## Inspect the resolved training command

### PowerShell

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

### Linux and macOS

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

## What the command is doing

`run_lab.py` maps `--iteration 1` to `configs/models/gpt_first_cpu.json`, resolves the device profile, displays the profile's default batch size and precision, and assembles the exact `train.py` command. With `--dry-run`, it stops before executing that command.

| Flag | Why it is present |
| --- | --- |
| `--iteration 1` | Selects the 8.1M-parameter CPU-first model profile |
| `--device cpu` | Makes the hardware choice explicit |
| `--epochs 1` | Requests one pass over the selected examples |
| `--training-examples 500` | Bounds the training set for a quick end-to-end verification |
| `--validation-examples 100` | Bounds validation while still exercising the validation path |
| `--train-tokens` | Identifies the encoded training tensor |
| `--validation-tokens` | Identifies the held-out encoded tensor |
| `--tokenizer` | Preserves the token-ID meaning associated with this experiment |
| `--checkpoint-directory` | Isolates the experiment's saved state |
| `--dry-run` | Prints the resolved command without training |

## What success looks like

The dry run should print:

```text
Iteration: 1
Model: <profile name>
Objective: <learning objective>
Device: cpu
Profile batch size for this device: 4
Profile precision for this device: fp32
Command:
<the resolved train.py command>
```

When the command is later run without `--dry-run`, `train.py` prints a startup JSON object. Use the fields below to verify what is actually about to run:

| Startup field | What it tells you |
| --- | --- |
| `model` | Human-readable model profile name |
| `model_config` | JSON profile used to construct the architecture |
| `architecture.sequence_length` | Tokens in each training example |
| `architecture.embedding_dimension` | Width of each internal token representation |
| `architecture.layers` | Number of transformer blocks |
| `architecture.attention_heads` | Parallel attention projections per block |
| `architecture.feed_forward_dimension` | Hidden width of each block's feed-forward network |
| `device` | CPU or CUDA device receiving tensors and weights |
| `precision` | FP32, BF16, or FP16 execution mode |
| `tensor_cores_enabled` | Whether the selected CUDA path can use Tensor Core acceleration |
| `tf32_enabled` | Whether TensorFloat-32 acceleration is enabled for supported FP32 CUDA operations |
| `fused_adamw` | Whether the fused CUDA AdamW implementation is active |
| `parameters` | Total model parameter count |
| `training_examples` | Number of training windows selected |
| `validation_examples` | Number of validation windows selected |
| `training_batches` | Batches in each training epoch |
| `maximum_training_steps` | Planned optimizer steps across the run |
| `checkpoint_directory` | Destination for all checkpoint files |

During training, the epoch summary uses this form:

```text
Epoch <n>: training loss <value>, validation loss <value>, steps <value>
```

At the end, the application prints JSON fields including:

```text
training_complete
interrupted
completed_epochs
global_step
best_validation_loss
verified_checkpoint
restored_epoch
restored_checkpoint_type
```

`verified_checkpoint` identifies a checkpoint that the application loaded into a fresh model after saving. This verification catches a file that was written but cannot be restored.

## Stop and check

Before starting a real training run, compare the dry-run command with your intended experiment. The model profile, token files, tokenizer, epoch count, example limits, device, and checkpoint directory must all be correct.

Stop instead of training if:

- the tokenizer path belongs to a different corpus;
- the validation tensor is missing;
- the checkpoint directory already contains an unrelated experiment;
- CUDA was expected but the resolved device is CPU;
- the parameter count or sequence length does not match the intended iteration;
- `maximum_training_steps` is unexpectedly large for your available time.

## Common problems and exact responses

| Problem | Likely cause | Exact response |
| --- | --- | --- |
| Training and validation loss are both flat | Too few useful updates, an unsuitable learning rate, weak data, or a pipeline problem | Inspect several batches and gradients, verify the corpus and tokenizer, and increase only one budget variable at a time. |
| Training loss falls while validation loss rises | Possible overfitting or a train/validation distribution difference | Confirm the split, inspect duplicates and source balance, and compare several validation points before changing the model. |
| Loss becomes `nan` or `inf` | Numeric instability, an invalid batch, or excessive update magnitude | Stop the run, retain the logs and checkpoint, inspect precision, learning rate, gradient norms, and the triggering batch. |
| Gradient clipping happens constantly | The configured update dynamics may be unstable | Check learning rate, data quality, precision, and gradient norms instead of simply increasing the clip threshold. |
| Throughput falls during the run | Data loading, thermal throttling, storage, or device utilization changed | Record tokens per second and system utilization, then isolate whether the CPU, GPU, memory transfer, or data loader is waiting. |
| Validation loss cannot be compared with another run | Tokenizer or validation data differs | Treat it as a separate experiment and compare behavior with fixed prompts instead of ranking the raw losses. |

## What to record

For every training run, record the full command, corpus profile, tokenizer path, encoded training and validation token counts, model profile, parameter count, device, precision, batch size, epochs, training examples, validation examples, maximum steps, completed steps, processed tokens, training loss, validation loss, gradient norm, learning rate, throughput, elapsed time, checkpoint paths, interruption status, and any warnings.

Use the [run-record worksheet](RUN_RECORD_WORKSHEET.md) once it is introduced in the course support material.

## Under the hood

`src/dataset.py` creates shifted input and target windows. `src/datamodule.py` batches those windows. `src/training/engine.py` owns the mechanics of a single step and epoch. `src/training/trainer.py` owns validation cadence, checkpoint policy, early stopping, interruption, and the multi-epoch lifecycle. `src/training/scheduler.py` advances the learning rate. `src/checkpoint.py` saves and restores state.

The core step in `TrainingEngine.train_step` is:

```text
batch -> device -> autocast forward -> loss
-> backward -> unscale FP16 gradients if needed
-> validate and clip gradients
-> optimizer step -> scheduler step
-> update processed examples, tokens, and global step
```

The global step increments only after a completed optimizer update. This distinction matters during interruption: a batch that did not complete an update must not be reported as completed work.

Scaling involves separate decisions:

| Scaling control | What it changes |
| --- | --- |
| Parameter count | The model's representational capacity and weight/optimizer memory |
| Corpus size and quality | The language and topics available to learn |
| Encoded token count | The length of the usable training stream |
| Context length | How many tokens each example contains and the attention cost |
| Batch size | Examples per update, memory use, and steps per epoch |
| Optimizer steps | Number of opportunities to update weights |
| Epochs | Number of passes through the selected examples |

The controlled comparison path holds the 50M corpus and tokenizer constant across all three models so architecture is the main planned change. The larger-data training path changes the model, corpus, tokenizer, and compute budget, so it demonstrates an intended system configuration rather than a controlled architecture experiment.

## Check your understanding

1. What happens between calculating loss and updating weights?
2. Why is an optimizer step a better unit than an epoch when datasets differ?
3. Why does FP16 use gradient scaling in this application?
4. What does validation measure that training loss cannot?
5. Why can two runs with the same epoch count perform very different amounts of work?
6. Which startup fields would reveal that a supposed GPU run resolved to CPU?
7. Why must optimizer and scheduler state be restored when resuming?

You are ready to continue when you can explain one complete optimizer step and use the startup JSON to decide whether a run should proceed.

## Next lesson

Next: [Train Iteration 1 on CPU](07_ITERATION_ONE.md).
