# Train and Compare Iteration 2

## What you will learn

In this lesson you will train the 42.1M-parameter model in two distinct experiments. First, you will use the same 50M corpus, tokenizer, example limits, and epoch count as Iteration 1 to make a controlled comparison. Then you will build the broader balanced corpus and run the intended larger-data training path for three epochs.

## Where you are in the build

```text
Controlled comparison path
50M corpus + 50M tokenizer -> Iteration 1
                             -> ITERATION 2 <- first experiment
                             -> Iteration 3

Larger-data training path
balanced corpus + balanced tokenizer -> ITERATION 2 <- second experiment
800M corpus + 800M tokenizer --------> Iteration 3
```

The two experiments answer different questions. The controlled run asks what changes when the architecture grows while the learning data and bounded training budget remain the same. The balanced run asks what the 42M model can do with broader data and a full three-epoch GPU budget.

## Before you begin

Use a CUDA GPU for meaningful Iteration 2 training. A bounded CPU run is supported for code tracing, but a full run may be impractically slow.

Estimate runtime from a bounded measurement, confirm sufficient checkpoint storage, prevent system sleep, and choose a unique checkpoint directory. Do not build the balanced corpus or start training while another valuable run is writing to the same paths.

## Files you should already have

For the controlled comparison:

```text
configs/models/gpt_42m.json
tokenizer/learning_50m_tokenizer.json
data/tokens/learning_50m/train_tokens.pt
data/tokens/learning_50m/validation_tokens.pt
data/tokens/learning_50m/encoding_report.json
checkpoints/iteration_1_learning_50m/<verified checkpoint>
```

For the larger-data path, you also need the Simple Wikipedia source built earlier:

```text
configs/corpus_balanced.yaml
data/processed/wikipedia_simple.jsonl
```

The RFC and FineWeb Edu sources are acquired or streamed by `build_training_corpus.py` according to the YAML configuration.

## Files this lesson will create

Controlled comparison:

```text
checkpoints/iteration_2_learning_50m
```

Larger-data path:

```text
data/processed/training_corpus_balanced.jsonl
data/reports/training_corpus_balanced_report.json
tokenizer/balanced_tokenizer.json
data/tokens/balanced/train_tokens.pt
data/tokens/balanced/validation_tokens.pt
data/tokens/balanced/encoding_report.json
checkpoints/iteration_2_balanced
```

## Key ideas in plain language

### What changed in the architecture

| Setting | Iteration 1 | Iteration 2 | Why it matters |
| --- | ---: | ---: | --- |
| Parameters | 8,092,800 | 42,112,000 | More learned values can represent more patterns but require more memory and compute |
| Context length | 128 | 256 | Each example can include twice as many token positions |
| Embedding width | 192 | 512 | Each token position has a wider internal representation |
| Transformer blocks | 4 | 8 | Information is transformed through twice as many blocks |
| Attention heads | 6 | 8 | More learned attention workspaces operate in each block |
| Feed-forward width | 768 | 2,048 | Each block has a much larger nonlinear transformation |
| Intended runtime | CPU FP32 | CUDA BF16 | The larger model is designed for GPU acceleration |

Parameter count grows faster than width because attention and feed-forward layers contain large matrices whose dimensions both increase. Doubling sequence length also makes each attention head's score matrix grow from `128 × 128` to `256 × 256`, which is four times as many position-to-position scores per head.

### Why one model needs two experiments

The controlled comparison keeps the 50M tokenizer and token tensors, 20,000 training examples, 1,000 validation examples, and three epochs. Architecture is the main planned change, although context length still changes the exact windows and processed-token count.

The larger-data path uses `configs/corpus_balanced.yaml`, a separately trained tokenizer, every available encoded example, and three epochs. Because architecture, corpus, tokenization, token count, context, and compute change, any improvement must be attributed to the system configuration rather than model size alone.

### What the balanced corpus offers

The balanced profile accepts up to a one-billion-character global ceiling, with enabled targets for Simple Wikipedia, FineWeb Edu, and Request for Comments (RFC) documents. Topic weights aim to prevent one abundant subject from consuming the entire corpus, while quality thresholds, exact duplicate filtering, and near-duplicate filtering protect useful diversity.

The configured ceiling is not a promise that the report will contain exactly one billion accepted characters. Source availability, rejections, duplicate removal, per-source targets, and balancing can produce a smaller result. Record the report's actual accepted characters and document counts.

## Run the lab

## Part A: controlled 50M comparison

### Step 1: inspect the architecture

PowerShell:

```powershell
python inspect_model.py `
  --model-config configs/models/gpt_42m.json
```

Linux and macOS:

```bash
python inspect_model.py \
  --model-config configs/models/gpt_42m.json
```

Confirm 42,112,000 parameters, eight blocks, an embedding width of 512, eight heads, a head width of 64, a feed-forward width of 2,048, and a sequence length of 256.

### Step 2: inspect the controlled command

PowerShell:

```powershell
python run_lab.py `
  --iteration 2 `
  --device cuda `
  --epochs 3 `
  --training-examples 20000 `
  --validation-examples 1000 `
  --train-tokens data/tokens/learning_50m/train_tokens.pt `
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --checkpoint-directory checkpoints/iteration_2_learning_50m `
  --dry-run
```

Linux and macOS use the same arguments with backslash line continuations.

Compare the resolved startup plan with the Iteration 1 run record. The corpus and tokenizer must match; the model profile, sequence length, default batch size, precision, and parameter count should differ.

### Step 3: train the controlled experiment

Use the same command without `--dry-run`.

If CUDA is not available, you can demonstrate the architecture with `--device cpu --epochs 1 --training-examples 500 --validation-examples 100` and a new checkpoint directory. Do not present that bounded CPU demonstration as equivalent to the controlled GPU experiment.

### Step 4: generate controlled samples

PowerShell:

```powershell
python generate_text.py `
  --checkpoint checkpoints/iteration_2_learning_50m/best_checkpoint.pt `
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

Use the same five prompts and decoding values recorded for Iteration 1.

## Part B: larger-data training path

### Step 1: build the balanced corpus

PowerShell:

```powershell
python build_training_corpus.py `
  --config configs/corpus_balanced.yaml
```

Linux and macOS:

```bash
python build_training_corpus.py \
  --config configs/corpus_balanced.yaml
```

Inspect `data/reports/training_corpus_balanced_report.json`. Record actual accepted characters and documents by source, quality rejections, exact duplicates, near duplicates, topic counts, and final output path.

### Step 2: train the balanced tokenizer

PowerShell:

```powershell
python train_tokenizer.py `
  --corpus data/processed/training_corpus_balanced.jsonl `
  --format jsonl `
  --vocabulary-size 32768 `
  --output tokenizer/balanced_tokenizer.json
```

Linux and macOS:

```bash
python train_tokenizer.py \
  --corpus data/processed/training_corpus_balanced.jsonl \
  --format jsonl \
  --vocabulary-size 32768 \
  --output tokenizer/balanced_tokenizer.json
```

### Step 3: encode and split the balanced corpus

PowerShell:

```powershell
python encode_corpus.py `
  --corpus data/processed/training_corpus_balanced.jsonl `
  --format jsonl `
  --tokenizer tokenizer/balanced_tokenizer.json `
  --output-directory data/tokens/balanced
```

Linux and macOS:

```bash
python encode_corpus.py \
  --corpus data/processed/training_corpus_balanced.jsonl \
  --format jsonl \
  --tokenizer tokenizer/balanced_tokenizer.json \
  --output-directory data/tokens/balanced
```

Read `data/tokens/balanced/encoding_report.json` and record the actual training and validation document and token counts.

### Step 4: inspect and start the three-epoch run

PowerShell:

```powershell
python run_lab.py `
  --iteration 2 `
  --device cuda `
  --epochs 3 `
  --training-examples 0 `
  --validation-examples 0 `
  --train-tokens data/tokens/balanced/train_tokens.pt `
  --validation-tokens data/tokens/balanced/validation_tokens.pt `
  --tokenizer tokenizer/balanced_tokenizer.json `
  --checkpoint-directory checkpoints/iteration_2_balanced `
  --dry-run
```

`0` means use every available example. Review the planned batches and maximum steps, then remove `--dry-run` only when the runtime and storage plan are acceptable.

### Step 5: generate larger-data samples

PowerShell:

```powershell
python generate_text.py `
  --checkpoint checkpoints/iteration_2_balanced/best_checkpoint.pt `
  --tokenizer tokenizer/balanced_tokenizer.json `
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

Use all five fixed prompts. Keep the controlled and larger-data outputs in different comparison tables.

## What the commands are doing

The controlled run reuses the 50M tokenizer and tensors but constructs the 42M architecture. The balanced sequence rebuilds the entire data contract: YAML selects and limits source text, the corpus builder cleans and deduplicates it, the tokenizer learns a new token-to-ID mapping, the encoder creates a new document-level split and token tensors, and training associates the new mapping with a new checkpoint directory.

The flags `--training-examples 0` and `--validation-examples 0` mean no example limit. They do not mean zero training.

## What success looks like

For both training runs:

- startup JSON identifies the correct model, data counts, CUDA device, precision, and checkpoint directory;
- all loss values remain finite;
- every epoch runs validation;
- checkpoint messages finish successfully;
- final JSON reports the correct completion or interruption state;
- `verified_checkpoint` loads into a fresh model;
- generation uses the tokenizer belonging to that checkpoint.

For the controlled comparison, you should be able to explain the effects of architecture while acknowledging that the 256-token window changes example contents and processed-token totals. For the balanced run, you should explicitly state that model, data, tokenizer, and compute changed together.

## Stop and check

Do not start balanced training until the corpus and encoding reports have been inspected, the tokenizer verification passed, a bounded GPU run fits, and the estimated three-epoch duration and checkpoint storage are acceptable.

Do not combine the controlled and balanced validation losses in one ranked table. Their tokenizers and held-out prediction units differ.

## Common problems and exact responses

| Problem | Likely cause | Exact response |
| --- | --- | --- |
| CUDA out of memory | Iteration 2 batch and activation memory exceed the GPU | Reduce `--batch-size`, record the new value, estimate the new step count, and keep the architecture unchanged. |
| BF16 unsupported | The GPU does not support the profile's preferred precision | Use `--precision fp16` or `--precision fp32`, record it, and verify finite loss. |
| Balanced corpus is below its global ceiling | Sources exhausted targets or documents were rejected/deduplicated | Use the report's actual totals; do not fabricate the configured maximum as the achieved size. |
| Training unexpectedly uses no examples | A path is wrong or the tensor cannot form a 256-token window | Confirm the encoding report and token file, then check sequence length and example counts. |
| Balanced checkpoint is paired with the 50M tokenizer | Experiment identities were mixed | Stop generation and use `tokenizer/balanced_tokenizer.json`. |
| Iteration 2 is called better based on one output | Sampling luck replaced evaluation | Run the fixed prompt set, preserve all outputs, and combine behavior with comparable loss evidence. |

## What to record

For each run, record the complete artifact chain, model profile, parameter count, sequence length, corpus report totals, tokenizer path, encoding token counts, device, precision, batch size, workers, epochs, example limits, planned and completed steps, processed tokens, training and validation loss, throughput, elapsed time, peak memory, checkpoint paths, fixed samples, finish reasons, and limitations.

## Under the hood

Iteration 2 uses the same `GPTModel`, attention, feed-forward, loss, training engine, trainer, validator, checkpoint manager, and generator as Iteration 1. Configuration changes the dimensions and runtime defaults.

When comparing model capacity, do not collapse these controls into one word such as “scale”:

| Control | Controlled 50M run | Balanced larger-data run |
| --- | --- | --- |
| Model | Iteration 2 | Iteration 2 |
| Corpus | 50M reproducible profile | Balanced multi-source profile |
| Tokenizer | 50M tokenizer | Balanced tokenizer |
| Context | 256 | 256 |
| Examples | 20,000 training / 1,000 validation | All available |
| Epochs | 3 | 3 |
| Intended claim | Architecture evidence against similarly bounded Iteration 1 | Intended 42M system behavior with broader data |

## Check your understanding

1. Why does a wider model grow by more than the width ratio alone?
2. Why does a 256-token context make attention more expensive than a 128-token context?
3. What is held constant in the controlled comparison?
4. What changes in the balanced run?
5. Why does `--training-examples 0` mean all examples?
6. Why can validation perplexity not be ranked directly between the 50M and balanced tokenizers?
7. What evidence would justify the next experiment?

You are ready to continue when both experiments have independent run records and you can state exactly which conclusions each one supports.

## Next lesson

Next: [Learn checkpoint recovery](11_CHECKPOINTS_AND_RECOVERY.md).
