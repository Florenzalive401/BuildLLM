# BuildLLM GPT 100M Training

The included `configs/models/gpt_100m.json` profile creates a 98,506,080 parameter decoder-only transformer with a 32,768-token vocabulary.

Architecture:

* 12 Transformer layers
* 720 embedding dimension
* 12 attention heads
* 60 dimensions per attention head
* 2,880 feed-forward dimension
* 256-token sequence length
* Weight-tied token embeddings and output projection

## Choose the experiment

The controlled comparison run uses `configs/corpus_learning_50m.yaml`, `tokenizer/learning_50m_tokenizer.json`, and `data/tokens/learning_50m`. Use 20,000 training examples, 1,000 validation examples, and three epochs when following the course comparison so the data contract and bounded budget match the other iterations.

The larger-data run uses `configs/corpus_800m.yaml`, `tokenizer/800m_tokenizer.json`, and `data/tokens/800m`. Use it to train the 100M model on the largest included corpus. This run changes the corpus, tokenizer, token budget, and compute budget, so compare it separately from the controlled run.

## Verify the model profile

```powershell
python inspect_model.py `
  --model-config configs/models/gpt_100m.json
```

## Run the controlled 50M comparison

Inspect the resolved command with `--dry-run`, confirm the startup plan, and then remove `--dry-run` to train:

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

Use the same 50M tokenizer, example limits, epoch count, fixed prompts, and sampling settings as the controlled Iteration 1 and 2 experiments. Record the different context length, batch size, processed tokens, optimizer steps, and hardware.

## Prepare the 800M data

Run the complete preparation helper:

```powershell
python prepare_800m_corpus.py
```

The helper builds `data/processed/training_corpus_800m.jsonl`, trains `tokenizer/800m_tokenizer.json`, encodes the data into `data/tokens/800m`, and prints the training command. It does not start model training.

## Start a fresh larger-data run

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
  --checkpoint-directory checkpoints/iteration_3_800m
```

Do not add `--resume` to the first run. The CUDA profile defaults to batch size 16 and BF16. The CPU profile uses batch size 1 and FP32. If CUDA runs out of memory, reduce the batch size and record the resulting change in optimizer steps and training time.

## Resume the same run

Use the identical command and add `--resume`:

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

Training time depends on the accepted token count, GPU, precision, batch size, thermals, storage, and competing workloads. Record the encoding report, processed tokens, optimizer steps, validation trend, checkpoint paths, and fixed-prompt output instead of describing the run only by epoch count.

Follow [Plan and train Iteration 3](course/12_ITERATION_THREE.md) for the complete controlled and 800M sequences, go/no-go checks, expected output fields, recovery procedure, and generation commands.
