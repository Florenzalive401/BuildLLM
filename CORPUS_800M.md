# 800M Corpus

The `configs/corpus_800m.yaml` profile builds the largest included BuildLLM corpus while retaining quality scoring, topic balancing, exact duplicate filtering, and near-duplicate filtering.

It uses the local Simple Wikipedia corpus, current RFCs, and streamed FineWeb Edu. FineWeb Edu fills the remaining accepted-character budget, so the complete remote dataset is not downloaded before processing.

## Complete preparation

```powershell
python prepare_800m_corpus.py
```

The helper performs these operations in order:

1. Builds the corpus with `configs/corpus_800m.yaml`.
2. Trains a 32,768-entry tokenizer from that corpus.
3. Encodes and splits complete documents into training and validation tokens.
4. Prints the Iteration 3 training command without executing it.

The command creates:

```text
data/processed/training_corpus_800m.jsonl
data/reports/training_corpus_800m_report.json
tokenizer/800m_tokenizer.json
data/tokens/800m/train_tokens.pt
data/tokens/800m/validation_tokens.pt
data/tokens/800m/encoding_report.json
```

Use `--skip-build`, `--skip-tokenizer`, or `--skip-encode` only when the matching artifact already exists and belongs to this exact 800M experiment.

For the full learner-oriented explanation, success gates, resource planning, resume procedure, and troubleshooting responses, follow [Plan and train Iteration 3](course/12_ITERATION_THREE.md). Keep a copy of the [run-record worksheet](course/RUN_RECORD_WORKSHEET.md) before starting.

## Inspect before training

Review the corpus and encoding reports. Record accepted characters, source contribution, topic distribution, rejected documents, duplicate counts, training tokens, validation tokens, and split seed.

The YAML limit is measured in cleaned characters. Model training is measured in tokens and optimizer steps. Use the encoding report and trainer output when estimating or comparing the actual training budget.

## Start a fresh 18-epoch training run

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

Do not add `--resume` for the first run. This creates a fresh model, optimizer, and learning-rate schedule for the 800M experiment.

## Resume after an interruption

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

Keep the epoch target, model iteration, tokenizer, token files, batch settings, and checkpoint directory unchanged when resuming.
