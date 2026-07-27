# BuildLLM Corpus Pipeline

The corpus pipeline turns multiple document sources into one versioned JSONL training corpus.

## What it does

1. Streams documents instead of loading complete remote datasets into memory.
2. Converts each source into a common document record.
3. Normalizes Unicode, control characters, and whitespace.
4. Scores each document from zero to one hundred.
5. Rejects short, malformed, low language, boilerplate heavy, and fragmented documents.
6. Removes exact duplicates after normalization.
7. Detects near duplicates with 64 bit SimHash and locality sensitive hashing.
8. Labels broad topics for reporting and quota based balancing.
9. Enforces per source size targets.
10. Writes a detailed JSON report with source, topic, rejection, and duplicate statistics.

## Included sources

The default balanced configuration includes:

* A local Wikipedia JSONL file produced by `build_wikipedia_corpus.py`
* FineWeb Edu through streaming Hugging Face datasets
* Current, non obsolete RFC text from the RFC Editor

Project Gutenberg and Stack Exchange adapters are included but disabled in the default configuration. Enable them after the first complete run and inspect their output before adding them to the tokenizer corpus.

## Recommended reproducible build

The ordered course first creates `data/processed/wikipedia_simple.jsonl`, then
builds a bounded multi-source corpus:

```powershell
python build_training_corpus.py `
  --config configs/corpus_learning_50m.yaml
```

The profile processes Simple Wikipedia, RFCs, and FineWeb Edu in that order.
Expected outputs:

```text
data/processed/training_corpus_learning_50m.jsonl
data/reports/training_corpus_learning_50m_report.json
```

For a smaller Wikipedia-only pipeline verification, use:

```powershell
python build_training_corpus.py `
  --config configs/corpus_pipeline_verification.yaml
```

This produces `data/processed/training_corpus_pipeline_verification.jsonl` and
`data/reports/training_corpus_pipeline_verification_report.json`. It verifies
the corpus machinery but is too small for evaluating model quality.

## Build the balanced corpus

```powershell
python build_training_corpus.py `
  --config configs/corpus_balanced.yaml
```

The default configuration targets one billion cleaned characters across enabled sources. Edit the YAML file to change source targets or the global limit.

It writes:

```text
data/processed/training_corpus_balanced.jsonl
data/reports/training_corpus_balanced_report.json
```

## Build the 800M corpus

```powershell
python build_training_corpus.py `
  --config configs/corpus_800m.yaml
```

The 800M configuration processes Simple Wikipedia and RFCs before FineWeb Edu
fills the remaining accepted-character budget. It writes:

```text
data/processed/training_corpus_800m.jsonl
data/reports/training_corpus_800m_report.json
```

Use the balanced corpus for the Iteration 2 larger-data run and the 800M corpus for the Iteration 3 larger-data run. Use the 50M corpus and tokenizer with all three iterations when the goal is a controlled architecture comparison.

The four permanent profiles serve different purposes:

| Profile | What it offers | Intended use |
| --- | --- | --- |
| `configs/corpus_pipeline_verification.yaml` | Small Wikipedia-only output that exercises corpus code quickly | Verify corpus mechanics, not language quality |
| `configs/corpus_learning_50m.yaml` | Bounded 50M-character blend of Simple Wikipedia, RFCs, and FineWeb Edu | Recommended reproducible course path and controlled model comparisons |
| `configs/corpus_balanced.yaml` | Broader topic-balanced multi-source corpus with enabled targets totaling 750M characters and a 1B global ceiling | Iteration 2 larger-data training for three epochs |
| `configs/corpus_800m.yaml` | Largest included bounded corpus, filled in the order Wikipedia, RFCs, then FineWeb Edu | Iteration 3 larger-data training for eighteen epochs |

## Use full English Wikipedia

Build English Wikipedia first:

```powershell
python build_wikipedia_corpus.py `
  --project enwiki `
  --output data/processed/wikipedia_en.jsonl `
  --report data/reports/wikipedia_en_report.json
```

Then change this line in `configs/corpus_balanced.yaml`:

```yaml
path: data/processed/wikipedia_en.jsonl
```

## Train a new tokenizer

A corpus change requires a new tokenizer and a new model run.

```powershell
python train_tokenizer.py `
  --corpus data/processed/training_corpus_learning_50m.jsonl `
  --format jsonl `
  --vocabulary-size 32768 `
  --output tokenizer/learning_50m_tokenizer.json
```

## Encode the corpus

```powershell
python encode_corpus.py `
  --corpus data/processed/training_corpus_learning_50m.jsonl `
  --format jsonl `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --output-directory data/tokens/learning_50m
```

## Configuration controls

`limits.maximum_characters` controls the total clean corpus size.

Each source supports `target_characters`, `maximum_documents`, and `minimum_quality_score`. The Hugging Face `shuffle_buffer` controls how many streamed records participate in randomized ordering; it does not cap the number of documents processed.

`quality.minimum_score` is the default acceptance threshold.

`deduplication.maximum_hamming_distance` controls near duplicate sensitivity. Three is conservative.

`balancing.topic_weights` establishes maximum topic shares relative to the global character target. `overflow_factor` provides limited flexibility so a source is not rejected at the exact quota boundary.

## Output format

Each JSONL line contains fields such as:

```json
{
  "text": "Document text",
  "source": "wikipedia",
  "title": "Example",
  "document_id": "123",
  "license": "CC BY SA 4.0 and GFDL",
  "quality_score": 78.4,
  "topic": "technology",
  "metadata": {
    "quality_metrics": {
      "characters": 4200,
      "words": 710
    }
  }
}
```

## Operational notes

The pipeline retains source and license metadata in every output record. Dataset licenses and terms still need to be reviewed for the specific model release and distribution plan.

Near duplicate detection is intentionally memory resident for speed. A very large multi billion document build will eventually need a disk backed index or a distributed data processing system. The current implementation is designed for the scale of this project and the available workstation.
