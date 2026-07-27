# Build the Training Corpus

## What you will learn

You will combine source documents into a cleaned, filtered, deduplicated, topic-balanced JSONL corpus. You will learn what each corpus configuration controls, why the 50M profile is the recommended learning path, how to inspect the report, and when to choose the pipeline-verification, balanced, or 800M profiles.

## Where you are in the build

```text
Simple Wikipedia JSONL
RFC stream
FineWeb Edu stream
    -> cleaning and quality checks
    -> duplicate removal
    -> topic balancing
    -> training corpus  <--- you are here
    -> tokenizer
    -> encoded training and validation tokens
```

This stage decides which text reaches the tokenizer and model. A model cannot learn a pattern that is absent from the corpus, and repeated or damaged text can become repeated or damaged model behavior.

## Before you begin

Confirm:

- `.venv` is active;
- the terminal is at the repository root;
- `data/processed/wikipedia_simple.jsonl` exists and is not empty;
- `data/reports/wikipedia_simple_report.json` exists;
- you reviewed several Wikipedia records;
- the machine has network access for RFC and FineWeb Edu sources;
- the selected output and report paths do not contain valuable files from another experiment.

## Files you should already have

```text
data/processed/wikipedia_simple.jsonl
data/reports/wikipedia_simple_report.json
configs/corpus_pipeline_verification.yaml
configs/corpus_learning_50m.yaml
configs/corpus_balanced.yaml
configs/corpus_800m.yaml
```

## Files this lesson will create

The exact output depends on the selected YAML configuration:

| Purpose | Configuration | Corpus output | Report output |
| --- | --- | --- | --- |
| Small corpus-pipeline check | `configs/corpus_pipeline_verification.yaml` | `data/processed/training_corpus_pipeline_verification.jsonl` | `data/reports/training_corpus_pipeline_verification_report.json` |
| Recommended learning path | `configs/corpus_learning_50m.yaml` | `data/processed/training_corpus_learning_50m.jsonl` | `data/reports/training_corpus_learning_50m_report.json` |
| Iteration 2 larger-data path | `configs/corpus_balanced.yaml` | `data/processed/training_corpus_balanced.jsonl` | `data/reports/training_corpus_balanced_report.json` |
| Iteration 3 larger-data path | `configs/corpus_800m.yaml` | `data/processed/training_corpus_800m.jsonl` | `data/reports/training_corpus_800m_report.json` |

The course continues with the 50M output. The other profiles are available when their corresponding lessons explicitly direct you to them.

## Key idea: a corpus is a designed dataset

A corpus is not just a folder of downloaded text. It is the accepted output of source selection, cleaning, quality rules, duplicate handling, topic limits, and size budgets.

Two engineers can begin with the same source names and produce different corpora when their source snapshots, thresholds, seeds, or limits differ. Preserve the YAML, raw sources, output corpus, and report when reproducibility matters.

## The corpus pipeline

```text
source adapter
    -> common document record
    -> normalize text
    -> calculate quality metrics
    -> reject documents below thresholds
    -> calculate exact fingerprint
    -> find near-duplicate candidates
    -> assign broad topic
    -> apply source and topic limits
    -> write accepted JSONL record
    -> update report counters
```

Each source adapter converts its input into a `CorpusDocument`. This lets the later stages treat local Wikipedia, downloaded RFCs, and streamed Hugging Face documents consistently.

## Cleaning

Cleaning normalizes Unicode, control characters, and whitespace. It removes some formatting noise without trying to rewrite the meaning of a document.

Cleaning is a tradeoff. Removing too little leaves markup and boilerplate. Removing too much can damage code, tables, standards language, or meaningful punctuation. Inspect records rather than assuming the cleaner is correct for every domain.

## Quality scoring

The pipeline measures features such as:

- total characters;
- alphabetic character share;
- word structure;
- line fragmentation;
- boilerplate signals;
- source-specific minimum quality.

The global `minimum_score` is a default acceptance threshold. A source can set a different `minimum_quality_score`.

A quality score is a heuristic, not a truth label. Technical specifications and code-heavy text may look unusual compared with prose. Review accepted and rejected examples when changing thresholds.

## Exact duplicates

The pipeline normalizes the cleaned text and calculates a fingerprint. If the same fingerprint has already been accepted, the later copy is rejected as an exact duplicate.

Exact duplicate removal prevents repeated documents from receiving extra influence simply because they appeared more than once.

## Near duplicates

Documents can repeat the same content with small changes. The pipeline uses a 64-bit SimHash, locality-sensitive bands, and Hamming-distance verification to find substantially similar documents.

Near-duplicate filtering matters because:

- copied or templated pages can dominate updates;
- corpus size can grow without adding much information;
- similar copies can land on both sides of the later training/validation split;
- repeated passages can make validation results look better than they should.

The duplicate index is memory-resident. It fits the workstation-scale corpora in this repository. A multi-billion-document system would need a disk-backed or distributed design.

## Topic balancing

The pipeline assigns broad topics from keyword evidence and applies maximum shares from the YAML profile.

Topic balancing is a guardrail, not semantic understanding. It can prevent an obvious domain from consuming most of the accepted-character budget, but it cannot perfectly classify every document.

Inspect the report’s topic counts and topic-character totals. If a topic is unexpectedly absent or dominant, inspect records before changing weights.

## Source and global limits

The configurations use cleaned accepted characters as their primary size unit.

`target_characters` limits one source. `limits.maximum_characters` limits the complete corpus. `maximum_documents` can bound document count.

A source may finish below its target because:

- the source runs out of documents;
- quality rules reject documents;
- duplicate rules reject documents;
- topic limits reject documents;
- the global corpus limit is reached first.

The YAML target is not the final training-token count. Token count is produced later by the tokenizer and recorded by `encode_corpus.py`.

## Understand the four configurations

### Pipeline verification

`configs/corpus_pipeline_verification.yaml` uses up to 1,000 local Simple Wikipedia documents. It verifies corpus cleaning, filtering, duplicate detection, output, and reporting without streaming the larger remote sources.

Use it when debugging the corpus pipeline. Do not use it to judge model language quality.

### 50M learning corpus

`configs/corpus_learning_50m.yaml` is the recommended first complete build.

It requests sources in this order:

| Order | Source | Accepted-character target | Why it is included |
| ---: | --- | ---: | --- |
| 1 | Simple Wikipedia | 15,000,000 | General English and local reproducible input |
| 2 | RFCs | 10,000,000 | Technical standards and networking language |
| 3 | FineWeb Edu | 25,000,000 | Broader educational web text |

The global limit is 50,000,000 cleaned accepted characters. A fixed seed, explicit output paths, topic weights, quality thresholds, and duplicate settings make this the clearest teaching profile.

### Balanced corpus

`configs/corpus_balanced.yaml` is the Iteration 2 larger-data profile. Its global ceiling is one billion characters, while currently enabled source targets total 750 million characters.

The actual accepted size can be lower because Simple Wikipedia and RFCs may be exhausted before their requested targets and documents may be rejected. Use the report, not the configured ceiling, as the actual corpus size.

Gutenberg and Stack Exchange adapters are present but disabled. Enabling them changes the experiment, licensing review, data mix, output corpus, tokenizer, and training run.

### 800M corpus

`configs/corpus_800m.yaml` is the Iteration 3 larger-data profile. It processes Simple Wikipedia, then RFCs, then FineWeb Edu until the global 800M accepted-character budget is filled or sources stop.

This path requires substantially more network use, corpus-build time, tokenizer time, encoding memory, storage, and GPU training time than the 50M path. It is introduced again in the Iteration 3 lesson.

## Run the recommended 50M lab

### Windows PowerShell

```powershell
python build_training_corpus.py `
  --config configs/corpus_learning_50m.yaml
```

### Linux or macOS

```bash
python build_training_corpus.py \
  --config configs/corpus_learning_50m.yaml
```

## What the command is doing

`build_training_corpus.py` loads the YAML mapping, constructs the configured source adapters and pipeline, processes one document at a time, writes accepted JSONL records, and writes the final report.

RFC and FineWeb Edu data are downloaded or streamed during this command. FineWeb Edu uses a streaming shuffle buffer; it does not download the entire remote dataset before processing.

The script prints a final summary containing:

```json
{
  "corpus": "data/processed/training_corpus_learning_50m.jsonl",
  "report": "data/reports/training_corpus_learning_50m_report.json",
  "documents": "<your accepted document count>",
  "characters": "<your accepted character count>"
}
```

Your counts may differ as remote sources change.

## Inspect the report

Windows PowerShell:

```powershell
Get-Content data/reports/training_corpus_learning_50m_report.json
```

Linux or macOS:

```bash
cat data/reports/training_corpus_learning_50m_report.json
```

The report contains:

| Field | What to ask |
| --- | --- |
| `output` | Is this the corpus file you intended to build? |
| `elapsed_seconds` | How long did the build take on this machine? |
| `totals` | How many documents and characters were examined, written, rejected, or duplicated? |
| `sources` | How much accepted material came from each source? |
| `source_stop_reasons` | Did each source hit a target, exhaust data, or stop because the global limit was reached? |
| `topics` | How many accepted documents were assigned to each topic? |
| `topic_characters` | How much corpus text came from each topic? |
| `rejection_reasons` | Which quality, duplicate, or quota rules rejected documents? |
| `configuration` | Does the report preserve the YAML settings used for the build? |

## Inspect accepted records

Windows PowerShell:

```powershell
Get-Content data/processed/training_corpus_learning_50m.jsonl -TotalCount 5
```

Linux or macOS:

```bash
head -n 5 data/processed/training_corpus_learning_50m.jsonl
```

Every accepted record includes text, source, title, document identity, license, quality score, topic, and quality metrics.

Inspect documents from each source. Check readability, remaining markup, repeated boilerplate, broken code, suspiciously short records, and topic assignments.

## What success looks like

You can continue when:

- the selected corpus JSONL exists and is not empty;
- the matching report exists;
- report output paths match the files you built;
- accepted document and character counts are greater than zero;
- all expected sources appear in the report;
- source stop reasons are understandable;
- topic and rejection distributions are present;
- several accepted records are readable and preserve source information.

## Stop and check

Do not train a tokenizer from a corpus you have not inspected. If the report shows one source unexpectedly missing, nearly all documents rejected, or severe duplication, investigate before proceeding.

## Common problems and exact responses

| Problem | Likely cause | What to do |
| --- | --- | --- |
| Wikipedia source is missing | Lesson 2 output path differs or download was not completed | Build `data/processed/wikipedia_simple.jsonl` or update the YAML deliberately |
| Hugging Face download fails | Network, authentication, proxy, or remote availability | Confirm network access and organization policy, then repeat the same command |
| RFC source is slow | Many remote documents are being fetched or cached | Let the cache populate; later runs can reuse cached files |
| Corpus stops below a source target | Source exhaustion, quality rejection, duplicates, or topic limits | Read `source_stop_reasons` and rejection counters before changing limits |
| Memory grows during a large build | Near-duplicate index is memory-resident | Use the 50M path first; monitor large profiles and stop safely if memory pressure becomes unsafe |
| Output file already exists | Another experiment used the same path | Move or rename the old corpus and report; never overwrite an unidentified training artifact |
| Topic distribution looks wrong | Keyword classifier or source mix does not match expectations | Inspect examples from the affected topic before editing weights |
| Corpus contains repeated boilerplate | Cleaning or near-duplicate thresholds are insufficient | Inspect source records and adjust the configuration only as a new documented experiment |

## What to record

Record:

- selected YAML path;
- corpus and report paths;
- elapsed seconds;
- accepted document and character counts;
- accepted characters by source;
- source stop reasons;
- exact and near duplicates;
- top rejection reasons;
- topic distribution;
- three accepted-document observations;
- one known corpus limitation.

## Run another corpus option

Use only the option required by your current experiment.

### Pipeline verification

```powershell
python build_training_corpus.py `
  --config configs/corpus_pipeline_verification.yaml
```

```bash
python build_training_corpus.py \
  --config configs/corpus_pipeline_verification.yaml
```

### Balanced Iteration 2 corpus

```powershell
python build_training_corpus.py `
  --config configs/corpus_balanced.yaml
```

```bash
python build_training_corpus.py \
  --config configs/corpus_balanced.yaml
```

### 800M Iteration 3 corpus

```powershell
python build_training_corpus.py `
  --config configs/corpus_800m.yaml
```

```bash
python build_training_corpus.py \
  --config configs/corpus_800m.yaml
```

## Keep experiment artifacts together

| Corpus | Tokenizer to create next | Token directory to create next |
| --- | --- | --- |
| Pipeline verification | `tokenizer/pipeline_verification_tokenizer.json` | `data/tokens/pipeline_verification` |
| 50M learning | `tokenizer/learning_50m_tokenizer.json` | `data/tokens/learning_50m` |
| Balanced | `tokenizer/balanced_tokenizer.json` | `data/tokens/balanced` |
| 800M | `tokenizer/800m_tokenizer.json` | `data/tokens/800m` |

Do not train one tokenizer and later rename it to look like another experiment. The tokenizer must actually be trained from the corpus recorded in the experiment.

## Under the hood

Read these files in order:

1. `src/corpus/sources.py`
2. `src/text_cleaner.py`
3. `src/corpus/quality.py`
4. `src/corpus/dedup.py`
5. `src/corpus/topics.py`
6. `src/corpus/pipeline.py`

Trace one source document into a `CorpusDocument`, through cleaning and filtering, and into the JSONL writer. Then find where report counters are updated.

## Check your understanding

1. Why is a configured corpus different from a raw download?
2. What is the difference between exact and near duplicates?
3. Why can the balanced corpus finish below its configured global ceiling?
4. Why is a character limit not the same as a token budget?
5. Which report fields would reveal that RFC material was missing?
6. Why is the 50M profile recommended for the first complete run?
7. What must change when you select a different corpus?

## Next lesson

Next: [Tokenizer and dataset](04_TOKENIZER_AND_DATASET.md).
