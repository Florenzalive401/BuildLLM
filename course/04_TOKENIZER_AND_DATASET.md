# Tokenizer and Dataset

## What you will learn

You will train the tokenizer that defines the model’s vocabulary, convert the selected corpus into integer token IDs, split complete documents into training and validation regions, and understand how those token regions become next-token examples.

## Where you are in the build

```text
cleaned corpus JSONL
    -> train tokenizer
    -> encode each document
    -> assign document to training or validation
    -> write train_tokens.pt and validation_tokens.pt
    -> create shifted windows for model training
```

The transformer does not receive raw text. It receives token IDs. This lesson creates the exact numeric data consumed by `train.py`.

## Before you begin

Confirm:

- `.venv` is active;
- the terminal is at the repository root;
- the selected corpus JSONL exists and is not empty;
- its corpus report exists and has been inspected;
- you know which experiment identity you are building;
- the target tokenizer and token directory do not contain valuable files from another experiment.

The recommended first path uses:

```text
data/processed/training_corpus_learning_50m.jsonl
data/reports/training_corpus_learning_50m_report.json
```

## Files you should already have

Use the exact selected corpus JSONL and its report. For the recommended path these are `data/processed/training_corpus_learning_50m.jsonl` and `data/reports/training_corpus_learning_50m_report.json`.

## Files this lesson will create

For the 50M learning path:

| Path | Purpose |
| --- | --- |
| `tokenizer/learning_50m_tokenizer.json` | Vocabulary, merges, normalization, pre-tokenizer, decoder, and special-token definitions |
| `data/tokens/learning_50m/train_tokens.pt` | Flattened token IDs from documents assigned to training |
| `data/tokens/learning_50m/validation_tokens.pt` | Flattened token IDs from held-out documents |
| `data/tokens/learning_50m/encoding_report.json` | Corpus, tokenizer, split, document, character, and token counts |

The `.pt` files are PyTorch tensors stored on disk. They are data, not checkpoints.

## Key idea: the tokenizer defines the model’s text interface

A tokenizer has two directions:

```text
text -> token IDs
token IDs -> text
```

The model learns relationships among token IDs. If a checkpoint was trained when ID `1204` meant one text fragment, loading a tokenizer where ID `1204` means something else corrupts the interpretation of every input and output.

The tokenizer, encoded tokens, and checkpoint must therefore remain together as one experiment.

## A worked Byte Pair Encoding example

Byte Pair Encoding, or BPE, begins with small units and repeatedly merges frequent adjacent units.

Suppose the training text repeatedly contains:

```text
train
trained
training
trainer
```

A simplified merge process might discover:

```text
t + r -> tr
tr + a -> tra
tra + i -> trai
trai + n -> train
i + n + g -> ing
```

The resulting tokenizer may represent `training` as `[train] [ing]` instead of individual characters. Real byte-level BPE works over byte-aware units, uses a much larger corpus, and learns thousands of merges.

Frequent patterns often become shorter token sequences. Rare identifiers, unusual Unicode, URLs, or specialized code can require more tokens.

## Why byte-level BPE

BuildLLM uses:

- NFC Unicode normalization;
- a byte-level pre-tokenizer;
- a BPE vocabulary;
- a byte-level decoder.

Byte-level handling means every input can be represented through bytes even when a complete word is not in the vocabulary. This avoids a traditional unknown-word failure, although the vocabulary still contains an unknown special token for validation and compatibility.

## Vocabulary size

The course uses 32,768 vocabulary entries.

A larger vocabulary can represent frequent text with fewer tokens, but it increases the embedding table and output-scoring problem. Rare vocabulary rows may receive fewer useful updates.

A smaller vocabulary reuses units more often and reduces the vocabulary-sized matrices, but it creates longer token sequences.

Keep 32,768 when using the included model profiles. Changing vocabulary size changes embedding and output tensor shapes, parameter counts, and checkpoint compatibility.

## Special tokens

The tokenizer defines:

```text
<|padding|>
<|unknown|>
<|document_end|>
```

`<|document_end|>` is appended after each encoded document. It teaches the model that one document has finished and can also serve as a generation stop token.

Padding exists for compatibility even though the current fixed-length contiguous training windows usually do not require padded batches.

## Train the 50M tokenizer

### Windows PowerShell

```powershell
python train_tokenizer.py `
  --corpus data/processed/training_corpus_learning_50m.jsonl `
  --format jsonl `
  --vocabulary-size 32768 `
  --output tokenizer/learning_50m_tokenizer.json
```

### Linux or macOS

```bash
python train_tokenizer.py \
  --corpus data/processed/training_corpus_learning_50m.jsonl \
  --format jsonl \
  --vocabulary-size 32768 \
  --output tokenizer/learning_50m_tokenizer.json
```

## What each argument means

| Argument | Meaning |
| --- | --- |
| `--corpus` | Read training text from the selected cleaned JSONL corpus |
| `--format jsonl` | Treat each non-empty line as a JSON record and read its `text` field |
| `--vocabulary-size 32768` | Request the vocabulary size expected by the model profiles |
| `--output` | Save the complete tokenizer definition under this experiment identity |

The tokenizer is trained from the entire selected corpus, including documents that will later be assigned to validation. The validation split measures model generalization, not tokenizer generalization. Tokenizer vocabulary learning does not update model weights.

## What success looks like for tokenizer training

The program prints:

```text
Training tokenizer from data/processed/training_corpus_learning_50m.jsonl (jsonl)
Verification token count: <your value>
Saved tokenizer: tokenizer/learning_50m_tokenizer.json
Vocabulary size: 32,768
```

The verification encodes and decodes a fixed sentence and requires an exact round trip. If decoded text differs, the command fails instead of saving an apparently working text interface.

### Stop before encoding

Confirm the saved path and vocabulary size before encoding. Do not continue with a generic or differently named tokenizer.

## Encode and split the corpus

### Windows PowerShell

```powershell
python encode_corpus.py `
  --corpus data/processed/training_corpus_learning_50m.jsonl `
  --format jsonl `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --output-directory data/tokens/learning_50m
```

### Linux or macOS

```bash
python encode_corpus.py \
  --corpus data/processed/training_corpus_learning_50m.jsonl \
  --format jsonl \
  --tokenizer tokenizer/learning_50m_tokenizer.json \
  --output-directory data/tokens/learning_50m
```

## What encoding does

For each document, `encode_corpus.py`:

1. reads the `text` field;
2. encodes text into token IDs;
3. appends the document-end token;
4. hashes the document text with the split seed;
5. assigns the complete document to training or validation;
6. concatenates training documents into one training tensor;
7. concatenates validation documents into one validation tensor;
8. writes both tensors and an encoding report.

The default validation fraction is 0.05, meaning approximately five percent of documents are assigned to validation. Hash assignment makes the decision deterministic for the same document text and split seed.

## Why the split occurs by document

If one long document were divided at arbitrary token positions, part could land in training and a nearby part in validation. The model could then be evaluated on text nearly identical to text it trained on.

BuildLLM assigns a complete document before token windows are created. This is a stronger separation, although near-duplicate documents still need to be removed during corpus construction.

## Read the encoding report

Successful encoding prints and saves JSON with fields like:

```json
{
    "corpus": "data/processed/training_corpus_learning_50m.jsonl",
    "format": "jsonl",
    "tokenizer": "tokenizer/learning_50m_tokenizer.json",
    "vocabulary_size": 32768,
    "training_documents": "<your value>",
    "validation_documents": "<your value>",
    "training_tokens": "<your value>",
    "validation_tokens": "<your value>",
    "characters_encoded": "<your value>",
    "split_seed": 1729,
    "training_file": "data/tokens/learning_50m/train_tokens.pt",
    "validation_file": "data/tokens/learning_50m/validation_tokens.pt"
}
```

Windows PowerShell:

```powershell
Get-Content data/tokens/learning_50m/encoding_report.json
```

Linux or macOS:

```bash
cat data/tokens/learning_50m/encoding_report.json
```

Use `training_tokens` and `validation_tokens` when discussing data available to the model. Do not substitute the corpus character limit.

## From token tensors to training examples

`TokenDataset` creates input and target windows without storing a separate file for every example.

For sequence length four:

```text
flat tokens: [10] [20] [30] [40] [50]
input      : [10] [20] [30] [40]
target     : [20] [30] [40] [50]
```

The target is shifted one token to the right. One window supplies a next-token target at every position.

Iteration 1 uses sequence length 128. Iterations 2 and 3 use 256. The dataset stride equals sequence length, so the default windows are adjacent rather than heavily overlapping.

An approximate count of full non-overlapping windows is:

```text
floor((token_count - 1) / sequence_length)
```

The dataset’s exact indexing and any `--training-examples` or `--validation-examples` limit determine the final count printed by training.

## Inspect tokenizer behavior

Create a small set of inputs:

```text
The purpose of a firewall is
zero-day vulnerability
https://example.com/api/v2
def train_step(batch):
café
RareIdentifier_X9Q7
```

For each input, record:

- token IDs;
- token count;
- decoded text;
- whether the round trip is exact;
- how much of a 128-token context it consumes.

Ask why common English may use fewer tokens than rare identifiers or punctuation-heavy text.

## What success looks like

You can continue when:

- the tokenizer file exists and reports vocabulary size 32,768;
- the tokenizer verification round trip succeeded;
- both token tensor files exist;
- the encoding report exists;
- training and validation document counts are greater than zero;
- training and validation token counts are greater than the selected model’s sequence length;
- the report names the intended corpus and tokenizer;
- file paths in the report match the files you will pass to training.

## Stop and check

If the encoding report names the wrong tokenizer, delete nothing automatically. Identify the experiment, choose new output paths, and encode again.

## Common problems and exact responses

| Problem | Likely cause | What to do |
| --- | --- | --- |
| Corpus is missing or empty | Wrong path or corpus build did not complete | Return to the corpus lesson and verify the selected output |
| Invalid JSONL line | Truncated or manually damaged corpus | Inspect the reported line and rebuild or repair the corpus deliberately |
| Tokenizer verification fails | Normalizer, pre-tokenizer, decoder, or file problem | Do not encode; inspect the tokenizer configuration and input |
| Vocabulary is smaller than requested | Corpus does not contain enough repeated patterns for all merges | Verify the reported size and corpus; do not assume profile compatibility |
| Encoding uses large memory | Token chunks are accumulated before concatenation | Use the 50M learning path first and monitor larger corpora carefully |
| Validation split is empty | Corpus is too small or fraction unsuitable | Use more documents or adjust the split as a new recorded experiment |
| Old token files exist | Output directory belongs to another run | Select a new explicit token directory; do not overwrite unidentified tensors |
| Text decodes differently | Tokenizer mismatch or configuration error | Stop and verify the exact tokenizer used for both encoding and decoding |

## What to record

Record:

- corpus path and report;
- tokenizer command and output path;
- requested and actual vocabulary size;
- tokenizer verification result;
- encoding command and output directory;
- split seed and validation fraction;
- training and validation document counts;
- training and validation token counts;
- characters encoded;
- tokenization observations for common, technical, and unusual text.

## Build artifacts for another corpus

Use these commands only after building and inspecting the corresponding corpus.

### Pipeline-verification artifacts

```powershell
python train_tokenizer.py `
  --corpus data/processed/training_corpus_pipeline_verification.jsonl `
  --format jsonl `
  --vocabulary-size 32768 `
  --output tokenizer/pipeline_verification_tokenizer.json

python encode_corpus.py `
  --corpus data/processed/training_corpus_pipeline_verification.jsonl `
  --format jsonl `
  --tokenizer tokenizer/pipeline_verification_tokenizer.json `
  --output-directory data/tokens/pipeline_verification
```

```bash
python train_tokenizer.py \
  --corpus data/processed/training_corpus_pipeline_verification.jsonl \
  --format jsonl \
  --vocabulary-size 32768 \
  --output tokenizer/pipeline_verification_tokenizer.json

python encode_corpus.py \
  --corpus data/processed/training_corpus_pipeline_verification.jsonl \
  --format jsonl \
  --tokenizer tokenizer/pipeline_verification_tokenizer.json \
  --output-directory data/tokens/pipeline_verification
```

### Balanced artifacts

```powershell
python train_tokenizer.py `
  --corpus data/processed/training_corpus_balanced.jsonl `
  --format jsonl `
  --vocabulary-size 32768 `
  --output tokenizer/balanced_tokenizer.json

python encode_corpus.py `
  --corpus data/processed/training_corpus_balanced.jsonl `
  --format jsonl `
  --tokenizer tokenizer/balanced_tokenizer.json `
  --output-directory data/tokens/balanced
```

```bash
python train_tokenizer.py \
  --corpus data/processed/training_corpus_balanced.jsonl \
  --format jsonl \
  --vocabulary-size 32768 \
  --output tokenizer/balanced_tokenizer.json

python encode_corpus.py \
  --corpus data/processed/training_corpus_balanced.jsonl \
  --format jsonl \
  --tokenizer tokenizer/balanced_tokenizer.json \
  --output-directory data/tokens/balanced
```

### 800M artifacts

```powershell
python train_tokenizer.py `
  --corpus data/processed/training_corpus_800m.jsonl `
  --format jsonl `
  --vocabulary-size 32768 `
  --output tokenizer/800m_tokenizer.json

python encode_corpus.py `
  --corpus data/processed/training_corpus_800m.jsonl `
  --format jsonl `
  --tokenizer tokenizer/800m_tokenizer.json `
  --output-directory data/tokens/800m
```

```bash
python train_tokenizer.py \
  --corpus data/processed/training_corpus_800m.jsonl \
  --format jsonl \
  --vocabulary-size 32768 \
  --output tokenizer/800m_tokenizer.json

python encode_corpus.py \
  --corpus data/processed/training_corpus_800m.jsonl \
  --format jsonl \
  --tokenizer tokenizer/800m_tokenizer.json \
  --output-directory data/tokens/800m
```

## Under the hood

Read:

1. `train_tokenizer.py`
2. `encode_corpus.py`
3. `src/tokenizer.py`
4. `src/dataset.py`
5. `src/datamodule.py`

Find the BPE trainer, special-token list, round-trip verification, document hash split, document-end append, tensor concatenation, shifted dataset indexing, and maximum-example limit.

## Check your understanding

1. Why can two 32,768-entry tokenizers still be incompatible?
2. Why is vocabulary size part of the model architecture?
3. Why are documents split before windows are created?
4. What does the document-end token represent?
5. What is the difference between a token tensor and a checkpoint?
6. Which report tells you the actual training-token count?
7. Why must changing the corpus create a new tokenizer, token directory, and model run in this course?

## Next lesson

Next: [Transformer walkthrough](05_TRANSFORMER_WALKTHROUGH.md).
