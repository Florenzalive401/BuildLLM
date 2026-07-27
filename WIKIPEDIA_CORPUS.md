# Wikipedia Corpus Pipeline

This upgrade downloads the official current article dump from Wikimedia, resumes interrupted downloads, optionally verifies the published checksum, extracts main namespace articles, removes redirects and wiki markup, filters short or unusable documents, and writes one JSON record per article.

## Install the added parser

```powershell
pip install -r requirements.txt
```

## Fast pipeline test with Simple English Wikipedia

Simple English Wikipedia is the first local source in the ordered learning
course. It is small enough to prepare before building the larger multi-source
corpus.

```powershell
python build_wikipedia_corpus.py `
  --project simplewiki `
  --output data/processed/wikipedia_simple.jsonl `
  --report data/reports/wikipedia_simple_report.json `
  --max-articles 10000
```

Continue with the 50M learning corpus:

```powershell
python build_training_corpus.py `
  --config configs/corpus_learning_50m.yaml
```

Then train and encode the matching tokenizer:

```powershell
python train_tokenizer.py `
  --corpus data/processed/training_corpus_learning_50m.jsonl `
  --format jsonl `
  --vocabulary-size 32768 `
  --output tokenizer/learning_50m_tokenizer.json

python encode_corpus.py `
  --corpus data/processed/training_corpus_learning_50m.jsonl `
  --format jsonl `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --output-directory data/tokens/learning_50m
```

## Download and extract all current English Wikipedia articles

```powershell
python build_wikipedia_corpus.py `
  --project enwiki `
  --output data/processed/wikipedia_en.jsonl `
  --report data/reports/wikipedia_en_report.json
```

The command downloads the current `pages-articles-multistream.xml.bz2` dump into `data/raw/wikipedia`. It can resume a partially completed download. It excludes media, talk pages, user pages, redirects, and revision history. It includes the current text of every main namespace English Wikipedia article that passes the minimum quality filter.

Plan for at least 70 GB of free disk space. More is preferable because the compressed dump, extracted JSONL corpus, tokenizer input, and encoded token files may coexist.

## Practical first full model corpus

The complete English Wikipedia corpus is much larger than the current model needs. A good first target is 500 million to 1 billion clean characters. The extractor still reads articles from the official complete dump, but stops after reaching the selected target.

```powershell
python build_wikipedia_corpus.py `
  --project enwiki `
  --max-characters 1000000000 `
  --output data/processed/wikipedia_en_1b_chars.jsonl `
  --report data/reports/wikipedia_en_1b_chars_report.json
```

Then rebuild the tokenizer from the new corpus:

```powershell
python train_tokenizer.py `
  --corpus data/processed/wikipedia_en_1b_chars.jsonl `
  --vocabulary-size 32768 `
  --output tokenizer/wikipedia_32k.json
```

Encode the corpus with the matching tokenizer:

```powershell
python encode_corpus.py `
  --corpus data/processed/wikipedia_en_1b_chars.jsonl `
  --tokenizer tokenizer/wikipedia_32k.json `
  --output-directory data/tokens/wikipedia_en_1b
```

Do not use an old checkpoint with the new tokenizer. A changed vocabulary changes token identifiers and embedding dimensions, so the model must start a new training run.

## Files added or updated

* `build_wikipedia_corpus.py`
* `src/wikipedia_source.py`
* `train_tokenizer.py`
* `encode_corpus.py`
* `tests/test_wikipedia_source.py`
* `requirements.txt`
