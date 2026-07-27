# Download the Training Data

## What you will learn

You will download and extract Simple Wikipedia into a local JSON Lines file, inspect the extraction report, read several accepted documents, and verify that the first source required by the corpus builder is ready.

## Where you are in the build

```text
Repository
    -> Python environment
    -> downloaded source documents  <--- you are here
    -> cleaned multi-source corpus
    -> tokenizer and token files
    -> training
```

The model will not train directly from the Wikipedia dump. This lesson converts the compressed Wikimedia source into a simpler local document format that the next lesson can combine with RFCs and FineWeb Edu.

## Before you begin

Confirm:

- the terminal is at the repository root;
- `.venv` is active;
- `python main.py` completed successfully;
- the machine has an internet connection;
- the disk has room for the compressed dump and extracted JSONL file;
- no valuable file already occupies the selected output paths.

## Files you should already have

```text
build_wikipedia_corpus.py
configs/corpus_learning_50m.yaml
data/
```

The `data` subdirectories may be created by the script if they do not exist.

## Files this lesson will create

| Path | What it contains |
| --- | --- |
| `data/raw/wikipedia/<downloaded dump>` | The compressed Wikimedia pages-and-articles dump |
| `data/processed/wikipedia_simple.jsonl` | One cleaned JSON record per accepted article |
| `data/reports/wikipedia_simple_report.json` | Counts and paths describing the extraction |

JSONL means JSON Lines. Each line is a complete JSON object. This format allows later stages to stream one document at a time instead of loading the complete corpus into memory.

## Why Simple Wikipedia comes first

Simple Wikipedia uses general English and is smaller than full English Wikipedia. It gives the learning path a manageable local source while still exercising a real Wikimedia download, extraction, cleaning, duplicate-removal, reporting, and provenance workflow.

Simple does not mean perfect. Articles can still contain formatting remnants, uneven coverage, outdated information, or source bias. You will inspect the output instead of assuming the source is clean.

## What the downloader does

`build_wikipedia_corpus.py`:

1. locates the current pages-and-articles dump for the selected Wikimedia project;
2. downloads the compressed file or resumes a partial download when supported;
3. verifies the published checksum by default;
4. streams articles from the compressed XML;
5. removes redirects and non-article namespaces;
6. cleans article text;
7. removes exact duplicate text;
8. writes one JSON record per accepted article;
9. stops after the requested article or character limit;
10. writes a report.

Checksum verification reads the entire compressed dump and confirms that the downloaded bytes match the value published by Wikimedia. It protects against a corrupt or incomplete download.

## Run the lab

The learning path accepts up to 10,000 articles.

### Windows PowerShell

```powershell
python build_wikipedia_corpus.py `
  --project simplewiki `
  --output data/processed/wikipedia_simple.jsonl `
  --report data/reports/wikipedia_simple_report.json `
  --max-articles 10000
```

### Linux or macOS

```bash
python build_wikipedia_corpus.py \
  --project simplewiki \
  --output data/processed/wikipedia_simple.jsonl \
  --report data/reports/wikipedia_simple_report.json \
  --max-articles 10000
```

Run one version, not both.

## What each argument means

| Argument | Meaning |
| --- | --- |
| `--project simplewiki` | Select the Simple English Wikipedia dump |
| `--output ...jsonl` | Write accepted articles to the path expected by the course corpus configurations |
| `--report ...json` | Write extraction counts and source information |
| `--max-articles 10000` | Stop after writing 10,000 accepted articles |

The article limit bounds extraction output. The compressed source download may still be larger because the script needs the dump that contains those articles.

## What you will see

During download and extraction, progress output reports activity. Checksum verification prints:

```text
Verifying Wikimedia checksum. This reads the entire compressed dump once.
```

The final output is a JSON report with fields like:

```json
{
    "project": "simplewiki",
    "dump": "<downloaded dump path>",
    "output": "data/processed/wikipedia_simple.jsonl",
    "articles_examined": "<your value>",
    "articles_written": "<your value>",
    "articles_rejected": "<your value>",
    "exact_duplicates": "<your value>",
    "characters_written": "<your value>",
    "complete_dump_extracted": false
}
```

Your counts may differ when Wikimedia publishes a newer dump. `complete_dump_extracted` is false when an article or character limit intentionally stops extraction early.

## Inspect the report

Windows PowerShell:

```powershell
Get-Content data/reports/wikipedia_simple_report.json
```

Linux or macOS:

```bash
cat data/reports/wikipedia_simple_report.json
```

Check:

- `project` is `simplewiki`;
- `output` is the expected JSONL path;
- `articles_written` is greater than zero;
- `characters_written` is greater than zero;
- the dump path points to the downloaded source;
- rejected and duplicate counts are plausible rather than silently missing.

## Inspect accepted documents

Windows PowerShell:

```powershell
Get-Content data/processed/wikipedia_simple.jsonl -TotalCount 3
```

Linux or macOS:

```bash
head -n 3 data/processed/wikipedia_simple.jsonl
```

Each line should contain:

```json
{
  "source": "wikipedia",
  "page_id": "<page identifier>",
  "title": "<article title>",
  "text": "<cleaned article text>"
}
```

Open several documents from different topics. Look for readable sentences, meaningful titles, no obvious navigation pages, no empty text, and no repeated copies.

## Licensing and responsible use

The extracted article record keeps source identity, and the later corpus configuration adds the expected license description. You are still responsible for reviewing source terms before distributing a corpus or trained model.

Inspect for personal data, secrets, harmful material, and content you are not authorized to redistribute. Automated cleaning does not replace governance.

## What success looks like

You can continue when:

- `data/processed/wikipedia_simple.jsonl` exists and is not empty;
- `data/reports/wikipedia_simple_report.json` exists;
- the report names the correct project and output path;
- accepted document and character counts are greater than zero;
- several inspected JSONL lines contain valid JSON and readable text.

## Stop and check

The next lesson expects the exact path `data/processed/wikipedia_simple.jsonl`. If you changed the output path, either move the file or update the selected corpus YAML before continuing.

## Common problems and exact responses

| Problem | Likely cause | What to do |
| --- | --- | --- |
| Download stops partway through | Network interruption | Run the same command again; the downloader resumes when the server supports it |
| Checksum verification fails | Corrupt or incomplete compressed file | Remove or replace only the failed dump after confirming its exact path, then download again |
| Extraction is slow | Checksum and XML processing read large compressed data | Let the process continue; use system monitoring to confirm disk and CPU activity |
| Output exists from an older run | Reusing a path can mix experiment expectations | Move the old output and report to an explicitly named archive before rebuilding |
| JSONL contains no records | Source, limit, parser, or filtering problem | Read the terminal error and report; confirm the selected project and that the dump is valid |
| `mwparserfromhell` is unavailable | Optional accelerated parser was not installed for the Python version | The repository has a fallback parser; continue unless the program reports a failure |
| Disk becomes full | Compressed dump and extracted output need more space | Stop, free approved storage, and repeat only after verifying the destination |

Use the [troubleshooting guide](TROUBLESHOOTING.md) for additional download and data checks.

## What to record

Record:

- the complete download command;
- dump filename and date;
- `articles_examined`;
- `articles_written`;
- `articles_rejected`;
- `exact_duplicates`;
- `characters_written`;
- output and report paths;
- two observations from inspected articles.

## Under the hood

The downloader streams the compressed XML instead of loading the dump into memory. Each accepted article is cleaned and fingerprinted. The fingerprint detects exact duplicate cleaned text. The output preserves document boundaries because each article becomes one JSONL record.

Document boundaries matter later. Encoding assigns complete documents to training or validation before flattening their tokens, which reduces direct leakage between the two regions.

## Check your understanding

1. Why does the course keep the compressed dump and processed JSONL as separate files?
2. What does checksum verification protect against?
3. Why can the download be large even when `--max-articles` is 10,000?
4. What is one advantage of JSONL for corpus processing?
5. Which exact file does the next lesson require?

## Next lesson

Next: [Build the corpus](03_BUILD_THE_CORPUS.md).
