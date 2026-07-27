# BuildLLM Learning Lab

[![Tests](https://github.com/DrDeathLabs/BuildLLM/actions/workflows/tests.yml/badge.svg)](https://github.com/DrDeathLabs/BuildLLM/actions/workflows/tests.yml)
[![CodeQL](https://github.com/DrDeathLabs/BuildLLM/actions/workflows/codeql.yml/badge.svg)](https://github.com/DrDeathLabs/BuildLLM/actions/workflows/codeql.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE-CODE)
[![Course: CC BY-NC-SA 4.0](https://img.shields.io/badge/course-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE-CONTENT)

BuildLLM teaches engineers how decoder-only language models are built and
trained. The repository contains one corpus pipeline, one tokenizer workflow,
one transformer implementation, one training framework, and three model
profiles.

| Iteration | Parameters | Main purpose | Primary device |
| --- | ---: | --- | --- |
| 1 | 8,092,800 | Prove and study the complete pipeline | CPU |
| 2 | 42,112,000 | Study GPU training and model scaling | CUDA |
| 3 | 98,506,080 | Operate the current reference implementation | CUDA |

All three models use the same source code. Their architecture and device
defaults come from `configs/models`.

## Start here

The course must be followed in order. Begin with
[`course/README.md`](course/README.md).

The complete runnable sequence is:

```text
download repository
-> build Python environment
-> verify PyTorch and hardware
-> download Simple Wikipedia
-> build the 50M multi-source corpus
-> train the tokenizer
-> encode and split the corpus
-> inspect the model
-> train Iteration 1
-> generate and evaluate text
-> compare all three models on the same 50M data
-> train Iteration 2 on the balanced corpus
-> train Iteration 3 on the 800M corpus
```

No training lesson assumes that the corpus, tokenizer, or token files already
exist.

## 1. Download the repository

```text
git clone https://github.com/DrDeathLabs/BuildLLM.git
cd BuildLLM
```

Run the remaining commands from the repository root.

## 2. Create one Python environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the correct PyTorch build for the machine using the
[official PyTorch selector](https://pytorch.org/get-started/locally/). Then
install the remaining dependencies:

```powershell
pip install -r requirements.txt
python main.py
pytest
```

PyTorch is installed separately because CPU and CUDA machines require different
builds. It is installed into the same `.venv`, not a second environment.

## 3. Download Simple Wikipedia

```powershell
python build_wikipedia_corpus.py `
  --project simplewiki `
  --output data/processed/wikipedia_simple.jsonl `
  --report data/reports/wikipedia_simple_report.json `
  --max-articles 10000
```

## 4. Build the 50M learning corpus

```powershell
python build_training_corpus.py `
  --config configs/corpus_learning_50m.yaml
```

The profile processes Simple Wikipedia, RFCs, and FineWeb Edu in that order. It
creates:

```text
data/processed/training_corpus_learning_50m.jsonl
data/reports/training_corpus_learning_50m_report.json
```

## 5. Train the tokenizer

```powershell
python train_tokenizer.py `
  --corpus data/processed/training_corpus_learning_50m.jsonl `
  --format jsonl `
  --vocabulary-size 32768 `
  --output tokenizer/learning_50m_tokenizer.json
```

## 6. Encode and split the corpus

```powershell
python encode_corpus.py `
  --corpus data/processed/training_corpus_learning_50m.jsonl `
  --format jsonl `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --output-directory data/tokens/learning_50m
```

The document-level split creates:

```text
data/tokens/learning_50m/train_tokens.pt
data/tokens/learning_50m/validation_tokens.pt
data/tokens/learning_50m/encoding_report.json
```

## 7. Inspect and train Iteration 1

Inspect the profile and print the resolved command:

```powershell
python inspect_model.py `
  --model-config configs/models/gpt_first_cpu.json

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

Remove `--dry-run` to perform the pipeline verification run. This bounded run checks the complete training path; it does not measure language quality. Its separate checkpoint directory prevents a verification save from being mistaken for the learning experiment.

For the Iteration 1 learning run:

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

## 8. Generate text

```powershell
python generate_text.py `
  --checkpoint checkpoints/iteration_1_learning_50m/best_checkpoint.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --prompt "The purpose of a firewall is" `
  --device cpu
```

Use the same tokenizer that created the training tokens. A different tokenizer
can assign different text to the same token IDs even when the vocabulary size
matches.

## 9. Choose the next training track

The controlled comparison path uses the 50M corpus and `tokenizer/learning_50m_tokenizer.json` for all three iterations. This keeps the corpus and token-to-ID mapping fixed while model capacity changes. The course also uses the same bounded example counts and epoch target, while reporting the different context lengths and processed-token totals.

The larger-data training path uses:

| Model | Corpus configuration | Tokenizer | Token directory | Epochs |
| --- | --- | --- | --- | ---: |
| Iteration 2, 42M | `configs/corpus_balanced.yaml` | `tokenizer/balanced_tokenizer.json` | `data/tokens/balanced` | 3 |
| Iteration 3, 100M | `configs/corpus_800m.yaml` | `tokenizer/800m_tokenizer.json` | `data/tokens/800m` | 18 |

Follow the exact balanced build, tokenizer, encoding, training, and generation commands in [`course/10_ITERATION_TWO.md`](course/10_ITERATION_TWO.md). Follow the complete 800M sequence in [`course/12_ITERATION_THREE.md`](course/12_ITERATION_THREE.md).

Do not combine validation loss or perplexity from different tokenizers as if they measure the same prediction problem. Report controlled comparison results separately from larger-data results.

## Safe interruption and resume

Press `Ctrl+C` once and wait for the checkpoint confirmation.

Resume with the original command and add `--resume`. Keep the same model
iteration, tokenizer, token files, epoch target, batch settings, and checkpoint
directory.

The model, optimizer, scheduler, counters, losses, and processed-token totals are restored. The exact shuffled position inside an unfinished epoch is not stored, so that epoch starts its data iteration again.

## Learn more

- [AI and LLM foundations](course/01_AI_AND_LLM_FOUNDATIONS.md)
- [Corpus construction](course/03_BUILD_THE_CORPUS.md)
- [800M corpus preparation](CORPUS_800M.md)
- [Tokenizer and dataset](course/04_TOKENIZER_AND_DATASET.md)
- [Transformer walkthrough](course/05_TRANSFORMER_WALKTHROUGH.md)
- [Training and validation](course/06_TRAINING_AND_VALIDATION.md)
- [Checkpoints and recovery](course/11_CHECKPOINTS_AND_RECOVERY.md)
- [Plain-language glossary](course/GLOSSARY.md)
- [Run-record worksheet](course/RUN_RECORD_WORKSHEET.md)
- [Troubleshooting guide](course/TROUBLESHOOTING.md)
- [100M training](GPT_100M_TRAINING.md)
- [Reference implementation map](course/REFERENCE_IMPLEMENTATION.md)
- [Repository audit](REPOSITORY_AUDIT.md)

Iteration 3 remains the reference implementation. Its model dimensions,
training framework, checkpoint format, and existing profiles have not been
replaced by the course workflow.

## Generated artifacts

The GitHub repository contains the application, configurations, tests, course, and guides. Corpora, reports, trained tokenizers, encoded tensors, checkpoints, run outputs, and virtual environments are intentionally not stored in Git. You create those artifacts locally by following the course in order.

Do not download a tokenizer or checkpoint from an unrelated experiment and rename it. Corpus, tokenizer, token tensors, model configuration, and checkpoint identity must remain together.

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change. Report suspected vulnerabilities through the private process in [SECURITY.md](SECURITY.md), not through a public issue.

## Licensing

BuildLLM source code, tests, configurations, and automation use the [MIT License](LICENSE-CODE). Course lessons and educational documentation use [CC BY-NC-SA 4.0](LICENSE-CONTENT) with attribution as “BuildLLM Learning Lab by DrDeathLabs.” See the [license map](LICENSE) for details.
