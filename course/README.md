# BuildLLM Learning Lab

BuildLLM teaches you how a decoder-only language model is built, trained, evaluated, and scaled. You will work with the same implementation used by all three model iterations in this repository. The smaller models are not separate tutorial programs. Architecture, runtime, and data choices change through configuration and command-line arguments while the training framework remains the same.

## Who this lab is for

This lab is written for software engineers, systems engineers, cybersecurity professionals, technical leaders, and anyone who wants to understand language models by building one. You do not need graduate-level mathematics. You should be comfortable opening a terminal, editing a text file, and reading Python with guidance.

You can complete the first model on a CPU. The 42M and 100M models support CPU execution for short demonstrations, but meaningful full-data training requires a compatible GPU. If you do not have that hardware, you can still follow every explanation, inspect the profiles, run bounded demonstrations, and study real training output.

## What you are building

The course follows one chain of real files:

```text
Repository
    -> Python environment
    -> downloaded source documents
    -> cleaned corpus
    -> trained tokenizer
    -> training and validation token files
    -> model configuration
    -> training run
    -> checkpoint
    -> generated samples
    -> model comparison
```

Each arrow is a dependency. Training cannot begin until the corpus has been built, the tokenizer has been trained, and the corpus has been encoded. Generation cannot begin until training has saved a checkpoint. Follow the lessons in order the first time through.

## The three models

| Iteration | Parameters | Context length | Main lesson | Intended device |
| ---: | ---: | ---: | --- | --- |
| 1 | 8,092,800 | 128 tokens | Prove and understand the complete process | CPU |
| 2 | 42,112,000 | 256 tokens | Learn how architecture, data, and GPU execution affect training | CUDA GPU |
| 3 | 98,506,080 | 256 tokens | Operate a long-running model-training system reliably | CUDA GPU |

Parameters are the learned numerical values inside a model. More parameters give the model more capacity to represent patterns, but they do not guarantee better output. A larger model also needs suitable data, enough optimizer updates, stable training, and honest evaluation.

## Two different questions

The course uses two experiment paths because one comparison cannot answer every question.

### Controlled comparison path

Train Iterations 1, 2, and 3 with the same 50M corpus, tokenizer, and validation data. The model architecture is the largest intentional difference. This is the better path for asking, “What changed when the model became wider and deeper?”

Use:

```text
configs/corpus_learning_50m.yaml
tokenizer/learning_50m_tokenizer.json
data/tokens/learning_50m
```

### Larger-data training path

Train Iteration 2 with the balanced corpus for three epochs, then train Iteration 3 with the 800M corpus for eighteen epochs. These runs change the model, corpus, tokenizer, token count, and compute budget together. They answer, “What can this model do with its intended larger-data training setup?”

| Model | Corpus configuration | Tokenizer | Encoded tokens | Epoch target |
| --- | --- | --- | --- | ---: |
| Iteration 2 | `configs/corpus_balanced.yaml` | `tokenizer/balanced_tokenizer.json` | `data/tokens/balanced` | 3 |
| Iteration 3 | `configs/corpus_800m.yaml` | `tokenizer/800m_tokenizer.json` | `data/tokens/800m` | 18 |

Do not use the larger-data runs to claim that parameter count alone caused an improvement. Several variables changed. The final comparison lesson shows you how to report those results honestly.

## Three levels of execution

### Quick end-to-end check

Use one epoch, 500 training examples, and 100 validation examples. This verifies that the tokenizer loads, token windows are valid, forward and backward passes complete, validation runs, and a checkpoint can be saved and reloaded. It does not evaluate language quality.

### Learning experiment

Use Iteration 1 for three epochs with 20,000 training examples and 1,000 validation examples. This run is small enough for CPU learning and large enough to observe loss movement and early changes in generated text.

### Full-data run

Use all encoded training and validation tokens for the selected experiment. Full-data runs are evaluated with processed tokens, optimizer steps, training and validation trends, fixed prompts, unedited output, throughput, elapsed time, and checkpoint recovery.

## Course order

| Step | Lesson | You will finish with |
| ---: | --- | --- |
| 1 | [Getting started](00_GETTING_STARTED.md) | A cloned repository and verified Python/PyTorch environment |
| 2 | [AI and LLM foundations](01_AI_AND_LLM_FOUNDATIONS.md) | A practical mental model of data, tokens, training, and generation |
| 3 | [Download the training data](02_DOWNLOAD_TRAINING_DATA.md) | A local Simple Wikipedia JSONL source file and report |
| 4 | [Build the corpus](03_BUILD_THE_CORPUS.md) | A cleaned corpus selected for your experiment |
| 5 | [Tokenizer and dataset](04_TOKENIZER_AND_DATASET.md) | A matching tokenizer and separate training/validation token files |
| 6 | [Transformer walkthrough](05_TRANSFORMER_WALKTHROUGH.md) | An understanding of the tensors and components inside the model |
| 7 | [Training and validation](06_TRAINING_AND_VALIDATION.md) | The ability to read a training run and decide whether it is learning |
| 8 | [Iteration 1](07_ITERATION_ONE.md) | Your first CPU-trained checkpoint |
| 9 | [Generation and evaluation](08_GENERATION_AND_EVALUATION.md) | Repeatable samples and an evaluation record |
| 10 | [CPU and GPU execution](09_CPU_AND_GPU.md) | A hardware and precision plan for larger models |
| 11 | [Iteration 2](10_ITERATION_TWO.md) | Controlled and larger-data 42M experiments |
| 12 | [Checkpoints and recovery](11_CHECKPOINTS_AND_RECOVERY.md) | A tested interruption and resume procedure |
| 13 | [Iteration 3](12_ITERATION_THREE.md) | A production-style 100M training plan and run |
| 14 | [Compare the models](13_COMPARE_THE_MODELS.md) | An evidence-based comparison and next experiment |

## Support pages

- Use the [plain-language glossary](GLOSSARY.md) whenever a term is unfamiliar.
- Copy the [run-record worksheet](RUN_RECORD_WORKSHEET.md) before every training experiment.
- Use the [troubleshooting guide](TROUBLESHOOTING.md) when a command fails or output does not match the lesson.
- Use the [reference implementation map](REFERENCE_IMPLEMENTATION.md) when you want to trace a concept into the Python source.

## How to use each lesson

Every runnable lesson tells you what should already exist, what the next command creates, what the command does, what successful output looks like, what to record, and when not to continue. Sections labeled “Under the hood” are for implementation depth. You can complete the guided path first and return to those sections later.

Commands are shown for Windows PowerShell first. Linux and macOS commands are also provided when activation or line-continuation syntax differs. Run every command from the repository root unless the lesson explicitly says otherwise.

> **Important:** Keep each corpus, tokenizer, token directory, and checkpoint directory together as one experiment. A tokenizer with the same vocabulary size is not automatically compatible with another experiment.

## What to record

For every training experiment, record:

- the complete command;
- model iteration and JSON profile;
- corpus YAML and generated corpus report;
- tokenizer and vocabulary size;
- training and validation token files and counts;
- device, precision, batch size, workers, and context length;
- epoch target, optimizer steps, and processed tokens;
- training and validation loss;
- elapsed time and tokens per second;
- numbered and best checkpoint paths, including the metadata reason for last-state or interrupted saves;
- fixed prompts, generation settings, and unedited outputs;
- one conclusion supported by evidence;
- one limitation or unresolved question.

The worksheet provides a copyable format. This record prevents you from attributing an output change to model size when data, tokenizer, compute, or sampling also changed.

## Before you start

You are ready for Lesson 1 if you can open PowerShell, Terminal, or a shell; have enough disk space for a repository and Python environment; and can install software on the machine. You do not need a GPU to begin.

Next: [Getting started](00_GETTING_STARTED.md).
