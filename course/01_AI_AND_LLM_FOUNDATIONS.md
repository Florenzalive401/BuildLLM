# AI and LLM Foundations

## What you will learn

You will build a practical mental model of artificial intelligence, machine learning, neural networks, language models, pretraining, inference, tokens, parameters, context, loss, epochs, and hallucination. These terms will appear in every later lesson.

## Where you are in the build

Your repository and Python environment are ready. Before downloading data, you need to understand what the project is trying to learn from that data.

```text
text documents
    -> tokenizer converts text into token IDs
    -> transformer predicts the next token ID
    -> loss measures prediction error
    -> optimizer changes model parameters
    -> repeated updates improve useful patterns
```

## Before you begin

No prior machine-learning training is required. You should be comfortable with the idea that a program reads input, transforms it, and produces output. Keep the [plain-language glossary](GLOSSARY.md) open when a term is unfamiliar.

## Files you should already have

You should have the cloned repository and the working `.venv` verified in the previous lesson.

## Files this lesson will create

This conceptual lesson does not create a corpus, tokenizer, model, or checkpoint. Record your worked-example answers and definitions in your lab notes.

## AI, machine learning, and deep learning

Artificial intelligence is the broad engineering field of building systems that perform tasks associated with prediction, language, perception, reasoning, or decision-making.

Machine learning is one approach to artificial intelligence. Instead of writing every decision rule manually, you provide examples, define an objective, measure error, and use an optimization process to improve the system.

Deep learning uses neural networks with many layers. Each layer transforms numbers into new representations. During training, the network adjusts millions of internal numerical values so its output better matches the objective.

A large language model, usually shortened to LLM, is a neural network trained on token sequences. “Large” is relative. BuildLLM’s largest model has about 100 million parameters, which is small compared with modern commercial systems but large enough to expose real model-training engineering problems.

## What BuildLLM predicts

BuildLLM uses a decoder-only transformer. Its training objective is next-token prediction.

Consider:

```text
The server returned an error
```

After tokenization, the text becomes a sequence of integer token IDs. The model sees earlier tokens and tries to predict the next token at every position:

```text
input : [The] [server] [returned] [an]
target: [server] [returned] [an] [error]
```

The model does not receive a rule that says what a server is. It repeatedly sees language patterns and changes its parameters when its probability distribution does not match the actual next token.

## Tokens are not the same as words

A tokenizer breaks text into reusable units called tokens. A token may be:

- a complete common word;
- part of a word;
- punctuation;
- whitespace;
- a byte-level representation of an unusual character;
- a special marker such as the end of a document.

The model never reads a Python string directly. It receives integers. Tokenizer identity therefore matters: token ID `1204` has meaning only inside the tokenizer that assigned it.

You will train a byte-level Byte Pair Encoding tokenizer later. Byte-level encoding allows the tokenizer to represent any input without requiring every word to exist in a fixed dictionary.

## Parameters are learned values

Parameters are the numerical weights changed during training. Embedding tables, attention projections, feed-forward matrices, and normalization layers contain parameters.

BuildLLM has three profiles:

| Iteration | Parameters | What the additional capacity changes |
| ---: | ---: | --- |
| 1 | 8,092,800 | Small enough for CPU study; limited language capacity |
| 2 | 42,112,000 | Wider and deeper; can represent more patterns but costs more memory and compute |
| 3 | 98,506,080 | Current reference implementation for long-running GPU training |

More parameters do not insert knowledge into the model. They provide more adjustable capacity. Useful behavior still depends on clean and relevant data, sufficient updates, stable optimization, and evaluation.

## Context is the model’s working window

Context length is the maximum number of tokens processed together in one sequence. Iteration 1 uses 128 tokens. Iterations 2 and 3 use 256.

If a prompt and generated continuation exceed the context limit, the model cannot attend to the entire sequence. Longer context provides more nearby information but increases memory and compute, especially inside attention.

Context is not permanent memory. It is the active token window for one training example or generation request.

## Pretraining and inference

Pretraining is the process used in this repository. The model learns general token patterns from a corpus through next-token prediction.

Inference is what happens after training. You load a checkpoint, tokenize a prompt, compute next-token probabilities, select a token, append it, and repeat.

This course does not cover fine-tuning, alignment, retrieval-augmented generation, agents, or deployment. Those systems build on a base model. The purpose here is to understand and operate the base-model pipeline first.

## The units used during training

Several units describe different parts of the work:

| Unit | Meaning in BuildLLM |
| --- | --- |
| Character | A unit in the cleaned text; corpus YAML limits use characters |
| Token | An integer produced by the tokenizer; model training consumes tokens |
| Training example | One fixed-length input window and its shifted target |
| Batch | Several examples processed together before one optimizer update |
| Optimizer step | One completed parameter update |
| Epoch | One pass over the selected training examples |
| Parameter | A learned numerical value inside the model |

These units are related but not interchangeable. A 50M-character corpus does not contain exactly 50M tokens. Token count depends on the tokenizer and text. Increasing batch size changes how many optimizer steps occur in an epoch. Comparing runs by epoch count alone can be misleading when the corpora differ.

## Loss is a training signal

The model produces a score for every vocabulary token at every sequence position. Those scores become probabilities. Cross-entropy loss measures how much probability the model assigned to the correct next token.

Lower loss generally means the correct next tokens received more probability. A single batch loss is noisy, so you read trends across many batches and compare training loss with validation loss.

Loss is not an accuracy percentage and does not prove factual correctness. A model can become better at predicting patterns while still generating false statements.

## Validation asks whether learning transfers

Training data updates the model. Validation data is held out and does not update parameters.

If training loss improves while validation loss also improves, the model is learning patterns that transfer to held-out documents. If training loss improves while validation becomes worse, the model may be memorizing the selected training region or learning patterns that do not transfer.

Validation is evidence, not a complete intelligence score. Generated samples, repetition, prompt relevance, and operational behavior provide additional evidence.

## Why models hallucinate

A language model generates likely token sequences. It does not query a verified fact database unless another system is added around it.

Hallucination is fluent output that is unsupported, incorrect, or invented. It can occur because the corpus contains errors, the prompt lacks necessary context, the model has limited capacity, the model learned a plausible pattern without a reliable fact, or sampling selected an incorrect continuation.

Do not treat fluent wording as verification. The course records factual errors instead of hiding them.

## Why data quality matters

The corpus determines which patterns are available to learn. Repeated documents give some text too much influence. Broken markup teaches broken patterns. Missing topics remain weak. Biased source selection changes what the model encounters.

This is why the course builds and inspects a corpus before constructing the transformer. Data preparation is part of model engineering, not a preliminary chore.

## The two experiment paths

### Controlled comparison path

Use the same 50M corpus, tokenizer, validation tokens, prompts, and sampling settings with all three model architectures.

This path helps answer:

```text
When the data stays fixed, what changes as the model becomes larger?
```

It is not perfectly controlled because device, batch size, context length, and runtime defaults also differ, but it is much cleaner than changing every variable.

### Larger-data training path

Use the balanced corpus for Iteration 2 and the 800M corpus for Iteration 3. Train a separate tokenizer for each corpus and use the intended GPU run.

This path helps answer:

```text
What does each larger model achieve with its intended data and compute plan?
```

The result may improve because of model capacity, data quantity, data mix, tokenizer behavior, optimizer steps, or a combination. Report those changes instead of crediting model size alone.

## How this course measures learning

You will collect:

- training and validation loss;
- validation perplexity when tokenizers are comparable;
- examples, batches, optimizer steps, and processed tokens;
- throughput and elapsed time;
- device, precision, and memory observations;
- fixed-prompt outputs with unchanged sampling settings;
- repetition, grammar, prompt relevance, consistency, and factual errors;
- checkpoint and recovery evidence.

One good output is not enough. One bad output is not enough. You need repeatable observations across checkpoints and prompts.

## Worked example: separating variables

Suppose Iteration 3 produces better technical text than Iteration 1. Before concluding that the larger architecture caused the improvement, ask:

1. Did both models use the same corpus?
2. Did both use the same tokenizer?
3. Did both process the same number of tokens?
4. Were prompts and sampling settings identical?
5. Did Iteration 3 receive more context?
6. Did one run stop early or overfit?

If the answers differ, describe the result as an end-to-end experiment, not an isolated architecture result.

## What success looks like

You are ready to continue when you can explain the difference between a character, token, example, batch, optimizer step, and epoch; distinguish pretraining from inference; and describe why loss, validation, and generated samples provide different evidence.

## Stop and check

Do not continue while treating a token as always equal to a word, an epoch as a fixed amount of work across datasets, or fluent text as verified knowledge. Review the worked examples until you can correct each statement.

## Common misconceptions and exact corrections

| Misconception | Exact correction |
| --- | --- |
| “The model stores a database of answers.” | The model stores learned numerical patterns that produce next-token probabilities; some training text can also be memorized. |
| “A token is a word.” | A token can be a word, fragment, punctuation mark, byte sequence, or special marker. |
| “More parameters automatically produce better text.” | More parameters add capacity; data, updates, stability, and evaluation still determine the result. |
| “One epoch is the same training budget everywhere.” | Work per epoch depends on selected examples, sequence length, and batch size. |
| “Lower training loss proves generalization.” | Validation and fixed-prompt evidence are needed to assess behavior beyond the training examples. |
| “Fluent output is factual.” | A language model can generate a plausible false statement and every factual claim still requires verification. |

## What to record

Write one plain-language definition for token, parameter, context, loss, validation, optimizer step, epoch, pretraining, inference, and hallucination. Also state the question answered by each course experiment path.

## Under the hood

The core training relationship is:

```text
token IDs [B,T]
    -> transformer
    -> logits [B,T,V]
    -> cross-entropy against target IDs [B,T]
    -> gradients
    -> AdamW optimizer update
```

`B` is batch size, `T` is sequence length, and `V` is vocabulary size. Later lessons trace these tensors through the actual source code.

## Check your understanding

Explain:

1. Why a tokenizer is part of a model experiment.
2. Why corpus characters and training tokens are not the same count.
3. What one optimizer step changes.
4. Why validation data must not update model weights.
5. Why a fluent answer can still be false.
6. What the controlled comparison path tries to hold fixed.
7. Why the larger-data path cannot isolate model size.

Use the [glossary](GLOSSARY.md) before continuing if any term remains unclear.

## Next lesson

Next: [Download the training data](02_DOWNLOAD_TRAINING_DATA.md).
