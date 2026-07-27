# Compare the Three Models

## What you will learn

In this lesson you will turn run records, validation measurements, fixed-prompt samples, and operational evidence into defensible conclusions. You will create one table for the controlled 50M comparison and a separate table for the larger-data training path.

## Where you are in the build

```text
corpus -> tokenizer -> token files -> model -> training -> checkpoint -> samples
                                                                  |
                                                                  v
                                                        MODEL COMPARISON
                                                                  |
                                                                  v
                                                        next experiment
```

Comparison is not a victory lap for the largest model. It is the engineering process of identifying what changed, what evidence improved, what remained weak, and which next experiment is justified.

## Before you begin

Collect the run records, encoding reports, training output, checkpoint paths, and generated samples. Do not regenerate only the prompts that made one model look good.

Freeze the evaluation protocol before running missing samples:

- the exact prompt strings;
- maximum new tokens;
- greedy or sampling mode;
- temperature;
- top-k and top-p;
- repetition penalty;
- no-repeat n-gram size;
- random seed;
- stop-text behavior;
- device used for generation.

If any setting changes, label the results as a new comparison series.

## Files you should already have

Controlled comparison artifacts:

```text
tokenizer/learning_50m_tokenizer.json
checkpoints/iteration_1_learning_50m/<verified checkpoint>
checkpoints/iteration_2_learning_50m/<verified checkpoint>
checkpoints/iteration_3_learning_50m/<verified checkpoint>
```

Larger-data artifacts:

```text
tokenizer/balanced_tokenizer.json
checkpoints/iteration_2_balanced/<verified checkpoint>
tokenizer/800m_tokenizer.json
checkpoints/iteration_3_800m/<verified checkpoint>
```

You may complete the controlled comparison without completing the expensive larger-data runs. Mark missing experiments honestly rather than inventing values.

## Files this lesson will create

This lesson creates your final comparison report. Use the tables on this page and the [run-record worksheet](RUN_RECORD_WORKSHEET.md). Include raw samples or links to them so conclusions can be audited.

## Key ideas in plain language

### A fair comparison starts by naming the question

“Which model is better?” is too broad. Better at what, under which data, token budget, runtime, and decoding settings?

The controlled comparison asks:

> With the same 50M corpus, the same tokenizer, the same bounded example counts, the same epoch count, and the same prompt protocol, what changes as the model architecture grows?

The larger-data comparison asks:

> What behavior and operational cost did the intended Iteration 2 balanced run and Iteration 3 800M run produce when model, data, tokenizer, and compute scaled together?

### Controlled does not mean perfectly identical

The 50M path controls corpus, tokenizer, selected example counts, and epoch count. Iteration 1 uses a 128-token context while Iterations 2 and 3 use 256 tokens, so the exact windows and processed-token totals differ. Report those totals and describe architecture as the main planned variable rather than the only physical difference.

### Larger-data results combine several causes

The balanced and 800M runs differ in corpus contents, corpus size, tokenizer, encoded token count, parameter count, and training budget. A stronger result can be real and useful without proving which one of those changes caused it.

### Loss is comparable only under the same prediction contract

Cross-entropy loss and perplexity depend on token boundaries and held-out data. The controlled runs share the 50M tokenizer and validation tensor, so their validation losses can be compared with appropriate attention to context and budget.

Do not rank validation loss or perplexity from the 50M, balanced, and 800M tokenizers in one column. A tokenizer that splits text differently changes the number and difficulty of predictions.

### Samples need a rubric, not a feeling

Use the same prompts and score observable behavior. Keep every output so a reader can see why a score was assigned.

| Dimension | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| Token and word formation | Mostly broken fragments | Frequent damage | Common words with some damage | Mostly stable words | Stable formation throughout |
| Local grammar | No usable structure | Occasional fragments | Short grammatical spans | Mostly grammatical sentences | Sustained grammatical clauses |
| Prompt relevance | Unrelated | Weak association | Intermittently related | Usually on topic | Consistently continues the task |
| Long-range consistency | Immediate contradiction | Loses thread quickly | Holds a short thread | Maintains a paragraph idea | Sustains and develops the idea |
| Repetition | Immediate loop | Heavy repetition | Noticeable repeated phrases | Minor repetition | Varied continuation |
| Factual restraint | Incoherent certainty | Frequent invented specifics | Plausible but unreliable | Some uncertainty where appropriate | Better restraint, still requires verification |

This rubric is an observational course tool, not a benchmark of general intelligence or safety.

## Run the comparison lab

### Step 1: generate missing controlled samples

Use every model's verified checkpoint with the shared 50M tokenizer. The command pattern is:

PowerShell:

```powershell
python generate_text.py `
  --checkpoint <controlled checkpoint path> `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --prompt "The purpose of a firewall is" `
  --max-new-tokens 64 `
  --temperature 0.7 `
  --top-k 40 `
  --top-p 0.9 `
  --repetition-penalty 1.15 `
  --no-repeat-ngram-size 3 `
  --seed 42 `
  --device <cpu or cuda>
```

Linux and macOS use the same arguments with backslash line continuations.

Replace only the checkpoint and the device required to load it. Repeat for all five prompts.

### Step 2: complete the controlled evidence table

Use measured values:

| Field | Iteration 1 | Iteration 2 | Iteration 3 |
| --- | --- | --- | --- |
| Model profile | `gpt_first_cpu.json` | `gpt_42m.json` | `gpt_100m.json` |
| Parameters | 8,092,800 | 42,112,000 | 98,506,080 |
| Corpus | 50M | 50M | 50M |
| Tokenizer | 50M | 50M | 50M |
| Context length | 128 | 256 | 256 |
| Training examples | 20,000 | 20,000 | 20,000 |
| Validation examples | 1,000 | 1,000 | 1,000 |
| Epochs | 3 | 3 | 3 |
| Global step | `<your value>` | `<your value>` | `<your value>` |
| Processed tokens | `<your value>` | `<your value>` | `<your value>` |
| Best validation loss | `<your value>` | `<your value>` | `<your value>` |
| Device and precision | `<your value>` | `<your value>` | `<your value>` |
| Batch size | `<your value>` | `<your value>` | `<your value>` |
| Throughput | `<your value>` | `<your value>` | `<your value>` |
| Elapsed time | `<your value>` | `<your value>` | `<your value>` |
| Verified checkpoint | `<your path>` | `<your path>` | `<your path>` |

### Step 3: score controlled behavior

Create one row per prompt and model:

| Prompt | Iteration | Word formation | Grammar | Relevance | Consistency | Repetition | Factual restraint | Evidence note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| The purpose of a firewall is | 1 | `<0-4>` | `<0-4>` | `<0-4>` | `<0-4>` | `<0-4>` | `<0-4>` | `<quote or observation>` |

Do not average the numbers without retaining the evidence notes. A single total can hide whether a model improved grammar while becoming more confidently wrong.

### Step 4: complete the larger-data table

| Field | Iteration 2 balanced | Iteration 3 800M |
| --- | --- | --- |
| Corpus YAML | `configs/corpus_balanced.yaml` | `configs/corpus_800m.yaml` |
| Actual accepted characters | `<report value>` | `<report value>` |
| Actual training tokens | `<encoding report value>` | `<encoding report value>` |
| Tokenizer | `tokenizer/balanced_tokenizer.json` | `tokenizer/800m_tokenizer.json` |
| Parameters | 42,112,000 | 98,506,080 |
| Context length | 256 | 256 |
| Epoch target | 3 | 18 |
| Global step | `<your value>` | `<your value>` |
| Processed tokens | `<your value>` | `<your value>` |
| Validation loss | `<within-run value; do not rank across columns>` | `<within-run value; do not rank across columns>` |
| Device and precision | `<your value>` | `<your value>` |
| Throughput | `<your value>` | `<your value>` |
| Elapsed time | `<your value>` | `<your value>` |
| Verified checkpoint | `<your path>` | `<your path>` |

Score the same fixed prompts with the behavioral rubric. Compare observable behavior, but state directly that data, tokenizer, model, and compute changed together.

## What the comparison process is doing

You are triangulating three kinds of evidence:

1. Training evidence shows how optimization progressed on selected training windows.
2. Validation evidence shows next-token performance on held-out documents under the same tokenizer contract.
3. Behavioral evidence shows what one-token-at-a-time generation does under fixed prompts and decoding controls.

Operational evidence adds cost and reliability: throughput, elapsed time, peak memory, checkpoint size, interruption recovery, and system stability.

## What success looks like

A successful report:

- contains both tables rather than mixing the experiment paths;
- uses actual counts from reports and trainer output;
- retains all fixed-prompt samples;
- records the decoder settings;
- compares loss only where tokenization and validation data permit it;
- links every claim to an observation;
- reports surviving failure modes;
- names limitations;
- proposes the next experiment from evidence.

## Stop and check

Do not publish a conclusion if:

- one model used different prompts or sampling settings without a separate series;
- a checkpoint was paired with the wrong tokenizer;
- processed tokens and global steps were omitted;
- controlled and larger-data validation losses were ranked together;
- only selected good samples were retained;
- the proposed cause is not isolated by the experiment.

## Common problems and exact responses

| Problem | Why it weakens the result | Exact response |
| --- | --- | --- |
| “The largest model is best” with no measurements | Model size is being substituted for evidence | Complete the tables and cite specific loss, behavior, and operational observations. |
| One lucky continuation is shown | Sampling variance and selection bias are hidden | Include every fixed prompt and seed, including failures. |
| Perplexity from different tokenizers is ranked | Prediction units are not comparable | Keep within-tokenizer loss trends separate and use fixed-prompt behavior for cross-tokenizer observations. |
| Equal epochs are called equal compute | Dataset, batch, and context determine work per epoch | Report processed tokens, optimizer steps, throughput, and elapsed time. |
| Sampling settings changed for one model | Decoder policy became a confounding variable | Start a new labeled comparison series or rerun with the frozen settings. |
| A fluent falsehood receives a high quality score | Style was mistaken for factual reliability | Score factual restraint separately and mark every unsupported claim. |
| No next experiment is justified | The report summarizes but does not guide engineering | Select the dominant limitation and change one major variable to test it. |

## What to record

Preserve the complete commands, artifact identities, hardware and precision, run tables, raw outputs, rubric scores, evidence notes, loss curves, checkpoint paths, interruptions, limitations, and final decision. The comparison should be reproducible by another engineer with access to the same artifacts.

## Under the hood

Parameter count, corpus size, token count, context length, batch size, optimizer steps, epochs, and sampling policy are different controls:

| Control | Question it answers |
| --- | --- |
| Parameters | How much representational capacity does the architecture have? |
| Corpus content and quality | What text and patterns are available to learn? |
| Encoded tokens | How much tokenized data exists? |
| Context length | How much history fits in each example and generation? |
| Batch size | How many examples contribute to each update? |
| Optimizer steps | How many weight updates completed? |
| Epochs | How many passes were made through selected examples? |
| Sampling policy | How were learned logits converted to output? |

Use this evidence-to-next-experiment guide:

| Evidence | Plausible next experiment |
| --- | --- |
| Training and validation loss remain high | Inspect batches and tokenizer, then test more optimizer steps or a justified learning-rate change |
| Training improves while validation worsens | Improve split quality, deduplication, regularization, or stop earlier |
| Loss improves but one topic remains weak | Inspect accepted corpus coverage and source quality for that topic |
| Text loops despite improving validation | Compare checkpoints and then test one decoding control or training-budget change |
| Hardware is underutilized | Measure batch size, workers, transfers, precision, and kernel support |
| Run is memory-limited | Reduce batch size or context in a new documented experiment before changing architecture |
| Fluent factual errors persist | Improve data and evaluation, and retain the limitation; a larger model alone does not guarantee truth |

## Check your understanding

1. What question does the controlled comparison answer?
2. Why is the controlled path not perfectly identical across all tensor operations?
3. Why must balanced and 800M perplexity remain separate?
4. What does the behavioral rubric add beyond loss?
5. Why are processed tokens and optimizer steps both useful?
6. Which conclusion can be supported when model, corpus, tokenizer, and compute all changed?
7. What is your next experiment, and which evidence justifies it?

You have completed the course when you can explain the complete artifact chain, reproduce every run, distinguish operational success from model quality, and choose a next experiment without automatically choosing a larger model.

## Next lesson

Return to the [BuildLLM Learning Lab course home](README.md), use the [glossary](GLOSSARY.md) for reference, complete the [run-record worksheet](RUN_RECORD_WORKSHEET.md), and keep the [troubleshooting guide](TROUBLESHOOTING.md) with future experiments.
