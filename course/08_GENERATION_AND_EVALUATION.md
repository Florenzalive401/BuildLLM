# Generate and Evaluate Text

## What you will learn

In this lesson you will load the Iteration 1 checkpoint with its matching tokenizer, turn logits into generated tokens, compare decoding settings, and evaluate the model with more evidence than one impressive or disappointing sample.

## Where you are in the build

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
-> GENERATED SAMPLES AND EVALUATION <- you are here
-> model comparison
```

Training changed the model's weights. Generation is the first time you use those weights to continue a prompt one token at a time.

## Before you begin

Use the prompts selected before training. Do not tune the prompts after seeing the results. Keep the weak, repetitive, and incorrect samples because they are evidence about the checkpoint.

Generation is not fact lookup. The model predicts plausible next tokens from patterns learned in its corpus. Fluent wording does not verify that a statement is true.

## Files you should already have

```text
tokenizer/learning_50m_tokenizer.json
checkpoints/iteration_1_learning_50m/best_checkpoint.pt
```

If `best_checkpoint.pt` was not written before an interruption, use the exact numbered or interrupted checkpoint path printed by the trainer. Do not guess a filename.

## Files this lesson will create

`generate_text.py` prints generated text and metadata to the terminal. It does not automatically save a report, so copy every command and complete output into your run record.

## Key ideas in plain language

### Generation repeats one decision

```text
prompt
-> tokenize the prompt
-> model produces next-token logits
-> sampling policy chooses one token
-> append the token
-> repeat until a stop condition
```

Training can score every position in a window in parallel because the causal mask hides future answers. Generation does not know future tokens, so it adds one token at a time.

### A concrete token-choice example

Imagine the model produces these simplified logits after `The firewall blocks`:

| Candidate token | Logit |
| --- | ---: |
| ` traffic` | 5.2 |
| ` unauthorized` | 4.8 |
| ` the` | 3.1 |
| ` banana` | -1.4 |

Greedy decoding always selects ` traffic` because it has the largest logit. Sampling converts the adjusted logits to probabilities, which may sometimes select ` unauthorized`. A low-ranked token such as ` banana` can be removed by top-k or top-p filtering.

### Greedy decoding is deterministic

Greedy decoding selects the highest-scoring token at every step. It is useful as a repeatable baseline and for diagnosing whether a checkpoint loads, but it can settle into bland or repetitive sequences.

### Temperature changes distribution sharpness

Temperature divides the logits before softmax. A temperature below 1 makes high-scoring tokens more dominant. A temperature above 1 gives lower-ranked tokens more opportunity. Temperature changes token selection at inference time; it does not retrain the model or add knowledge.

### Top-k and top-p filter candidates differently

Top-k retains exactly the `k` highest-scoring candidates before sampling. Top-p retains the smallest high-probability set whose cumulative probability reaches `p`, so its candidate count changes with the model's confidence.

BuildLLM can apply both. Using restrictive values can reduce bizarre tail selections, but excessive restriction can increase repetition and suppress useful alternatives.

### Repetition controls constrain the decoder

The repetition penalty reduces scores for token IDs that already appeared. The no-repeat n-gram rule prevents a candidate that would recreate an earlier sequence of `n` tokens.

These controls change the output policy. They do not prove that the model learned not to repeat, so record them with every sample.

### A seed helps reproduce random sampling

The seed initializes the random token draws. The same checkpoint, prompt tokenization, device behavior, and sampling configuration should be more repeatable with the same seed. Different checkpoints can produce different outputs with the same seed because their probability distributions differ.

### Context is a fixed budget

Prompt tokens plus generated tokens cannot exceed the model's maximum sequence length. `generate_text.py` trims an overlong prompt from the left and caps generation to the remaining space.

Iteration 1 has a 128-token context. If the prompt consumes 100 tokens, no more than 28 new tokens fit. Record prompt length and finish reason so a context-limit stop is not mistaken for a learned sentence ending.

### Generation can stop for several reasons

Generation can end because it produced the document-end token, matched an encoded stop-text sequence, reached `--max-new-tokens`, or filled the model context. The command also uses default stop texts for common Wikipedia markup unless `--allow-wiki-markup` is supplied.

### The key-value cache saves repeated work

The first forward pass processes the prompt. Later passes process only the newest token while reusing stored keys and values from every attention layer. The cache speeds generation but does not increase the context limit or change the learned weights.

## Run the lab

Run all three decoding policies with the same checkpoint and prompt.

### PowerShell

```powershell
python generate_text.py `
  --checkpoint checkpoints/iteration_1_learning_50m/best_checkpoint.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --prompt "The purpose of a firewall is" `
  --max-new-tokens 64 `
  --device cpu `
  --greedy

python generate_text.py `
  --checkpoint checkpoints/iteration_1_learning_50m/best_checkpoint.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --prompt "The purpose of a firewall is" `
  --max-new-tokens 64 `
  --temperature 0.7 `
  --top-k 40 `
  --top-p 0.9 `
  --repetition-penalty 1.15 `
  --no-repeat-ngram-size 3 `
  --seed 42 `
  --device cpu

python generate_text.py `
  --checkpoint checkpoints/iteration_1_learning_50m/best_checkpoint.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --prompt "The purpose of a firewall is" `
  --max-new-tokens 64 `
  --temperature 1.1 `
  --top-k 80 `
  --top-p 0.97 `
  --repetition-penalty 1.15 `
  --no-repeat-ngram-size 3 `
  --seed 42 `
  --device cpu
```

### Linux and macOS

```bash
python generate_text.py \
  --checkpoint checkpoints/iteration_1_learning_50m/best_checkpoint.pt \
  --tokenizer tokenizer/learning_50m_tokenizer.json \
  --prompt "The purpose of a firewall is" \
  --max-new-tokens 64 \
  --device cpu \
  --greedy

python generate_text.py \
  --checkpoint checkpoints/iteration_1_learning_50m/best_checkpoint.pt \
  --tokenizer tokenizer/learning_50m_tokenizer.json \
  --prompt "The purpose of a firewall is" \
  --max-new-tokens 64 \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 3 \
  --seed 42 \
  --device cpu

python generate_text.py \
  --checkpoint checkpoints/iteration_1_learning_50m/best_checkpoint.pt \
  --tokenizer tokenizer/learning_50m_tokenizer.json \
  --prompt "The purpose of a firewall is" \
  --max-new-tokens 64 \
  --temperature 1.1 \
  --top-k 80 \
  --top-p 0.97 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 3 \
  --seed 42 \
  --device cpu
```

Repeat the conservative sampling command for all fixed prompts:

```text
The purpose of a firewall is
When a software service fails in production,
In the history of computing,
Water changes from a liquid to
The evidence does not prove that
```

## What the command is doing

`generate_text.py` loads the tokenizer, reads the model configuration stored in the checkpoint, reconstructs the matching model, restores its weights, tokenizes the prompt, builds the stop sequences, and calls the cached autoregressive generator.

| Flag | Effect |
| --- | --- |
| `--checkpoint` | Selects the trained weights and stored model configuration |
| `--tokenizer` | Gives the checkpoint's token IDs their original text meaning |
| `--prompt` | Supplies the text prefix |
| `--max-new-tokens` | Caps the number of generated tokens |
| `--temperature` | Controls probability sharpness |
| `--top-k` | Retains only the highest-ranked `k` candidates |
| `--top-p` | Retains the smallest candidate set reaching cumulative probability `p` |
| `--repetition-penalty` | Reduces scores for tokens already used |
| `--no-repeat-ngram-size` | Prevents repeating an existing n-token phrase |
| `--seed` | Initializes random sampling |
| `--greedy` | Selects the highest logit instead of sampling |
| `--device` | Selects CPU, CUDA, or automatic device resolution |

## What success looks like

Every successful command prints:

```text
Generated Text
======================================================================
<prompt and continuation>

Finish reason: <reason>
Generated tokens: <count>
```

Operational success means the checkpoint and tokenizer load, generation completes, and the finish reason is understandable. Model-quality evaluation begins after operational success and requires several prompts.

Compare:

- whether the continuation stays related to the prompt;
- whether words and punctuation are well formed;
- how long grammatical structure is maintained;
- whether phrases repeat;
- whether claims are supported or invented;
- why generation stopped;
- whether small setting changes produce useful variation or instability.

## Stop and check

Do not continue to larger models until you have retained output from every fixed prompt, recorded every sampling value, and described both strengths and failures.

Stop and fix the experiment identity if the checkpoint and tokenizer came from different corpus paths. A generated string can look superficially plausible even when the wrong tokenizer corrupts many token meanings.

## Common problems and exact responses

| Problem | Likely cause | Exact response |
| --- | --- | --- |
| Checkpoint file is missing | Training did not create `best_checkpoint.pt` or a different directory was used | Use the exact verified, numbered, or interrupted checkpoint path printed by training. |
| Tokenizer mismatch or out-of-range ID | The checkpoint and tokenizer are from different experiments | Restore the tokenizer used to encode the checkpoint's training corpus; never retrain a replacement and assume equivalence. |
| Output loops immediately | The model is undertrained, probabilities are concentrated, or controls are permissive | Preserve the sample, compare greedy and seeded settings, and review validation evidence before changing only repetition controls. |
| Output is random at high temperature | Lower-ranked tokens receive too much probability | Return to the conservative baseline and change one sampling control at a time. |
| Output ends abruptly | Maximum token count, stop text, end token, or context limit was reached | Read `Finish reason`, record `Generated tokens`, and shorten the prompt or change the explicit limit only if that is the intended experiment. |
| Text is fluent but false | Next-token prediction produced a plausible statement | Mark the factual error; do not treat fluency as verification or hide the failure from the comparison. |
| Same seed produces a different GPU result | Device kernels or execution details can affect logits and draws | Treat seed as a reproducibility aid, record the device, and compare behavior rather than promising bit-for-bit cross-device identity. |

## What to record

Record checkpoint path, tokenizer path, prompt, tokenized prompt length if inspected, device, greedy status, maximum new tokens, temperature, top-k, top-p, repetition penalty, no-repeat n-gram size, seed, stop settings, full generated text, finish reason, generated token count, factual errors, repetition, relevance, grammar, and your interpretation.

## Under the hood

Read `generate_text.py` for checkpoint and tokenizer loading, `src/generation.py` for filtering and token selection, `src/model.py` for cached forward passes, and `src/kv_cache.py` for cache validation.

Loss and samples answer different questions. Validation loss aggregates the model's next-token assignments across held-out text. A generated sample follows one path through the probability distribution and is also affected by decoder settings. Use both.

Generation does not read a corpus YAML or token tensor directly. Its required artifact pair is the checkpoint and the tokenizer from the same experiment:

| Experiment | Checkpoint | Tokenizer |
| --- | --- | --- |
| Iteration 1 controlled comparison | `checkpoints/iteration_1_learning_50m/best_checkpoint.pt` | `tokenizer/learning_50m_tokenizer.json` |
| Iteration 2 controlled comparison | `checkpoints/iteration_2_learning_50m/best_checkpoint.pt` | `tokenizer/learning_50m_tokenizer.json` |
| Iteration 3 controlled comparison | `checkpoints/iteration_3_learning_50m/best_checkpoint.pt` | `tokenizer/learning_50m_tokenizer.json` |
| Iteration 2 larger-data path | `checkpoints/iteration_2_balanced/best_checkpoint.pt` | `tokenizer/balanced_tokenizer.json` |
| Iteration 3 larger-data path | `checkpoints/iteration_3_800m/best_checkpoint.pt` | `tokenizer/800m_tokenizer.json` |

## Check your understanding

1. What is the difference between a logit and a probability?
2. Why does temperature not add knowledge to a checkpoint?
3. How do top-k and top-p differ?
4. Why must sampling settings be held constant in a model comparison?
5. What does the key-value cache save, and what does it not change?
6. Why is one strong sample insufficient evidence?
7. Which finish reasons can make a continuation end without a learned sentence ending?

You are ready to continue when another learner could reproduce your samples from the checkpoint, tokenizer, prompts, and recorded settings.

## Next lesson

Next: [Choose and understand CPU or GPU execution](09_CPU_AND_GPU.md).
