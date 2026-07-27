# Walk Through the Transformer

## What you will learn

In this lesson you will follow one sequence of token IDs through the decoder-only transformer until the model produces a score for every possible next token. You will learn what embeddings, causal self-attention, attention heads, feed-forward networks, normalization, residual connections, logits, and weight tying contribute to that process.

## Where you are in the build

```text
Repository
-> Python environment
-> downloaded source documents
-> cleaned corpus
-> trained tokenizer
-> training and validation token files
-> MODEL CONFIGURATION AND TRANSFORMER <- you are here
-> training run
-> checkpoint
-> generated samples
-> model comparison
```

The previous lesson converted text into token IDs. The transformer is the part of the application that receives those IDs and learns which token is likely to come next.

## Before you begin

You do not need calculus to complete this lesson. You should understand that the tokenizer converts text to integers and that a tensor is an array with a defined shape.

The symbols used in the diagrams are:

| Symbol | Meaning |
| --- | --- |
| `B` | Batch size, or the number of training examples processed together |
| `T` | Sequence length, or the number of token positions in one example |
| `C` | Embedding dimension, or the width of the model's internal representation |
| `H` | Number of attention heads |
| `D` | Width of one attention head, calculated as `C / H` |
| `F` | Feed-forward hidden dimension |
| `V` | Vocabulary size, which is 32,768 in the course experiments |

## Files you should already have

You should have the repository, the model profiles, and the application source:

```text
configs/models/gpt_first_cpu.json
configs/models/gpt_42m.json
configs/models/gpt_100m.json
src/model.py
src/embeddings.py
src/stack.py
src/block.py
src/attention.py
src/feed_forward.py
```

The encoded 50M token files from the previous lesson are needed for training later, but this walkthrough does not modify them.

## Files this lesson will create

This is a conceptual and code-reading lesson. It does not create model weights or checkpoints. Record your completed shape table and answers in your lab notes.

## Key ideas in plain language

### What the transformer is trying to do

Suppose the input is `The firewall blocks`. The tokenizer converts that text to IDs. At each position the model produces a list of scores for all 32,768 vocabulary entries. The score at the final position answers the question, “Which token should follow `The firewall blocks`?”

The complete path is:

```text
token IDs [B,T]
-> token and position embeddings [B,T,C]
-> transformer block 1 [B,T,C]
-> more transformer blocks [B,T,C]
-> final layer normalization [B,T,C]
-> vocabulary projection [B,T,V]
-> logits for every possible next token
```

The shape remains `[B,T,C]` through the transformer blocks. The final projection changes the last dimension from the internal width `C` to the vocabulary size `V`.

### Embeddings turn IDs into useful vectors

A token ID is only an index. ID 824 is not mathematically more meaningful than ID 823. The token embedding table looks up a learned vector for each token, while the position embedding table looks up a learned vector for each location in the sequence.

```text
token vector for "firewall"
+ position vector for position 2
= input representation at position 2
```

Adding position information matters because a decoder-only transformer processes the sequence in parallel during training. Without positions, it could see which tokens were present but could not reliably distinguish `dog bites person` from `person bites dog`.

### Causal attention controls which earlier positions matter

Self-attention lets each position build a new representation by mixing information from other positions. “Causal” means a position may use its own token and earlier tokens, but it may not look at future tokens.

For four token positions, the allowed attention pattern is:

```text
                 Key position
                 0  1  2  3
Query position 0 Y  -  -  -
Query position 1 Y  Y  -  -
Query position 2 Y  Y  Y  -
Query position 3 Y  Y  Y  Y
```

The mask is essential during training. The target for position 2 is stored at position 3 of the original token stream. If position 2 could inspect that future token, the model could copy the answer instead of learning next-token prediction.

### Queries, keys, and values are learned views of the same sequence

The attention layer creates three projections for every position:

- A query represents what this position is looking for.
- A key represents what each available position offers.
- A value represents the information that can be copied from that position.

The model compares each query with the available keys. A larger compatibility score produces a larger softmax weight, and those weights mix the values into a context vector.

```text
scores = query @ transpose(key) / sqrt(head width)
masked scores = hide every future position
weights = softmax(masked scores)
context = weights @ value
```

Softmax converts the allowed scores to nonnegative weights that total one. Dividing by the square root of the head width prevents dot products from growing too large simply because each vector has many elements.

### Multiple heads give attention several workspaces

The embedding width is divided into attention heads. Each head has its own learned projections and can develop a different way to relate positions. One head might become useful for nearby syntax while another becomes useful for a longer reference, but those roles are learned rather than assigned by the programmer.

Iteration 1 has an embedding width of 192 and six heads, so each head is 32 values wide. Iteration 2 uses `512 / 8 = 64` values per head. Iteration 3 uses `720 / 12 = 60` values per head. Calculate the value from each profile rather than assuming every model uses the same head width.

### The feed-forward network transforms each position

Attention moves information between token positions. The feed-forward network then applies the same learned nonlinear transformation to each position independently.

```text
[B,T,C]
-> Linear(C,F)
-> GELU activation
-> dropout
-> Linear(F,C)
-> [B,T,C]
```

GELU, or Gaussian Error Linear Unit, is the nonlinear activation function. Without a nonlinear transformation, stacking linear layers would still behave like one larger linear transformation and would sharply limit what the network could represent.

### Residual connections preserve an information path

Each transformer block adds a sublayer's output back to its input:

```text
x
|\
| LayerNorm -> causal self-attention --+
|                                      |
+--------------------------------------> add -> h
                                        |\
                                        | LayerNorm -> feed-forward --+
                                        |                            |
                                        +----------------------------> add -> output
```

This residual path lets a block keep useful existing information while learning a correction. It also gives gradients a more direct path through a deep stack during backpropagation.

### Pre-normalization stabilizes each sublayer input

BuildLLM applies layer normalization before attention and before the feed-forward network. Layer normalization rescales the values for each token position to a more controlled distribution. This arrangement is called a pre-normalized transformer block.

### Logits are scores, not probabilities

After the final transformer block, the model applies one more layer normalization and projects each position to `V` scores. These raw scores are logits. A higher logit means the model currently prefers that vocabulary entry, but logits do not need to be between zero and one and do not total one.

Training passes logits to cross-entropy loss. Generation adjusts the logits with temperature and filtering, then converts them to probabilities for token selection.

### Weight tying reuses the vocabulary table

The output projection uses the same weight matrix as the input token embedding when `weight_tying` is enabled. This saves a second matrix containing approximately `V × C` parameters and makes the learned input and output token representations share geometry.

The tokenizer, vocabulary size, model configuration, and checkpoint must agree. A checkpoint trained with one token-to-ID mapping cannot safely be used with a different tokenizer even if both tokenizers contain 32,768 entries.

## Run the model-inspection lab

These commands only construct each model and report its configuration. They do not train it.

### PowerShell

```powershell
python inspect_model.py `
  --model-config configs/models/gpt_first_cpu.json

python inspect_model.py `
  --model-config configs/models/gpt_42m.json

python inspect_model.py `
  --model-config configs/models/gpt_100m.json
```

### Linux and macOS

```bash
python inspect_model.py \
  --model-config configs/models/gpt_first_cpu.json

python inspect_model.py \
  --model-config configs/models/gpt_42m.json

python inspect_model.py \
  --model-config configs/models/gpt_100m.json
```

## What the commands are doing

`inspect_model.py` loads one JSON model profile, validates the configuration, creates the same `GPTModel` class used by training, counts its trainable parameters, and prints the resolved architecture. It does not open token files, update weights, or create checkpoints.

## What success looks like

The report should include these field names:

```text
Model
Configuration
Parameters
Trainable parameters
Layers
Embedding dimension
Attention heads
Head dimension
Feed-forward dimension
Sequence length
Default batch size
Default precision
```

Record the values rather than relying on memory. The expected architecture landmarks are:

| Setting | Iteration 1 | Iteration 2 | Iteration 3 |
| --- | ---: | ---: | ---: |
| Parameters | 8,092,800 | 42,112,000 | 98,506,080 |
| Sequence length `T` | 128 | 256 | 256 |
| Embedding width `C` | 192 | 512 | 720 |
| Blocks | 4 | 8 | 12 |
| Heads `H` | 6 | 8 | 12 |
| Head width `D` | 32 | 64 | 60 |
| Feed-forward width `F` | 768 | 2,048 | 2,880 |

## Stop and check

Do not continue until you can explain why the input shape is `[B,T]`, why hidden states use `[B,T,C]`, why attention scores use a square `T × T` relationship, and why the final logits use `[B,T,V]`.

Complete this shape table:

| Boundary | Iteration 1 | Iteration 2 | Iteration 3 |
| --- | --- | --- | --- |
| Input IDs | `[B,128]` | `[B,256]` | `[B,256]` |
| Embeddings | `[B,128,192]` | `[B,256,512]` | `[B,256,720]` |
| Q, K, or V | `[B,6,128,32]` | `[B,8,256,64]` | `[B,12,256,60]` |
| Attention scores | `[B,6,128,128]` | `[B,8,256,256]` | `[B,12,256,256]` |
| Feed-forward hidden | `[B,128,768]` | `[B,256,2048]` | `[B,256,2880]` |
| Logits | `[B,128,32768]` | `[B,256,32768]` | `[B,256,32768]` |

## Common problems and exact responses

| Problem | What it means | Exact response |
| --- | --- | --- |
| `embedding_dimension must be divisible by number_of_attention_heads` | The model width cannot be divided evenly among the configured heads. | Restore the checked-in model profile or choose a head count that divides the embedding width exactly. |
| `sequence_length exceeds maximum_sequence_length` | An input window is longer than the configured context limit. | Use the matching data configuration or shorten the requested sequence; do not silently change a trained checkpoint's architecture. |
| `token_ids contain a value outside the configured vocabulary` | The token tensor contains an ID the model cannot embed. | Confirm that the token files, tokenizer, and model vocabulary size belong to the same experiment. |
| Parameter count differs from the table | A model profile or architecture implementation changed. | Compare the JSON profile and current source before treating old checkpoints or course measurements as equivalent. |
| A learner reads an attention head as having a fixed assigned job | Attention heads are being interpreted too literally. | Describe a head as a learned projection workspace and inspect behavior as evidence rather than assigning it a role in advance. |

## What to record

Record the configuration path, parameter count, sequence length, embedding width, block count, head count, head width, feed-forward width, vocabulary size, whether weight tying is enabled, and any difference between your output and the table.

## Under the hood

Read these files in order:

1. `src/model.py` validates token IDs, calls the embedding and transformer stack, applies the final normalization, and creates logits.
2. `src/embeddings.py` adds token and learned position embeddings.
3. `src/stack.py` sends the hidden states through every block.
4. `src/block.py` implements the pre-normalized residual structure.
5. `src/attention.py` creates Q, K, and V, applies the causal mask, runs softmax, and mixes values.
6. `src/feed_forward.py` implements the position-wise nonlinear transformation.
7. `src/kv_cache.py` validates cached keys and values used during generation.

During training, an entire window can be processed in parallel because the causal mask prevents information leakage. During generation, the model produces one new token at a time. The key-value cache preserves each layer's previous keys and values so the application does not recompute the entire prefix at every step.

Memory and compute do not grow from parameter count alone. More parameters increase weights, gradients, and optimizer state. A longer sequence increases activation memory and makes the attention score matrix grow approximately with `T²`. A larger batch holds more examples at once. A larger vocabulary makes the embedding and output scoring work larger. These controls must be considered separately when diagnosing performance.

## Check your understanding

1. Why does a token need both a token embedding and a position embedding?
2. What information does the causal mask hide?
3. What is the difference between attention and the feed-forward network?
4. Why is Iteration 3's head width 60 rather than 64?
5. Why are logits not probabilities?
6. What would go wrong if a checkpoint were paired with a tokenizer that assigns different IDs?
7. Why does doubling sequence length increase attention work by more than two times?

You are ready to continue when you can trace `[B,T]` token IDs to `[B,T,V]` logits and describe the purpose of every stage without treating the transformer as a black box.

## Next lesson

Next: [Learn how training and validation change the model](06_TRAINING_AND_VALIDATION.md).
