# BuildLLM Plain-Language Glossary

Use this glossary when a course term is unfamiliar. Definitions describe how the term is used in this repository.

## Repository and data artifacts

| Term | Plain-language meaning |
| --- | --- |
| Repository | The project directory containing code, configurations, documentation, and tests. |
| Virtual environment | The `.venv` directory that isolates this project's Python packages from other projects. |
| Dependency | A software package the project needs, such as PyTorch or PyYAML. |
| Configuration | A JSON or YAML file containing named settings so the shared code can run different experiments. |
| YAML | A human-readable configuration format used by the corpus profiles. |
| JSON | A structured text format used by model profiles and reports. |
| JSONL | JSON Lines, where each physical line is one complete JSON record. BuildLLM stores one accepted document per line in a corpus. |
| Raw data | Source material before the project's cleaning, scoring, duplicate filtering, and topic balancing. |
| Corpus | A collection of cleaned text documents used to train a tokenizer or model. |
| Corpus profile | A YAML file that defines sources, limits, quality rules, balancing, output paths, and duplicate filtering. |
| Provenance | Information describing where a document came from. |
| License metadata | The source license recorded with an accepted document; it does not replace reviewing the actual license terms. |
| Quality score | A numeric summary used by the corpus pipeline to accept or reject text based on configured rules. |
| Exact duplicate | A document whose normalized fingerprint matches an already accepted document. |
| Near duplicate | A document whose content is substantially similar to an accepted document according to SimHash distance. |
| Topic balancing | Limits that reduce the chance that abundant topics dominate the accepted corpus. |
| Character target | A corpus size limit measured after text cleaning; it is not a token count. |
| Report | A JSON artifact recording counts, paths, rejections, sources, topics, and other evidence from a pipeline stage. |
| Artifact chain | The ordered relationship from corpus to tokenizer to token files to checkpoint to generated samples. |

## AI and language-model terms

| Term | Plain-language meaning |
| --- | --- |
| Artificial intelligence (AI) | A broad category of systems that perform tasks associated with intelligent behavior. |
| Machine learning (ML) | A way to build behavior by fitting parameters from examples instead of coding every rule directly. |
| Neural network | A layered numerical function with trainable parameters. |
| Language model | A model that assigns probabilities to token sequences and predicts likely next tokens. |
| Large language model (LLM) | A language model with substantial parameters, data, and compute; “large” is contextual rather than a fixed threshold. |
| Decoder-only transformer | The causal transformer architecture used by BuildLLM to predict the next token from earlier tokens. |
| Pretraining | Training a model on broad text with a next-token prediction objective before task-specific use. |
| Inference | Running a trained model without optimizer updates, including text generation. |
| Hallucination | Fluent model output that is unsupported, invented, or incorrect. |
| Memorization | Reproduction of training material rather than a general pattern that transfers to new examples. |
| Bias | A systematic skew in data, model behavior, measurement, or decisions. |
| Capacity | The amount and complexity of patterns an architecture can potentially represent. More capacity does not guarantee learning. |

## Tokenizer and dataset terms

| Term | Plain-language meaning |
| --- | --- |
| Token | One unit produced by the tokenizer, such as a word fragment, punctuation mark, byte sequence, or special marker. |
| Token ID | The integer assigned to a token by one specific tokenizer. |
| Vocabulary | The complete token-to-ID mapping. |
| Vocabulary size | The number of entries the model can predict; course tokenizers use 32,768. |
| Byte-level BPE | Byte-level Byte Pair Encoding, which begins with byte representations and repeatedly learns common adjacent merges. |
| Special token | A reserved marker with application meaning, such as document start or document end. |
| Round trip | Encoding text to IDs and decoding those IDs back to text as a tokenizer verification. |
| Tokenizer identity | The exact learned token-to-ID mapping. Equal vocabulary sizes do not make two tokenizers compatible. |
| Encoding | Converting corpus documents to token IDs. |
| Document-level split | Assigning complete documents to training or validation before concatenating their tokens. |
| Training tensor | The serialized one-dimensional token-ID stream used to create training windows. |
| Validation tensor | The held-out token-ID stream used to measure next-token loss without updating weights. |
| Sequence length | The number of input tokens in one model window. |
| Context length | The maximum sequence the model can process, including prompt and generated tokens during inference. |
| Shifted target | The input token sequence moved one position forward so every position learns its next token. |

## Transformer terms

| Term | Plain-language meaning |
| --- | --- |
| Parameter | A learned numeric value inside the model. |
| Embedding | A learned vector that represents a token or position. |
| Embedding dimension | The width of each internal token representation. |
| Position embedding | A learned vector that tells the model where a token occurs in the sequence. |
| Self-attention | A learned operation that lets each token position mix information from allowed positions in the same sequence. |
| Causal mask | The rule preventing a position from seeing future tokens. |
| Query | The attention projection representing what a position is looking for. |
| Key | The attention projection representing what an available position offers. |
| Value | The attention projection containing information mixed into the result. |
| Attention head | One learned projection workspace inside multi-head attention. |
| Head dimension | The embedding width assigned to one attention head. |
| Feed-forward network | The nonlinear transformation applied independently to every token position in a block. |
| GELU | Gaussian Error Linear Unit, the nonlinear activation used in the feed-forward network. |
| Residual connection | A path that adds a sublayer's result to its original input. |
| Layer normalization | A normalization applied to values at each token position to improve training stability. |
| Pre-normalization | Applying layer normalization before attention or feed-forward work. |
| Logit | An unnormalized score for a vocabulary token. |
| Softmax | The function that converts scores to nonnegative probabilities totaling one. |
| Weight tying | Reusing the token embedding matrix as the output vocabulary projection. |
| Key-value cache | Stored attention keys and values from earlier generated tokens, used to avoid recomputing the full prefix. |

## Training units and optimization

| Term | Plain-language meaning |
| --- | --- |
| Example | One input window and its shifted target window. |
| Batch | Several examples processed together for one optimizer update. |
| Forward pass | Running input IDs through the model to produce logits and loss. |
| Loss | A numeric measurement of prediction error; lower is better only under a comparable evaluation contract. |
| Cross-entropy | The loss comparing the correct next token with the model's scores for every vocabulary token. |
| Backpropagation | Calculating how each parameter contributed to loss. |
| Gradient | The local direction and sensitivity of loss for a parameter. |
| Gradient norm | A summary of gradient magnitude across parameters. |
| Gradient clipping | Limiting total gradient norm before an optimizer update. |
| Optimizer | The algorithm that converts gradients into parameter updates. |
| AdamW | The adaptive optimizer used by BuildLLM with decoupled weight decay. |
| Weight decay | A regularizing update that discourages selected weights from growing unnecessarily. |
| Learning rate | The scale of optimizer updates. |
| Warmup | Gradually increasing learning rate during early optimizer steps. |
| Cosine decay | Lowering learning rate over later steps following a cosine-shaped schedule. |
| Optimizer step | One completed parameter update after processing a batch. |
| Epoch | One pass through the selected training examples. |
| Processed tokens | The valid target tokens included in completed optimizer work. |
| Training loss | Loss measured on examples that update the model. |
| Validation loss | Loss measured on held-out examples without optimizer updates. |
| Overfitting | Improving on training data while performance on held-out data stops improving or worsens. |
| Undertraining | Ending before the model receives enough useful updates to fit the available patterns. |
| Perplexity | `exp(loss)`, an effective choice-count interpretation that should only be compared under compatible tokenization and validation. |

## Hardware and numeric terms

| Term | Plain-language meaning |
| --- | --- |
| CPU | Central processing unit; accessible and suitable for Iteration 1 and bounded demonstrations. |
| GPU | Graphics processing unit; highly parallel hardware suited to transformer matrix operations. |
| CUDA | NVIDIA's GPU computing platform used by PyTorch. |
| VRAM | GPU memory used for weights, gradients, optimizer state, activations, batches, and temporary buffers. |
| FP32 | 32-bit floating-point format used by the CPU profiles. |
| BF16 | 16-bit bfloat format with an FP32-like exponent range, preferred on supported CUDA GPUs. |
| FP16 | 16-bit IEEE half precision with a smaller exponent range; BuildLLM uses gradient scaling with it. |
| Mixed precision | Running selected operations in lower precision while retaining suitable higher-precision state or operations. |
| Autocast | PyTorch's mechanism for selecting operation precision inside a mixed-precision region. |
| Gradient scaling | Protecting small FP16 gradients by scaling loss before backward and unscaling before clipping and updating. |
| Tensor Core | Specialized GPU hardware for supported matrix operations. |
| TF32 | TensorFloat-32 acceleration for supported FP32 CUDA matrix operations. |
| Fused AdamW | A GPU optimizer path combining work into fewer kernel launches where supported. |
| Out of memory (OOM) | The device cannot allocate required memory; reducing batch size is usually the first response. |
| Throughput | Completed work per unit time, such as tokens per second or optimizer steps per second. |

## Checkpoint and generation terms

| Term | Plain-language meaning |
| --- | --- |
| Checkpoint | A serialized file containing model state and training recovery state. |
| State dictionary | A mapping of component names to saved tensors or values. |
| Periodic checkpoint | A numbered save created at a configured interval. |
| Best checkpoint | The checkpoint associated with the lowest validation loss observed by the run. |
| Interrupted checkpoint | A save requested after controlled interruption; the reason is stored in metadata. |
| Atomic save | Writing a temporary file and replacing the final path only after serialization succeeds. |
| Retention | Automatically keeping only a configured number of numbered checkpoints. |
| Resume | Reconstructing compatible components, restoring saved state, and continuing the original epoch target. |
| Global step | Count of completed optimizer updates. |
| Sampling | Choosing a token from an adjusted probability distribution. |
| Greedy decoding | Always selecting the highest-scoring token. |
| Temperature | A control that sharpens or flattens token probabilities. |
| Top-k | Keeping only the `k` highest-scoring candidates. |
| Top-p | Keeping the smallest candidate set reaching cumulative probability `p`. |
| Repetition penalty | A decoder adjustment reducing scores for previously used tokens. |
| No-repeat n-gram | A rule preventing regeneration of an earlier n-token sequence. |
| Seed | A value used to initialize random token draws for more repeatable sampling. |
| Finish reason | The condition that ended generation, such as an end token, stop text, token limit, or context limit. |

## Continue learning

Return to the [course home](README.md), copy the [run-record worksheet](RUN_RECORD_WORKSHEET.md), or open the [troubleshooting guide](TROUBLESHOOTING.md).
