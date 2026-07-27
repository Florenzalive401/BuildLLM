# Reference Implementation Map

This page shows where each course topic appears in the code. Iteration 3 uses this complete path. Iterations 1 and 2 use the same path with different model profiles.

## End-to-end call path

```text
run_lab.py
  -> loads ModelProfile
  -> resolves device runtime
  -> invokes train.py
       -> BPETokenizer
       -> TokenDataset x 2
       -> LanguageModelDataModule
       -> GPTModel
       -> OptimizerFactory
       -> LearningRateScheduler
       -> TrainingEngine
       -> Validator
       -> CheckpointManager
       -> Trainer.fit
```

## Configuration

| File | Role |
| --- | --- |
| `configs/models/*.json` | Versioned iteration dimensions and runtime defaults |
| `src/model_profiles.py` | Validates profile JSON and selects CPU/GPU defaults |
| `src/config.py` | Complete serializable `ModelConfig` used by the network/checkpoint |
| `src/runtime.py` | Resolves device and supported precision |
| `run_lab.py` | Course launcher with model, device, data, and checkpoint-path selection |
| `train.py` | Applies profile values, then explicit CLI overrides |

Settings are resolved in this order:

```text
explicit CLI override > selected device runtime > model profile > code default
```

Architecture comes from the profile; vocabulary size comes from the active tokenizer.

`run_lab.py --checkpoint-directory` isolates checkpoints for separate course experiments. When the option is omitted, the existing `checkpoints/iteration_<n>` default is used.

## Corpus and tokenization

| File | Role |
| --- | --- |
| `src/corpus/document.py` | Common source-neutral record |
| `src/corpus/sources.py` | Local, Hugging Face, and RFC stream adapters |
| `src/text_cleaner.py` | Unicode/control/whitespace normalization |
| `src/corpus/quality.py` | Quality metrics and score |
| `src/corpus/dedup.py` | Exact fingerprint and near-duplicate SimHash |
| `src/corpus/topics.py` | Broad topic classification |
| `src/corpus/pipeline.py` | Quotas, acceptance, JSONL output, reports |
| `configs/corpus_pipeline_verification.yaml` | Small Wikipedia-only corpus-pipeline verification |
| `configs/corpus_learning_50m.yaml` | Reproducible 50M Wikipedia, RFC, and FineWeb learning corpus |
| `configs/corpus_balanced.yaml` | Balanced Iteration 2 larger-data corpus |
| `configs/corpus_800m.yaml` | Largest included Iteration 3 larger-data corpus |
| `prepare_800m_corpus.py` | Builds, tokenizes, and encodes the 800M corpus without starting model training |
| `train_tokenizer.py` | Byte-level BPE training and round-trip verification |
| `src/tokenizer.py` | Runtime encode/decode and special-token validation |
| `encode_corpus.py` | Deterministic document split and tensor encoding |

## Data

`src/dataset.py` converts one token region into shifted language-model windows. `src/datamodule.py` creates deterministic loaders and accepts distinct validation data. The older single-dataset split remains for compatibility, but production training passes separate training and validation datasets.

## Model

| File | Tensor responsibility |
| --- | --- |
| `src/embeddings.py` | `[B,T] -> [B,T,C]` token plus position embeddings |
| `src/attention.py` | `[B,T,C] -> [B,T,C]` causal multi-head attention |
| `src/feed_forward.py` | `[B,T,C] -> [B,T,F] -> [B,T,C]` |
| `src/transformer.py` | One pre-norm residual block |
| `src/stack.py` | Repeats blocks and coordinates layer caches |
| `src/model.py` | Final norm and `[B,T,C] -> [B,T,V]` tied projection |
| `src/kv_cache.py` | Validated per-layer attention cache structures |

## Training

| File | Responsibility |
| --- | --- |
| `src/loss.py` | Cross-entropy over vocabulary logits and target IDs |
| `src/training/optimizer.py` | AdamW groups and fused fallback |
| `src/training/scheduler.py` | Warmup and constant/linear/cosine learning rates |
| `src/training/engine.py` | One step/epoch, autocast, scaling, clipping, metrics |
| `src/training/validator.py` | Evaluation mode, no-grad loss and perplexity |
| `src/training/trainer.py` | Epoch lifecycle, validation, checkpoints, interrupts |
| `src/training_state.py` | Serializable counters and loss history |
| `src/training/checkpoint.py` | Atomic save, validation, load, retention |
| `src/experiment.py` | Run directory, config snapshot, event/metric logging |

## Generation

| File | Responsibility |
| --- | --- |
| `src/sampling.py` | Temperature, penalties, n-gram bans, top-k/top-p, draws |
| `src/generation.py` | Autoregressive loop, streams, stops, seeded RNG, cache |
| `generate_text.py` | Checkpoint loading and command-line generation |
| `interactive_generate.py` | Interactive prompt loop |

## Tests as executable documentation

Start with:

- `tests/test_dataset.py` for shifted windows;
- `tests/test_attention.py` for masks and attention shapes;
- `tests/test_model.py` for end-to-end logits;
- `tests/test_engine.py` for optimizer-step mechanics;
- `tests/test_checkpoint.py` for save/load and retention;
- `tests/test_generation.py` and `tests/test_kv_cache.py` for decoding;
- `tests/test_corpus_pipeline.py` for corpus output and quotas.

The tests show the expected behavior of each component. If a lesson and the code disagree, check the implementation and its tests before updating the lesson.

Use the [plain-language glossary](GLOSSARY.md) for unfamiliar terms, the [run-record worksheet](RUN_RECORD_WORKSHEET.md) for experiment evidence, and the [troubleshooting guide](TROUBLESHOOTING.md) for stage-specific responses.
