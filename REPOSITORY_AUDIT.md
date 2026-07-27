# Repository Audit

This audit records the repository state before the educational expansion. It is
intended to keep future changes anchored to the working implementation.

## Executive assessment

The codebase already satisfies the most important architectural requirement:
all three model generations use one implementation. `run_lab.py` selects a JSON
profile and delegates to `train.py`; it does not maintain three trainers.

The production path is substantially more capable than the original course material exposed. The repository includes a real causal transformer, separate training and validation token regions, mixed precision, gradient scaling for FP16, Tensor Core settings, fused AdamW where PyTorch supports it, cosine decay with warmup, atomic checkpoint writes, retention, numbered and best checkpoints with last-state or interrupted metadata, scheduler recovery, generation controls, stop sequences, and a key-value cache.

The principal gap was not model code. It was the educational layer. The initial course consisted of short pages that described outcomes but did not teach the implementation, provide code-reading paths, distinguish pipeline verification from learning experiments, or define reproducible lab records. The rebuilt 14-lesson course and support pages now address that gap while keeping the application path intact.

## Architecture found

### Entry points

- `main.py` verifies the Python/PyTorch runtime.
- `build_training_corpus.py` runs the configurable multi-source corpus pipeline.
- `train_tokenizer.py` trains the byte-level BPE tokenizer.
- `encode_corpus.py` performs a deterministic document-level split and writes
  training and validation token tensors.
- `run_lab.py` maps course iteration numbers to model profiles and invokes the
  shared trainer.
- `train.py` composes datasets, model, AdamW, scheduler, engine, validator,
  checkpoint manager, and trainer.
- `generate_text.py` reconstructs model configuration from a checkpoint and
  performs controlled generation.

### Core implementation

- `src/model.py` composes input embeddings, the transformer stack, final layer
  normalization, and a tied output projection.
- `src/attention.py` implements multi-head causal self-attention and cached
  key/value reuse.
- `src/transformer.py` implements pre-normalization residual blocks.
- `src/training/engine.py` owns forward, loss, backward, gradient handling,
  mixed precision, optimizer steps, and per-step metrics.
- `src/training/trainer.py` owns the multi-epoch lifecycle, validation cadence,
  checkpoint policy, interruption, and callbacks.
- `src/training/checkpoint.py` writes complete recovery state through an atomic
  temporary-file replacement.

### Configuration boundary

The three model profiles change sequence length, width, depth, attention heads,
feed-forward width, learning rate, weight decay, and per-device runtime
defaults. They do not select alternate implementations.

`ModelConfig` is serialized into every checkpoint. Resume requires the restored
configuration to match the requested configuration exactly. This protects the
active run from accidental architecture drift.

## What is already strong

1. **One framework.** The iteration launcher delegates to a single trainer.
2. **Production compatibility.** The 100M dimensions remain 720/12/12/2880
   with sequence length 256.
3. **Explicit device behavior.** CPU uses FP32; CUDA supports BF16, FP16, and
   FP32 with capability checks.
4. **Data separation.** Encoding assigns complete documents to training or
   validation before token windows are created.
5. **Recovery completeness.** Checkpoints contain model, optimizer, scheduler,
   training state, model configuration, and metadata.
6. **Operational interruption.** The trainer has an interrupted-checkpoint
   path and protects the save from repeated `Ctrl+C`.
7. **Generation depth.** Sampling includes temperature, top-k, top-p,
   repetition penalty, no-repeat n-grams, stop tokens, stop sequences, seeded
   sampling, streaming, and KV caching.
8. **Corpus provenance.** Output records retain source, license, document
   identity, quality score, topic, and quality metrics.
9. **Test breadth.** Tests exist for the transformer, attention, embeddings,
   dataset, data module, optimizer, scheduler, engine, trainer, checkpointing,
   validation, sampling, generation, cache, tokenizer, and corpus pipeline.

## Gaps found

### Educational gaps addressed in this expansion

- No central course navigation or defined learning paths.
- No full corpus, tokenizer, dataset, transformer, training, checkpoint, or
  generation lessons.
- No code-to-concept reference map.
- No consistent lab record or three-model comparison protocol.
- No explicit explanation of the tokenizer/checkpoint compatibility contract.
- No clear separation between integration-test commands and meaningful
  training experiments.
- A malformed arrow sequence in the overview.
- A stale 100M guide statement that said batch size 8 while the actual GPU
  profile uses 16.

### Engineering gaps deliberately not hidden

1. **Environment reproducibility.** PyTorch is intentionally outside the pinned
   requirements because the correct wheel depends on the accelerator. Setup
   documentation must make the one-environment installation sequence explicit.
2. **Mid-epoch exactness.** Resume restores training state but not the exact
   shuffled sampler cursor or RNG state. An incomplete epoch restarts its data
   iteration.
3. **Experiment identity.** Checkpoint metadata stores paths and profile names,
   but not content hashes for corpus, tokenizer, profile, source revision, or
   environment lock data.
4. **Configuration scope.** Architecture and a few optimizer/runtime values are
   profile-driven. Scheduler, checkpoint cadence, retention, random seed, and
   generation defaults are still partly embedded in Python defaults.
5. **Corpus scale ceiling.** Near-duplicate state is memory-resident. It is
   appropriate for this workstation-scale project, not an unbounded
   multi-billion-document pipeline.
6. **Documentation maintenance.** Automated checks now cover local links, course order, artifact paths, learner-facing structure, key commands, terminology, and WordPress paragraph formatting. They cannot prove that external services or hardware-specific commands will remain available.
7. **Repository metadata.** This workspace copy contains no `.git` directory,
   so history, tracked-file status, and source revision could not be audited.

These are roadmap items, not reasons to replace the framework.

## Validation evidence

The documentation checks were updated during the human-centered course rebuild. Per the active operator instruction, pytest, model inspection, dry runs, corpus preparation, tokenization, encoding, generation, and training were not executed while the large-model run was active.

Read-only checks confirmed that local Markdown links resolve, referenced corpus and model configuration files exist, deprecated corpus names are absent from learner-facing material, required runnable-lesson sections are present, and course prose has no adjacent hard-wrapped paragraph lines.

## Prioritized roadmap

### Now

- Keep the course synchronized with the real code.
- Install the correct PyTorch wheel into the single project environment and run
  the full suite.
- Use the lab record in `course/13_COMPARE_THE_MODELS.md` for every iteration.

### Next

- Add a run manifest containing content hashes, Python/PyTorch/CUDA versions,
  hardware identity, precision, and resolved configuration.
- Move scheduler and checkpoint policy into versioned experiment
  configuration without changing checkpoint compatibility.
- Add deterministic sampler/RNG restoration for exact mid-epoch continuation.
- Extend documentation checks when new profiles, commands, or course pages are added.

### Later

- Add benchmark fixtures and fixed-prompt generation reports to compare
  checkpoints automatically.
- Add disk-backed or partitioned duplicate indexing only if corpus scale
  requires it.
- Publish the Markdown course through the Academy presentation layer while
  keeping this repository as the source of truth.

## Change boundary

This educational expansion does not modify the model architecture, training
loop, checkpoint format, tokenizer artifact, corpus profiles, or any saved run.
Iteration 3 remains the reference implementation.
