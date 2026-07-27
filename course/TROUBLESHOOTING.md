# BuildLLM Troubleshooting Guide

Start with the stage that failed. Preserve the full command and error before changing anything. Make one correction at a time so you know which change solved the problem.

## Environment and repository

| Symptom | Check | Response |
| --- | --- | --- |
| `python` or `py` is not recognized | Python is not installed or not on the shell path | Install a supported Python version, open a new terminal, and repeat the environment lesson. |
| Activation script is blocked in PowerShell | PowerShell execution policy blocks local scripts | Use the current-user process policy permitted by your organization or run the environment's Python executable directly; do not create a second environment. |
| Package import fails | Wrong Python interpreter or incomplete installation | Activate `.venv`, run `python -m pip --version`, and install requirements into that interpreter. |
| `torch` cannot import | PyTorch was not installed into `.venv` | Use the official PyTorch selector for the machine, install into the active environment, and rerun `python main.py`. |
| A command cannot find a repository file | Terminal is in the wrong directory | Navigate to the repository root and confirm `main.py`, `run_lab.py`, and `configs` are visible. |
| `CUDA Available: False` | CPU-only wheel, driver issue, unsupported GPU, or wrong environment | Confirm driver and PyTorch build compatibility; use CPU only for bounded work until CUDA verifies. |

## Wikipedia download

| Symptom | Check | Response |
| --- | --- | --- |
| Download is slow | Network speed and dump size | Keep the terminal open; a 10,000-article extraction can still require downloading and reading a large compressed dump. |
| Download is interrupted | Partial download file and network availability | Rerun the same command; the downloader resumes when the server supports it. |
| Checksum verification takes time | The complete compressed file must be read | Wait for verification; do not delete a valid download because the terminal is temporarily quiet. |
| Output already exists | Previous extraction used the same path | Inspect and preserve it, or intentionally choose a new path; do not overwrite an artifact you have not identified. |
| Report counts are unexpectedly low | Namespace, redirect, cleaning, duplicate, or article limit filtering | Read the report fields and inspect accepted JSONL records before changing limits. |

## Corpus construction

| Symptom | Check | Response |
| --- | --- | --- |
| Local Wikipedia file is missing | Previous lesson did not finish or path differs | Build `data/processed/wikipedia_simple.jsonl` with the documented command. |
| Hugging Face source cannot load | Network, package version, dataset access, or remote change | Preserve the error, verify network and installed dependencies, then retry; do not substitute an unknown corpus silently. |
| RFC acquisition fails | Network or RFC cache state | Inspect `data/raw/rfc`, preserve valid cached files, and rerun the same profile. |
| Corpus is smaller than configured maximum | Rejections, deduplication, balancing, or source targets ended first | Use actual report totals; the global maximum is a ceiling rather than a guaranteed result. |
| Too many documents are rejected | Quality threshold or source content | Inspect rejection reasons and sample documents before changing the YAML in a new experiment. |
| Near-duplicate processing uses substantial memory | SimHash state is memory resident | Use the bounded 50M profile for the course path or plan sufficient memory for the larger build. |
| Output path already contains another build | Artifact identity is ambiguous | Preserve the existing file and choose a new configuration/output path rather than overwriting evidence. |

## Tokenizer training

| Symptom | Check | Response |
| --- | --- | --- |
| Corpus file not found | Incorrect profile output path | Use the exact path from the corpus YAML and report. |
| Tokenizer training stops | Empty/invalid corpus, dependency issue, or resource limit | Inspect several JSONL records, confirm `--format jsonl`, and preserve the error before retrying. |
| Verification round trip is unexpected | Byte-level segmentation or corrupted tokenizer artifact | Compare encoded IDs and decoded text; do not proceed if decoding fails to reconstruct the verification text. |
| Existing tokenizer would be overwritten | A different corpus used the same output path | Use the corpus-specific tokenizer path documented by the course. |
| Same vocabulary size is assumed compatible | Token IDs were learned independently | Use only the exact tokenizer that encoded the checkpoint's corpus. |

## Encoding and splitting

| Symptom | Check | Response |
| --- | --- | --- |
| Encoding cannot load tokenizer | Path is wrong or tokenizer artifact is invalid | Return to tokenizer verification and use the corpus-specific path. |
| Training or validation token count is zero | Corpus has too few valid documents or split/input failed | Inspect `encoding_report.json`, source corpus records, and validation fraction. |
| Files are unexpectedly large | Corpus produced more tokens than estimated | Use report counts to plan storage and training; do not interrupt a valid write without reason. |
| Train and validation content may overlap | Split process or custom data changed | Confirm document-level splitting and split seed; rebuild into a new output directory if identity is uncertain. |
| Token ID exceeds vocabulary during training | Token files and tokenizer differ | Pair the token directory with the tokenizer recorded in its encoding report. |

## CPU training

| Symptom | Check | Response |
| --- | --- | --- |
| Training is slow | Model iteration, examples, batch, background load, and CPU | Use Iteration 1 and bounded limits for learning; estimate larger work from measured throughput. |
| System becomes unresponsive | Batch, memory pressure, or competing applications | Stop safely, close competing work, reduce batch size, and resume only with a documented plan. |
| Loss is `nan` or `inf` | Data, learning rate, gradients, or numeric problem | Stop, retain output and checkpoint, inspect the triggering settings and batches before retrying. |
| No validation appears | Validation token path/count or lifecycle settings | Confirm startup `validation_examples` and the held-out token file. |
| Run duration differs from expectation | Example limits or steps were not checked | Use startup `training_batches` and `maximum_training_steps` as authoritative. |

## CUDA training

| Symptom | Check | Response |
| --- | --- | --- |
| CUDA out of memory | Peak allocated memory and batch size | Reduce `--batch-size` first, record the changed step count, and verify with a bounded fresh run. |
| BF16 unsupported | GPU capability and PyTorch build | Use `--precision fp16` or `--precision fp32` and record the fallback. |
| FP16 becomes unstable | Non-finite loss, gradient scaling, and learning rate | Stop, preserve evidence, compare a bounded FP32 run, and inspect gradients and data. |
| GPU utilization is low | Batch size, workers, transfer, validation, saving, or other process | Measure one bottleneck at a time instead of adding workers automatically. |
| Throughput falls over time | Thermals, power, competing process, or storage | Correlate throughput with temperature, clocks, utilization, validation, and checkpoint saves. |
| Fused AdamW is not active | Installed PyTorch/device does not support it | Accept the standard AdamW fallback and record `fused_adamw`; functionality remains. |

## Checkpoints

| Symptom | Check | Response |
| --- | --- | --- |
| Save appears frozen | Checkpoint size and storage speed | Wait for `Checkpoint saved:`; do not close the terminal or press `Ctrl+C` again. |
| No `last_checkpoint.pt` exists | BuildLLM stores save reason in metadata and writes numbered step files | Use the exact `Checkpoint saved:` or `verified_checkpoint` path. |
| `best_checkpoint.pt` is missing | No completed validation established a best checkpoint | Use the verified numbered checkpoint printed by the trainer. |
| Checkpoint cannot load | Partial/corrupt file, wrong model profile, or missing path | Try a prior verified numbered checkpoint and preserve the failed file for diagnosis. |
| Disk is nearly full | Temporary and final saves require headroom | Stop safely before the next save, protect existing files, and create adequate storage capacity. |
| Older numbered checkpoint disappeared | Retention removed it after newer saves | Use a backup policy for rollback points that must outlive automatic retention. |

## Resume

| Symptom | Check | Response |
| --- | --- | --- |
| No checkpoint available | Wrong directory or no numbered save | Select the correct experiment directory or pass an explicit existing path. |
| Model configuration mismatch | Iteration or architecture setting changed | Restore the original command; start a new experiment for changed architecture. |
| Restored step is not expected | Latest selection found a different numbered file | Inspect paths and use `--resume <exact path>`. |
| Incomplete epoch repeats data | Sampler cursor and RNG states are not checkpointed | Document the limitation; this is recoverable but not bit-for-bit continuation. |
| Resume uses a different tokenizer or corpus | Experiment identity changed | Stop and restore the original tokenizer and token paths. |
| Scheduler behavior differs | Epoch target, batch, or example count changed | Restore the original training command and planned epoch target. |

## Generation

| Symptom | Check | Response |
| --- | --- | --- |
| Checkpoint path does not exist | Best checkpoint absent or directory differs | Use the exact verified path from training output. |
| Token IDs are invalid or text is corrupted | Tokenizer mismatch | Use the tokenizer associated with that checkpoint's training token files. |
| Output loops | Undertraining, concentrated logits, or decoder settings | Preserve the output, compare fixed settings and checkpoints, then change one control. |
| Output is incoherent | Small/undertrained model, high temperature, data weakness, or mismatch | Verify identity, return to conservative settings, and compare loss evidence. |
| Output ends early | End token, stop text, maximum new tokens, or context limit | Read `Finish reason` and `Generated tokens`. |
| Output includes invented facts | Expected next-token model failure | Mark the claim as unverified and include it in evaluation evidence. |
| Repeating a seed changes results across devices | Kernel and numeric behavior differ | Record the device and treat the seed as an aid rather than a cross-device guarantee. |

## A disciplined response sequence

1. Stop safely if continued execution risks the run or data.
2. Copy the full command, error, startup JSON, and last visible progress.
3. Identify which artifact boundary failed.
4. Verify inputs without overwriting them.
5. Change one variable.
6. Use a new output or checkpoint directory when the experiment identity changed.
7. Repeat a bounded check.
8. Record the cause and response in the run worksheet.

Return to the [course home](README.md), the [glossary](GLOSSARY.md), or the [run-record worksheet](RUN_RECORD_WORKSHEET.md).
