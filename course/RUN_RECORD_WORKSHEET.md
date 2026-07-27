# BuildLLM Run-Record Worksheet

Copy this worksheet before every training experiment. Fill values from configuration files, reports, startup JSON, epoch output, final JSON, checkpoints, and generated samples. Use `<not run>` or `<not available>` instead of leaving an ambiguous blank.

## Experiment question

**What specific question is this run intended to answer?**

`<your question>`

**Experiment path:** `<controlled comparison / larger-data training / pipeline verification / other>`

**What is the one main variable you intend to study?**

`<your variable>`

## Artifact identity

| Field | Value |
| --- | --- |
| Date and start time | `<value>` |
| Repository revision or workspace identifier | `<value>` |
| Full command | `<paste command>` |
| Corpus YAML | `<path>` |
| Corpus JSONL | `<path>` |
| Corpus report | `<path>` |
| Accepted documents | `<report value>` |
| Accepted characters | `<report value>` |
| Source contributions | `<report values>` |
| Exact duplicates removed | `<report value>` |
| Near duplicates removed | `<report value>` |
| Tokenizer | `<path>` |
| Vocabulary size | `<value>` |
| Encoding report | `<path>` |
| Training token file | `<path>` |
| Validation token file | `<path>` |
| Training documents | `<encoding report value>` |
| Validation documents | `<encoding report value>` |
| Training tokens | `<encoding report value>` |
| Validation tokens | `<encoding report value>` |
| Split seed | `<encoding report value>` |

## Model and runtime

| Field | Value |
| --- | --- |
| Iteration | `<1 / 2 / 3>` |
| Model profile | `<path>` |
| Parameter count | `<startup value>` |
| Sequence length | `<startup value>` |
| Embedding dimension | `<startup value>` |
| Layers | `<startup value>` |
| Attention heads | `<startup value>` |
| Feed-forward dimension | `<startup value>` |
| CPU | `<model>` |
| GPU | `<model or none>` |
| GPU memory | `<value>` |
| PyTorch version | `<value>` |
| Device | `<startup value>` |
| Precision | `<startup value>` |
| Tensor Cores enabled | `<startup value>` |
| TF32 enabled | `<startup value>` |
| Fused AdamW | `<startup value>` |
| Batch size | `<resolved value>` |
| Workers | `<resolved value>` |
| Peak accelerator memory | `<measured value>` |

## Planned training work

| Field | Value |
| --- | --- |
| Epoch target | `<value>` |
| Training example limit | `<value; 0 means all>` |
| Validation example limit | `<value; 0 means all>` |
| Resolved training examples | `<startup value>` |
| Resolved validation examples | `<startup value>` |
| Training batches per epoch | `<startup value>` |
| Maximum training steps | `<startup value>` |
| Learning rate | `<value>` |
| Weight decay | `<value>` |
| Estimated steps per second | `<bounded measurement>` |
| Estimated tokens per second | `<bounded measurement>` |
| Estimated duration | `<calculation and allowance>` |
| Checkpoint directory | `<path>` |
| Free storage before run | `<value>` |
| Backup destination | `<path or policy>` |

## Epoch record

| Epoch | Training loss | Validation loss | Ending learning rate | Global step | Processed tokens | Throughput | Elapsed time | Checkpoint path | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<path>` | `<observation>` |
| 2 | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<path>` | `<observation>` |
| 3 | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<path>` | `<observation>` |

Add rows for additional epochs.

## Completion or interruption

| Final JSON field | Value |
| --- | --- |
| `training_complete` | `<value>` |
| `interrupted` | `<value>` |
| `completed_epochs` | `<value>` |
| `global_step` | `<value>` |
| `best_validation_loss` | `<value>` |
| `verified_checkpoint` | `<path>` |
| `restored_epoch` | `<value>` |
| `restored_checkpoint_type` | `<value>` |

If resumed:

| Resume field | Value |
| --- | --- |
| `resumed_from` | `<path>` |
| `restored_epoch` | `<value>` |
| `restored_global_step` | `<value>` |
| `checkpoint_type` | `<value>` |
| Settings compared with original | `<confirmation or difference>` |

## Fixed generation protocol

| Setting | Value |
| --- | --- |
| Checkpoint | `<path>` |
| Tokenizer | `<path>` |
| Device | `<value>` |
| Maximum new tokens | `<value>` |
| Greedy | `<true / false>` |
| Temperature | `<value>` |
| Top-k | `<value>` |
| Top-p | `<value>` |
| Repetition penalty | `<value>` |
| No-repeat n-gram size | `<value>` |
| Seed | `<value>` |
| Stop settings | `<value>` |

## Generated evidence

| Prompt | Full unedited output location | Finish reason | Generated tokens | Relevance | Grammar | Repetition | Factual issues |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| The purpose of a firewall is | `<paste or link>` | `<value>` | `<value>` | `<observation>` | `<observation>` | `<observation>` | `<observation>` |
| When a software service fails in production, | `<paste or link>` | `<value>` | `<value>` | `<observation>` | `<observation>` | `<observation>` | `<observation>` |
| In the history of computing, | `<paste or link>` | `<value>` | `<value>` | `<observation>` | `<observation>` | `<observation>` | `<observation>` |
| Water changes from a liquid to | `<paste or link>` | `<value>` | `<value>` | `<observation>` | `<observation>` | `<observation>` | `<observation>` |
| The evidence does not prove that | `<paste or link>` | `<value>` | `<value>` | `<observation>` | `<observation>` | `<observation>` | `<observation>` |

## Conclusion

**Did the engineering pipeline work?**

`<evidence-based answer>`

**What did training and validation show?**

`<evidence-based answer>`

**What did generation show?**

`<evidence-based answer>`

**What can this experiment support as a conclusion?**

`<specific conclusion>`

**What can this experiment not support?**

`<limitation>`

**What is the next experiment, and which evidence justifies it?**

`<one-variable experiment and reason>`

Return to the [course home](README.md) or the [model comparison lesson](13_COMPARE_THE_MODELS.md).
