# Understand CPU and GPU Execution

## What you will learn

In this lesson you will learn how the same BuildLLM model and trainer run on a central processing unit (CPU) or a CUDA-capable graphics processing unit (GPU), how numeric precision affects speed and memory, how to respond to out-of-memory errors, and how to estimate a long run from measured throughput.

## Where you are in the build

```text
one training framework
        |
        +-> CPU runtime: accessible, FP32, best for Iteration 1 and bounded demonstrations
        |
        +-> CUDA runtime: parallel, BF16 or FP16 where supported, intended for Iterations 2 and 3
```

Device selection changes where tensors are processed and which runtime optimizations are available. It does not select a simplified tutorial model.

## Before you begin

Run from the repository root with the virtual environment active. Complete the environment verification in the getting-started lesson. If `main.py` reports `CUDA Available: False`, use the CPU route until the PyTorch installation and GPU driver are corrected.

Do not start a second long training job while another valuable run is using the same GPU or checkpoint directory.

## Files you should already have

```text
configs/models/gpt_first_cpu.json
configs/models/gpt_42m.json
configs/models/gpt_100m.json
data/tokens/learning_50m/train_tokens.pt
data/tokens/learning_50m/validation_tokens.pt
tokenizer/learning_50m_tokenizer.json
```

## Files this lesson will create

If you perform the optional bounded device comparison, it creates:

```text
checkpoints/iteration_1_cpu_comparison
checkpoints/iteration_1_cuda_comparison
```

Use different directories because they are separate runs even when other settings match.

## Key ideas in plain language

### CPU and GPU execute the same model

The CPU is a general-purpose processor with a small number of powerful cores. A GPU contains many execution units designed for highly parallel operations such as the matrix multiplications used throughout a transformer.

Iteration 1 is small enough to teach the full training process on CPU. Iterations 2 and 3 still support CPU execution for bounded demonstrations, but meaningful runs are intended for CUDA because their matrix operations and training state are much larger.

### Device resolution

```text
--device auto
    |
    +-- CUDA is available -> use CUDA and the model profile's GPU defaults
    |
    +-- CUDA unavailable --> use CPU and the model profile's CPU defaults
```

Use `--device cpu` when you intentionally want CPU. Use `--device cuda` when a GPU is required; the application fails clearly rather than silently switching to CPU.

### Precision changes representation, memory traffic, and supported kernels

| Precision | Meaning | Course use |
| --- | --- | --- |
| FP32 | 32-bit floating point | Default CPU format and a diagnostic CUDA option |
| BF16 | 16-bit bfloat with an FP32-like exponent range | Preferred larger-model CUDA format when supported |
| FP16 | 16-bit IEEE half with a smaller exponent range | CUDA fallback that uses gradient scaling |

The parameter count does not change with precision. A 98.5M-parameter model remains a 98.5M-parameter model, but storing and moving selected values in 16 bits can reduce memory traffic and enable faster hardware paths.

### Tensor Cores and TF32 are acceleration features

Tensor Cores are specialized GPU units for matrix operations in supported formats and dimensions. TensorFloat-32, or TF32, accelerates supported FP32 matrix operations on newer NVIDIA hardware while retaining FP32 storage and exponent range with reduced multiplication precision.

`train.py` configures high float32 matrix-multiplication precision and permits TF32 on CUDA backends. The startup JSON tells you whether Tensor Core and TF32 paths are enabled. Enabled does not mean every operation uses the same kernel.

### Fused AdamW can reduce optimizer overhead

Fused AdamW combines optimizer operations into fewer GPU kernel launches where the installed PyTorch build supports it. BuildLLM requests the fused implementation on CUDA and falls back to standard AdamW if construction is unsupported.

Read the startup field `fused_adamw` instead of assuming it is active.

### Memory contains more than model weights

Training memory includes model weights, gradients, AdamW optimizer state, activations saved for backpropagation, temporary operation buffers, and loaded batches. Checkpoints use storage for many of the same persistent states.

Activation memory grows with batch size, sequence length, model width, and layer count. Attention score memory grows approximately with the square of sequence length. A model that fits for generation may still fail to fit during training because generation does not keep the same backward-pass state.

### Batch size is the first practical memory control

Reducing batch size usually lowers activation memory without changing architecture or checkpoint tensor shapes. It also changes examples per optimizer step, steps per epoch, gradient noise, throughput, and the newly constructed learning-rate schedule.

Record the actual batch size. Do not call two runs equal-budget comparisons when one processed a different number of tokens or optimizer steps.

### Workers and pinned memory affect data delivery

CUDA data loaders can pin host memory so transfers to the GPU are more efficient. Worker processes can prepare batches concurrently, but more workers are not automatically faster, especially on Windows or when the token tensor is already in memory.

The included profiles default to zero workers for predictable behavior. Change `--workers` only as a measured experiment.

## Run the lab

### Step 1: verify the environment

```powershell
python main.py
```

The command is the same on Linux and macOS.

Record `PyTorch Version`, `CUDA Available`, and `Device`.

### Step 2: inspect the intended runtime without training

PowerShell:

```powershell
python run_lab.py `
  --iteration 2 `
  --device cuda `
  --epochs 1 `
  --training-examples 500 `
  --validation-examples 100 `
  --train-tokens data/tokens/learning_50m/train_tokens.pt `
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --checkpoint-directory checkpoints/iteration_2_device_check `
  --dry-run
```

Linux and macOS:

```bash
python run_lab.py \
  --iteration 2 \
  --device cuda \
  --epochs 1 \
  --training-examples 500 \
  --validation-examples 100 \
  --train-tokens data/tokens/learning_50m/train_tokens.pt \
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt \
  --tokenizer tokenizer/learning_50m_tokenizer.json \
  --checkpoint-directory checkpoints/iteration_2_device_check \
  --dry-run
```

If CUDA is unavailable, use `--device cpu` for the bounded demonstration. Do not remove the example limits.

### Optional measured CPU and CUDA comparison

Run this only when you want to measure both devices and no other valuable training run is active. The commands intentionally use Iteration 1 and bounded data.

PowerShell:

```powershell
python run_lab.py `
  --iteration 1 `
  --device cpu `
  --precision fp32 `
  --batch-size 4 `
  --epochs 1 `
  --training-examples 500 `
  --validation-examples 100 `
  --train-tokens data/tokens/learning_50m/train_tokens.pt `
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --checkpoint-directory checkpoints/iteration_1_cpu_comparison

python run_lab.py `
  --iteration 1 `
  --device cuda `
  --precision fp32 `
  --batch-size 4 `
  --epochs 1 `
  --training-examples 500 `
  --validation-examples 100 `
  --train-tokens data/tokens/learning_50m/train_tokens.pt `
  --validation-tokens data/tokens/learning_50m/validation_tokens.pt `
  --tokenizer tokenizer/learning_50m_tokenizer.json `
  --checkpoint-directory checkpoints/iteration_1_cuda_comparison
```

Linux and macOS use the same arguments with backslash line continuations.

## What the command is doing

`run_lab.py` resolves the requested device, loads the model profile's runtime defaults, and passes explicit overrides to `train.py`. `train.py` moves the model and batches to that device, resolves precision, configures CUDA backends when applicable, and reports the actual runtime in startup JSON.

## What success looks like

Before starting a large run, the resolved output must show the intended device. A CUDA training startup should also make `precision`, `tensor_cores_enabled`, `tf32_enabled`, and `fused_adamw` visible.

For a device comparison, both runs must use the same model, token files, tokenizer, example limits, batch size, precision, and epoch count. Compare elapsed time and throughput only after confirming those controls.

Estimate a longer run from measurement:

```text
estimated seconds = planned optimizer steps / measured optimizer steps per second
```

or:

```text
estimated seconds = planned processed tokens / measured tokens per second
```

Add checkpoint, validation, startup, and normal system variability. A short measurement is an estimate, not a promised completion time.

## Stop and check

Do not start Iteration 2 or 3 full training until:

- CUDA resolves correctly for the intended GPU route;
- the selected precision is supported;
- a bounded run fits in memory;
- you recorded batch size and measured throughput;
- estimated runtime and checkpoint storage are acceptable;
- the computer has stable power, cooling, and free disk space;
- the checkpoint directory is unique.

## Common problems and exact responses

| Problem | Likely cause | Exact response |
| --- | --- | --- |
| `CUDA Available: False` | CPU-only PyTorch, missing driver, unsupported GPU, or environment mismatch | Confirm the virtual environment, driver, and PyTorch CUDA build before using `--device cuda`; use CPU only for bounded work. |
| CUDA out of memory | Batch and activations exceed available GPU memory | Stop, reduce `--batch-size`, record the change, and rerun from a fresh directory unless deliberately resuming an identical schedule. |
| BF16 is unsupported | GPU generation lacks BF16 training support | Use `--precision fp16` with the application's gradient scaling or `--precision fp32`, then record the fallback. |
| GPU utilization is low | Data delivery, small batches, another workload, or frequent non-compute work limits the GPU | Measure workers, batch size, transfer time, validation, and checkpoint overhead one variable at a time. |
| More workers make training slower or unstable | Process startup and data-loader overhead exceed the benefit | Return to `--workers 0`, especially on Windows, and retain the measurement. |
| Throughput decreases over time | Thermal throttling, power limits, competing workload, or storage delay | Check temperatures, clocks, other GPU processes, checkpoint timing, and data-loader behavior. |
| A model fits for generation but not training | Backpropagation and optimizer states require more memory | Reduce training batch size or model workload; do not infer training capacity from inference alone. |

## What to record

Record processor, GPU model, GPU memory, PyTorch version, CUDA availability, device, precision, Tensor Core status, TF32 status, fused AdamW status, batch size, workers, sequence length, parameter count, maximum steps, measured tokens per second, measured steps per second, peak memory, checkpoint save time, and estimated total runtime.

## Under the hood

Read `src/runtime.py` for device and precision resolution, `src/model_profiles.py` for CPU and GPU profile defaults, `src/datamodule.py` for workers and pinned memory, `src/training/optimizer.py` for fused AdamW fallback, and `train.py` for CUDA backend configuration.

Use this decision guide:

| Goal | Recommended route |
| --- | --- |
| Learn the complete pipeline without a GPU | Iteration 1 on CPU |
| Prove Iteration 2 or 3 can execute on CPU | Small bounded example counts in FP32 |
| Train Iteration 2 meaningfully | CUDA, normally BF16 |
| Train Iteration 3 meaningfully | CUDA, normally BF16, with long-run planning |
| Diagnose lower-precision instability | Controlled FP32 run |
| Compare CPU and GPU throughput | Same model, data, batch, precision, limits, and epoch count |

## Check your understanding

1. Why does lower precision not change parameter count?
2. What additional memory is required during training beyond model weights?
3. Why does lowering batch size affect optimizer steps per epoch?
4. What does `fused_adamw` tell you?
5. Why should runtime be estimated from processed tokens or steps rather than a model name?
6. Why is an unbounded Iteration 3 CPU run technically supported but usually impractical?
7. Which variables must remain fixed for a fair CPU and GPU throughput comparison?

You are ready to continue when you can select a device deliberately, interpret the startup runtime fields, recover from an out-of-memory error, and estimate a larger run from measured evidence.

## Next lesson

Next: [Train and compare Iteration 2](10_ITERATION_TWO.md).
