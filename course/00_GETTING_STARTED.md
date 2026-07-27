# Getting Started

## What you will learn

You will download BuildLLM, create the one Python environment used by the entire project, install the correct PyTorch build for your hardware, and verify that Python can see the CPU or GPU you intend to use.

## Where you are in the build

```text
Repository
    -> Python environment  <--- you are here
    -> downloaded source documents
    -> corpus
    -> tokenizer and token files
    -> training
    -> checkpoint and generation
```

Nothing has been trained yet. The goal of this lesson is a reliable foundation. Environment problems discovered during a long training run are much more expensive than environment problems discovered now.

## Before you begin

You need:

- Git installed and available from your terminal;
- Python installed;
- permission to create files and install Python packages;
- an internet connection;
- enough disk space for the repository, virtual environment, downloaded data, token files, and checkpoints;
- Windows PowerShell, a Linux shell, or macOS Terminal.

You do not need a GPU for Iteration 1. Iterations 2 and 3 can run short demonstrations on CPU, but practical full-data training expects a compatible NVIDIA GPU.

## Files you should already have

No BuildLLM artifact is required yet. You need Git, Python, a terminal, and permission to create the repository directory and install packages.

## Files this lesson will create

| Path | Purpose |
| --- | --- |
| `<repository directory>` | The BuildLLM source code, course, and configuration files |
| `.venv/` | The single isolated Python environment used by every project command |

The virtual environment prevents BuildLLM packages from being mixed with unrelated global Python packages. Corpus building, tokenizer training, encoding, model training, tests, and generation all use this same environment.

## Download the repository

Windows PowerShell:

```powershell
git clone https://github.com/DrDeathLabs/BuildLLM.git
cd BuildLLM
```

Linux or macOS:

```bash
git clone https://github.com/DrDeathLabs/BuildLLM.git
cd BuildLLM
```

You are at the repository root when the current directory contains at least:

```text
run_lab.py
train.py
generate_text.py
configs/
course/
src/
```

> **Stop and check:** Do not continue from your Downloads directory, home directory, or another project. Every later command assumes the current directory is the BuildLLM repository root.

## Create the Python environment

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

The activation command changes which Python executable receives package installations. Your prompt will usually show `(.venv)` while the environment is active.

When you return to the project later, open a terminal at the repository root and activate the same environment again. Do not create a new environment for the GPU lessons or data tools.

## Confirm which Python is active

Windows PowerShell:

```powershell
Get-Command python
python --version
```

Linux or macOS:

```bash
which python
python --version
```

The reported Python path should be inside `.venv`. If it is not, stop and activate the environment before installing anything.

## Install PyTorch

PyTorch provides tensors, GPU execution, automatic differentiation, neural-network layers, and optimizers. It is installed separately from `requirements.txt` because CPU and CUDA machines need different PyTorch builds.

Open the official selector:

<https://pytorch.org/get-started/locally/>

Choose:

1. the stable PyTorch release;
2. your operating system;
3. Pip;
4. Python;
5. CPU or the CUDA version supported by the machine.

Run the command produced by the selector while `.venv` is active.

The current project workstation used this CUDA 12.8 command:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

That command is an example, not a universal choice. A CPU-only machine should use the CPU command from the selector. A different NVIDIA driver or platform may require a different CUDA wheel.

## Install the remaining packages

Windows PowerShell:

```powershell
pip install -r requirements.txt
```

Linux or macOS:

```bash
pip install -r requirements.txt
```

The requirements include corpus-download tools, Hugging Face datasets, YAML support, the tokenizer library, progress displays, numerical packages, and pytest. They do not install a second training framework.

## Verify the environment

Windows PowerShell:

```powershell
python main.py
```

Linux or macOS:

```bash
python main.py
```

Successful output has this structure:

```text
Build LLM
========================================
Python Version : <your Python version>
Operating System : <your operating system>
Processor : <your processor>

PyTorch
========================================
Version : <your PyTorch version>
CUDA Available : <True or False>
Device : <your GPU name>

Environment verification completed successfully.
```

A CPU system prints `Running on CPU` instead of a GPU device name. That is a valid result for the first model.

## What success looks like

You are ready to continue when:

- `.venv` is active;
- `python` resolves to the environment inside this repository;
- `python main.py` imports PyTorch successfully;
- the operating system and Python version are reported;
- CUDA availability matches your intended hardware;
- the final line says `Environment verification completed successfully.`

## What to record

Record your repository location, Python version, PyTorch version, operating system, CUDA availability, and detected device in the [run-record worksheet](RUN_RECORD_WORKSHEET.md).

## Stop and check

Do not continue to a long GPU run if you expected CUDA but `main.py` reports `CUDA Available : False`. Iteration 1 can still run on CPU, but a missing CUDA runtime must be corrected before Iterations 2 or 3.

Do not install another copy of PyTorch globally to work around an environment problem. Confirm the active Python path first.

## Common problems and exact responses

| Problem | Likely cause | What to do |
| --- | --- | --- |
| `git` is not recognized | Git is not installed or not on `PATH` | Install Git, close and reopen the terminal, then run `git --version` |
| `py` is not recognized on Windows | Python launcher is unavailable | Use `python -m venv .venv` if `python --version` works |
| PowerShell blocks `Activate.ps1` | Local execution policy prevents scripts | Review the local PowerShell execution policy; do not create a second environment |
| `No module named torch` | PyTorch was installed into another Python environment | Activate `.venv`, confirm `Get-Command python`, and reinstall the selected PyTorch build |
| CUDA is false on an NVIDIA machine | CPU-only PyTorch wheel, driver issue, or unsupported hardware | Check the NVIDIA driver and reinstall the appropriate CUDA wheel from the PyTorch selector |
| `requirements.txt` is not found | Terminal is outside the repository root | Change to the cloned repository directory and confirm the expected files exist |
| Package installation is denied | Permissions, proxy, or endpoint security | Use the approved organizational package process or proxy; do not disable security controls without authorization |

Use the [troubleshooting guide](TROUBLESHOOTING.md) for a longer environment decision tree.

## Under the hood

`main.py` does not train a model. It imports PyTorch, asks PyTorch whether CUDA is available, and prints the first CUDA device name when one is visible. This is a dependency and hardware check.

The `.venv` directory contains the Python interpreter links and installed packages used by the repository. The model code remains in `src/`; the environment contains dependencies, not a second copy of the application.

## Check your understanding

Before continuing, answer:

1. Why is PyTorch installed separately from the other requirements?
2. What does activating `.venv` change?
3. Is `CUDA Available : False` an error for the Iteration 1 CPU lesson?
4. Which command confirms that PyTorch can see the intended hardware?
5. From which directory should every course command run?

## Next lesson

Next: [AI and LLM foundations](01_AI_AND_LLM_FOUNDATIONS.md).
