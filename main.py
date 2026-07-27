"""
Project initialization.

The purpose of this program is to verify that the local development
environment is working correctly before introducing model code.

Every issue discovered here is an environment problem.
Every issue discovered later should be an application problem.
"""

import platform
import sys

import torch


def main() -> None:
    """Display runtime information."""

    print()
    print("Build LLM")
    print("=" * 40)

    print(f"Python Version : {sys.version.split()[0]}")
    print(f"Operating System : {platform.system()}")
    print(f"Processor : {platform.processor()}")

    print()

    print("PyTorch")
    print("=" * 40)

    print(f"Version : {torch.__version__}")
    print(f"CUDA Available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Device : {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    print()
    print("Environment verification completed successfully.")


if __name__ == "__main__":
    main()