# Contributing to BuildLLM

BuildLLM is both a working language-model training system and an engineering course. Contributions must preserve the shared implementation, configuration-driven model differences, CPU support, and the Iteration 3 reference path.

## Before opening a pull request

1. Create and activate one Python 3.11 virtual environment.
2. Install the correct PyTorch build for your machine.
3. Install `requirements.txt`.
4. Run `python main.py`.
5. Run the complete `pytest` suite.

Do not commit corpora, tokenizer files, token tensors, checkpoints, virtual environments, run outputs, or verification logs.

## Pull requests

- Keep changes focused and explain the engineering reason for them.
- Add or update tests for behavior changes.
- Update the course when a public command, path, output field, or workflow changes.
- Keep course prose paragraphs on one physical Markdown line for clean WordPress pasting.
- Preserve backward compatibility unless the pull request clearly documents and justifies a breaking change.
- Include the commands you ran and their results.

## Reporting problems

Use the bug-report form for application defects and the course-feedback form for unclear, missing, or incorrect learning material. Report security vulnerabilities privately as described in `SECURITY.md`.

By contributing source code, you agree that it may be distributed under the MIT License. By contributing course or educational content, you agree that it may be distributed under CC BY-NC-SA 4.0.
