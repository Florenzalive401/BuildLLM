# Learning Lab Changes

This copy was created from the codebase supplied while the 100 million parameter model was training in epoch 8 of 18.

## Added

1. One repository course covering three model iterations
2. An 8.1 million parameter CPU first model
3. The existing 42.1 million parameter model as Iteration 2
4. The existing 98.5 million parameter model as Iteration 3
5. Device specific runtime defaults inside each JSON model profile
6. Automatic CPU or CUDA selection
7. FP32 CPU execution for every model
8. BF16 or FP16 CUDA execution where supported
9. One `run_lab.py` command for the entire course
10. Fourteen course lessons covering the complete artifact chain, model evolution, recovery, and evidence-based comparison
11. A plain-language glossary, run-record worksheet, and stage-based troubleshooting guide

## Ordered course workflow

The course now begins with repository download and environment setup, then
creates every required artifact in order:

1. Simple Wikipedia source
2. 50M-character Wikipedia, RFC, and FineWeb Edu corpus
3. Matching 32K tokenizer
4. Separate training and validation token files
5. Iteration 1 checkpoint and generated samples
6. Controlled Iteration 2 and Iteration 3 comparison runs on the same 50M data
7. Balanced-corpus Iteration 2 larger-data training
8. 800M-corpus Iteration 3 larger-data training

`run_lab.py` accepts `--checkpoint-directory` so course experiments can use
separate checkpoint folders. Omitting the option preserves the original
`checkpoints/iteration_<n>` behavior.

## Compatibility with the active 100 million parameter run

The Iteration 3 architecture remains unchanged at 720 embedding dimensions, 12 layers, 12 attention heads, 2880 feed forward dimensions, and sequence length 256. Its CUDA runtime remains batch size 16 with BF16. Existing checkpoints remain compatible when the same tokenizer, token files, epoch target, and checkpoint directory are supplied.

Do not replace files in the directory of the currently running process. Use this repository copy for the learning lab, or stop the active run only after its checkpoint save completes.
