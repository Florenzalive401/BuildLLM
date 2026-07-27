"""
FineWeb Edu ingestion.

The complete dataset is much larger than this project requires. Streaming
allows the corpus builder to process one document at a time and stop when
the selected corpus size has been reached.
"""

from collections.abc import Iterator

from datasets import load_dataset


DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIGURATION = "sample-10BT"
DATASET_SPLIT = "train"

SHUFFLE_SEED = 42
SHUFFLE_BUFFER_SIZE = 10_000


def stream_fineweb_documents() -> Iterator[dict]:
    """Stream FineWeb Edu documents in shuffled order."""

    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIGURATION,
        split=DATASET_SPLIT,
        streaming=True,
    )

    shuffled_dataset = dataset.shuffle(
        seed=SHUFFLE_SEED,
        buffer_size=SHUFFLE_BUFFER_SIZE,
    )

    for record in shuffled_dataset:
        yield record