import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from tokenizers import Tokenizer

from src.config import config
from src.embeddings import TokenEmbedding


DEFAULT_TOKENIZER_FILE = Path("tokenizer/tokenizer.json")
DEFAULT_OUTPUT_FILE = Path("artifacts/embedding_projection.png")
DEFAULT_TOKEN_COUNT = 150


def load_tokenizer(path: Path) -> Tokenizer:
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")

    return Tokenizer.from_file(str(path))


def load_embedding(
    checkpoint_path: Path | None,
) -> TokenEmbedding:
    layer = TokenEmbedding()

    if checkpoint_path is None:
        return layer

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    embedding_key = "input_embedding.token_embedding.embedding.weight"

    if embedding_key not in state_dict:
        raise KeyError(
            f"Embedding weights were not found under {embedding_key}"
        )

    layer.embedding.weight.data.copy_(
        state_dict[embedding_key]
    )

    return layer


def select_tokens(
    tokenizer: Tokenizer,
    limit: int,
) -> list[tuple[int, str]]:
    vocabulary = tokenizer.get_vocab()

    tokens = [
        (token_id, token)
        for token, token_id in vocabulary.items()
        if token.isprintable()
        and not token.startswith("<|")
        and len(token.strip()) >= 2
    ]

    tokens.sort(key=lambda item: item[0])

    return tokens[:limit]


def project_embeddings(
    embedding: TokenEmbedding,
    selected_tokens: list[tuple[int, str]],
) -> torch.Tensor:
    token_ids = torch.tensor(
        [token_id for token_id, _ in selected_tokens],
        dtype=torch.long,
    )

    vectors = embedding(token_ids).detach().cpu().numpy()

    projection = PCA(
        n_components=2,
        random_state=42,
    ).fit_transform(vectors)

    return torch.from_numpy(projection)


def create_plot(
    projection: torch.Tensor,
    selected_tokens: list[tuple[int, str]],
    output_file: Path,
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(14, 10),
    )

    x_values = projection[:, 0].numpy()
    y_values = projection[:, 1].numpy()

    axis.scatter(
        x_values,
        y_values,
        alpha=0.75,
    )

    for index, (_, token) in enumerate(selected_tokens):
        axis.annotate(
            token,
            (
                x_values[index],
                y_values[index],
            ),
            fontsize=7,
            alpha=0.8,
        )

    axis.set_title("Token Embedding Projection")
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    axis.grid(
        visible=True,
        alpha=0.2,
    )

    figure.tight_layout()
    figure.savefig(
        output_file,
        dpi=200,
    )

    plt.show()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER_FILE,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
    )

    parser.add_argument(
        "--token-count",
        type=int,
        default=DEFAULT_TOKEN_COUNT,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.token_count < 2:
        raise ValueError("token-count must be at least 2")

    if arguments.token_count > config.vocabulary_size:
        raise ValueError(
            "token-count exceeds the configured vocabulary size"
        )

    tokenizer = load_tokenizer(arguments.tokenizer)

    embedding = load_embedding(
        arguments.checkpoint
    )

    selected_tokens = select_tokens(
        tokenizer,
        arguments.token_count,
    )

    if len(selected_tokens) < 2:
        raise ValueError(
            "Not enough eligible tokens were found"
        )

    projection = project_embeddings(
        embedding,
        selected_tokens,
    )

    create_plot(
        projection,
        selected_tokens,
        arguments.output,
    )

    print(f"Saved visualization: {arguments.output}")


if __name__ == "__main__":
    main()