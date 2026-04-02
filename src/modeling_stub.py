from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - optional dependency for Phase 1
    torch = None
    nn = None


@dataclass(frozen=True)
class ModelConfig:
    numeric_dim: int
    categorical_cardinalities: dict[str, int]
    text_vocab_size: int = 32768
    text_embedding_dim: int = 64
    categorical_embedding_dim: int = 16
    hidden_dim: int = 128
    dropout: float = 0.1


def resolve_device(prefer_gpu: bool = True) -> str:
    if torch is None:
        return "cpu"
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchUnavailableError(RuntimeError):
    pass


if nn is not None:
    class TabularMLP(nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            total_cat_dim = len(config.categorical_cardinalities) * config.categorical_embedding_dim
            self.embeddings = nn.ModuleDict(
                {
                    name: nn.Embedding(cardinality, config.categorical_embedding_dim)
                    for name, cardinality in config.categorical_cardinalities.items()
                }
            )
            self.encoder = nn.Sequential(
                nn.Linear(config.numeric_dim + total_cat_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
            )
            self.output = nn.Linear(config.hidden_dim, 1)

        def forward(self, numeric_inputs: torch.Tensor, categorical_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
            categorical_vectors = [self.embeddings[name](values) for name, values in categorical_inputs.items()]
            features = torch.cat([numeric_inputs, *categorical_vectors], dim=-1)
            encoded = self.encoder(features)
            return self.output(encoded).squeeze(-1)


    class TextTabularFusionModel(nn.Module):
        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.tabular = TabularMLP(config)
            self.text_embedding = nn.EmbeddingBag(
                config.text_vocab_size,
                config.text_embedding_dim,
                mode="mean",
            )
            self.fusion = nn.Sequential(
                nn.Linear(config.hidden_dim + config.text_embedding_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(
            self,
            numeric_inputs: torch.Tensor,
            categorical_inputs: dict[str, torch.Tensor],
            text_indices: torch.Tensor,
            text_offsets: torch.Tensor,
        ) -> torch.Tensor:
            tabular_hidden = self.tabular.encoder(
                torch.cat(
                    [
                        numeric_inputs,
                        *[
                            self.tabular.embeddings[name](values)
                            for name, values in categorical_inputs.items()
                        ],
                    ],
                    dim=-1,
                )
            )
            text_hidden = self.text_embedding(text_indices, text_offsets)
            fused = torch.cat([tabular_hidden, text_hidden], dim=-1)
            return self.fusion(fused).squeeze(-1)
else:
    class TabularMLP:  # pragma: no cover - exercised only when torch is unavailable
        def __init__(self, config: ModelConfig) -> None:
            raise TorchUnavailableError("PyTorch is not installed. Install torch to use TabularMLP.")


    class TextTabularFusionModel:  # pragma: no cover - exercised only when torch is unavailable
        def __init__(self, config: ModelConfig) -> None:
            raise TorchUnavailableError("PyTorch is not installed. Install torch to use TextTabularFusionModel.")
