import json
from pathlib import Path

import numpy as np
import torch

from .vector_addition import embedding, layer_norm, linear, transformer_block


class GPT2CustomModel:
    def __init__(self, weights: dict[str, torch.Tensor], config: dict[str, int]):
        self.weights = weights
        self.config = config
        self.zero_bias = torch.zeros(config["n_vocab"], dtype=torch.float32)
        self.lm_head_weights = (
            weights["transformer.wte.weight"].transpose(0, 1).contiguous()
        )

    @classmethod
    def from_reference(cls, reference_dir: str | Path) -> "GPT2CustomModel":
        reference = Path(reference_dir)
        weights_dir = reference / "weights"
        manifest = json.loads((weights_dir / "manifest.json").read_text())
        config = json.loads((reference / "config.json").read_text())
        weights = {}

        for name, metadata in manifest.items():
            values = np.fromfile(weights_dir / metadata["file"], dtype=np.float32)
            weights[name] = torch.from_numpy(values.reshape(metadata["shape"]).copy())

        return cls(weights, config)

    def _weight(self, name: str) -> torch.Tensor:
        return self.weights[name]

    def forward(
        self,
        token_ids: torch.Tensor,
        capture_blocks: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        token_ids = token_ids.to(dtype=torch.int32, device="cpu").contiguous()
        sequence_length = token_ids.numel()
        hidden = embedding(
            token_ids,
            self._weight("transformer.wte.weight"),
            self._weight("transformer.wpe.weight"),
        )
        block_outputs = []

        for layer in range(self.config["n_layer"]):
            prefix = f"transformer.h.{layer}"
            qkv_weights = self._weight(f"{prefix}.attn.c_attn.weight")
            qkv_bias = self._weight(f"{prefix}.attn.c_attn.bias")
            query_weights, key_weights, value_weights = qkv_weights.chunk(3, dim=1)
            query_bias, key_bias, value_bias = qkv_bias.chunk(3)

            hidden = transformer_block(
                hidden,
                self._weight(f"{prefix}.ln_1.weight"),
                self._weight(f"{prefix}.ln_1.bias"),
                query_weights,
                query_bias,
                key_weights,
                key_bias,
                value_weights,
                value_bias,
                self._weight(f"{prefix}.attn.c_proj.weight"),
                self._weight(f"{prefix}.attn.c_proj.bias"),
                self._weight(f"{prefix}.ln_2.weight"),
                self._weight(f"{prefix}.ln_2.bias"),
                self._weight(f"{prefix}.mlp.c_fc.weight"),
                self._weight(f"{prefix}.mlp.c_fc.bias"),
                self._weight(f"{prefix}.mlp.c_proj.weight"),
                self._weight(f"{prefix}.mlp.c_proj.bias"),
                self.config["n_head"],
                self.config["n_inner"],
                1e-5,
            )
            if capture_blocks:
                block_outputs.append(hidden.clone())

        hidden = layer_norm(
            hidden,
            self._weight("transformer.ln_f.weight"),
            self._weight("transformer.ln_f.bias"),
            1e-5,
        )
        logits = linear(hidden, self.lm_head_weights, self.zero_bias)
        return logits, block_outputs
