from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    name: str
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    head_dim: int
    vocab_size: int
    max_seq_len: int
    dtype: str = "bf16"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @property
    def kv_hidden_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads != self.num_heads

    @property
    def gqa_groups(self) -> int:
        return self.num_heads // self.num_kv_heads


@dataclass
class BenchConfig:
    model: str                          # path to model yaml
    backends: list[str]
    prompt_lens: list[int]
    output_lens: list[int]
    batch_sizes: list[int]
    num_warmup: int = 5
    num_runs: int = 20
    dtype: str = "fp32"                 # override model default for kernel compat
    output_dir: str = "benchmarks/results"

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
