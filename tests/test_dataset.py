"""Tests for the memory-mapped tokenized dataset."""
import tempfile

import numpy as np
import torch

from terse.data.dataset import TokenizedDataset, build_dataloader


def test_tokenized_dataset(tmp_path):
    seq_len = 8
    tokens = np.arange(64, dtype=np.uint32)
    data_path = tmp_path / "tokens.bin"
    tokens.tofile(data_path)

    ds = TokenizedDataset(str(data_path), seq_len=seq_len)
    assert len(ds) == 8
    sample = ds[3]
    assert "input_ids" in sample
    assert sample["input_ids"].shape == (seq_len,)
    assert sample["input_ids"][0].item() == 24


def test_build_dataloader(tmp_path):
    seq_len = 16
    tokens = np.arange(128, dtype=np.uint32)
    data_path = tmp_path / "tokens.bin"
    tokens.tofile(data_path)

    loader = build_dataloader(str(data_path), seq_len, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, seq_len)
    assert batch["input_ids"].dtype == torch.int64
